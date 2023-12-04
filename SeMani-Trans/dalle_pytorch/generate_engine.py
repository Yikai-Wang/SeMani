import torch
import os
from einops import repeat
from torchvision.utils import make_grid, save_image
from pathlib import Path
# import moxing as mox
from dalle_pytorch.tokenizer import tokenizer
from collections import defaultdict
from tqdm import tqdm


def moving_files(dir_name, src_path, tgt_path):
    import moxing as mox
    mox.file.copy_parallel(
        os.path.join(src_path, dir_name),
        os.path.join(tgt_path, dir_name)
    )


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text_folder, text_len, tokenizer, truncate_captions, search_txt_suffix):
        super(TextDataset, self).__init__()
        path = Path(text_folder)
        text_files = [*path.glob("*{}".format(search_txt_suffix))]
        text_files = {text_file.stem: text_file for text_file in text_files}

        descriptions = []
        for text_file, text_file_path in text_files.items():
            dataset = text_file.split('_')[0]

            with open(text_file_path, 'r') as f:
                descriptions.extend([(line.strip('\n'), dataset, i) for i, line in enumerate(f)])
        self.descriptions = list(filter(lambda t: len(t[0]) > 0, descriptions))
        self.text_len = text_len
        self.tokenizer = tokenizer
        self.truncate_captions = truncate_captions

    def __getitem__(self, item):
        result = self.descriptions[item]
        sentence, dataset, img_id = result
        tokens = self.tokenizer.tokenize(
            sentence,
            self.text_len,
            truncate_text=self.truncate_captions
        ).squeeze(0)
        return tokens, sentence, dataset, img_id

    def __len__(self):
        return len(self.descriptions)


@torch.no_grad()
def generate_image_from_list(args, text_list, text_id_list, vae_model, model, step, folder_prefix):
    model.eval()

    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    RANK = int(os.environ.get("RANK", 0))
    base_split_size = len(text_list) // WORLD_SIZE
    if RANK != WORLD_SIZE - 1:
        current_text_list = text_list[RANK*base_split_size: (RANK+1)*base_split_size]
        current_text_id_list = text_id_list[RANK*base_split_size: (RANK+1)*base_split_size]
    else:
        current_text_list = text_list[RANK*base_split_size:]
        current_text_id_list = text_id_list[RANK * base_split_size:]

    text_tokens = tokenizer.tokenize(current_text_list, args.text_seq_len, truncate_text=args.truncate_captions)
    text_tokens = text_tokens.to(torch.cuda.current_device()).contiguous()

    """ Bucket Version """
    text_id = 0
    from tqdm import tqdm
    for text_chunk in tqdm(text_tokens.split(args.batch_size)):
        if hasattr(model, 'module'):
            output = model.module.generate_images(vae_model, text_chunk, filter_thres=0.9)
        else:
            output = model.generate_images(vae_model, text_chunk, filter_thres=0.9)

        # save all images
        for _, out in enumerate(output):
            sentence = current_text_list[text_id]

            outputs_dir = Path(args.outputs_dir) / "CLIP_rank" / sentence.replace(' ', '_')[:(100)] / \
                          '{}{}'.format(folder_prefix, step)
            outputs_dir.mkdir(parents=True, exist_ok=True)
            save_image(out, outputs_dir / f'{current_text_id_list[text_id]}.jpg', normalize=True)

            text_id += 1

    assert text_id == len(current_text_list) == text_tokens.size(0) == len(current_text_id_list)
    if args.move_results_to_s3:
        moving_files('/'.join(str(outputs_dir).split('/')[1:]), '..', args.s3_path)

    model.train()


@torch.no_grad()
def generate_image_from_loader(loader, vae_model, model, save_path, folder_prefix, filter_thres, subset):

    model.eval()

    for i, (text, images, original_text, image_name) in enumerate(tqdm(loader)):
        # inputs
        text = text.cuda()

        if hasattr(model, 'module'):
            outputs = model.module.generate_images(vae_model, text, filter_thres=filter_thres)
        else:
            outputs = model.generate_images(vae_model, text, filter_thres=filter_thres)  # ddp: dalle_pytorch.dalle_pytorch.DALLE

        assert len(outputs) == len(image_name)
        for k, out in enumerate(outputs):
            # save all images
            outputs_dir = Path(save_path) / \
                          '{}_{}_{}'.format(folder_prefix, filter_thres, subset)
            outputs_dir.mkdir(parents=True, exist_ok=True)

            save_image(out, outputs_dir / '{}.jpg'.format(image_name[k].split('.')[0]), normalize=True)


# @torch.no_grad()
# def generate_image_from_loader(loader, vae_model, model, step, save_path, s3_path, folder_prefix):
#
#     model.eval()
#     # from tqdm import tqdm
#     appear_dataset = []
#     description_dict = defaultdict(list)
#     RANK = int(os.environ.get("RANK", 0))
#
#     for i, (tokens, sentence, dataset, img_id) in enumerate(loader):  # 8 hours
#         appear_dataset.extend(dataset)
#
#         tokens = tokens.to(torch.cuda.current_device())
#         # if args.fp16 and images.dim() == 4:
#         #     images = images.half()
#
#         if hasattr(model, 'module'):
#             outputs = model.module.generate_images(vae_model, tokens, filter_thres=0.9)
#         else:
#             outputs = model.generate_images(vae_model, tokens, filter_thres=0.9)  # ddp: dalle_pytorch.dalle_pytorch.DALLE
#
#         assert len(outputs) == len(img_id)
#         for k, out in enumerate(outputs):
#             # save all images
#             outputs_dir = Path(save_path) / \
#                           'FID_IS_{}'.format(dataset[k].upper()) / \
#                           '{}{}'.format(folder_prefix, step)
#             outputs_dir.mkdir(parents=True, exist_ok=True)
#
#             description_dict[dataset[k]].append('{}.jpg {}\n'.format(img_id[k], sentence[k]))
#             save_image(out, outputs_dir / '{}.jpg'.format(img_id[k]), normalize=True)
#
#     # save descriptions
#     for ad in appear_dataset:
#         description_path = Path(save_path) / 'text_descriptions' / '{}{}'.format(folder_prefix, step) / ad
#         description_path.mkdir(parents=True, exist_ok=True)
#         with open(str(description_path / 'text_descriptions_rank_{}.txt'.format(RANK)), 'w') as f:
#             f.writelines(description_dict[ad])
#
#     if s3_path is not None:
#         import moxing as mox
#         appear_dataset = list(set(appear_dataset))
#         # base_path = '/'.join(str(save_path).split('/')[1:])
#         for ad in appear_dataset:
#             current_dataset_path = str(Path(save_path) / 'FID_IS_{}/{}{}'.format(ad.upper(), folder_prefix, step))
#             mox.file.copy_parallel(
#                 current_dataset_path,
#                 os.path.join(s3_path, 'FID_IS_{}/{}{}'.format(ad.upper(), folder_prefix, step))
#             )
#             # moving_files(current_dataset_path, '..', s3_path)
#
#         # text
#         mox.file.copy_parallel(
#             str(Path(save_path) / 'text_descriptions' / '{}{}'.format(folder_prefix, step)),
#             os.path.join(s3_path, 'text_descriptions', '{}{}'.format(folder_prefix, step))
#         )
#
#     model.train()

