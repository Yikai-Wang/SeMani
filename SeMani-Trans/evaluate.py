import argparse
from pathlib import Path
import os
import random
import numpy as np
import torch
import math
from functools import partial

from dalle_pytorch.vae import OpenAIDiscreteVAE, VQGanVAE

from dalle_pytorch import DALLE_CONTEXT, DALLE
from dalle_pytorch import distributed_utils
from dalle_pytorch.loader import *  # TextImageDataset
from dalle_pytorch.tokenizer import tokenizer, HugTokenizer, ChineseTokenizer, YttmTokenizer

from einops import repeat
import torchvision
from torchvision.utils import save_image
from tqdm import tqdm
import cv2
import copy


def make_colors():
    from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
    colors = []
    for cate in COCO_CATEGORIES:
        colors.append(cate["color"])
    return colors
COLORS = make_colors()

def mask_to_boundary(mask, dilation_ratio=0.0008):
    """
	Convert binary mask to boundary mask.
	:param mask (numpy array, uint8): binary mask
	:param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
	:return: boundary mask (numpy array)
	"""
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
        # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


# argument parsing

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=False)
parser.add_argument('--dataset', type=str, required=True,
                   help='a prompt label to compute the similarity')
parser.add_argument('--sim_threshold', type=float, default=0.163,
                   help='a prompt label to compute the similarity')
parser.add_argument('--without_context', action='store_true',
                   help='output path')
parser.add_argument('--outputs_dir', type=str, required=True,
                   help='output path')
parser.add_argument('--filter_thres', type=float, required=True,
                   help='output path')
parser.add_argument('--seed', type=int, default=110,
                   help='seed')
parser.add_argument('--img_size', type=int, default=256,
                   help='input image size')
parser.add_argument('--vae_path', type=str,
                   help='path to your trained discrete VAE')
parser.add_argument('--dalle_path', type=str, required=True,
                   help='path to your partially trained DALL-E')
parser.add_argument('--vqgan_model_path', type=str, default = None,
                   help='path to your trained VQGAN weights. This should be a .ckpt file. (only valid when taming option is enabled)')
parser.add_argument('--vqgan_config_path', type=str, default = None,
                   help='path to your trained VQGAN config. This should be a .yaml file. (only valid when taming option is enabled)')
parser.add_argument('--restrict_data_len', type=int, default=None,
                    help='training number')
parser.add_argument('--val_folder', type=str, default = "/home/work/user-job-dir/dataset/CUB-200",
                    help='folder of cub dataset')
parser.add_argument('--truncate_captions', dest='truncate_captions', action='store_true',
                    help='Captions passed in which exceed the max token length will be truncated if this is set.')
parser.add_argument('--random_resize_crop_lower_ratio', dest='resize_ratio', type=float, default=0.75,
                    help='Random resized crop lower ratio')
parser.add_argument('--chinese', dest='chinese', action='store_true')
parser.add_argument('--taming', dest='taming', action='store_true')
parser.add_argument('--hug', dest='hug', action='store_true')
parser.add_argument('--bpe_path', type=str,
                    help='path to your BPE json file')
parser.add_argument('--dalle_output_file_name', type=str, default = "dalle",
                    help='output_file_name')
parser.add_argument('--fp16', action='store_true',
                    help='(experimental) - Enable DeepSpeed 16 bit precision. Reduces VRAM.')
parser.add_argument('--amp', action='store_true',
	help='Apex "O1" automatic mixed precision. More stable than 16 bit precision. Can\'t be used in conjunction with deepspeed zero stages 1-3.')
parser.add_argument('--wandb_name', default='dalle_train_transformer',
                    help='Name W&B will use when saving results.\ne.g. `--wandb_name "coco2017-full-sparse"`')
parser.add_argument('--exp_result_path', default='dalle_experiments', help='path for storing the experiment results')
parser.add_argument('--save_token', action='store_true', help='save tokens for similar experiment')
parser.add_argument('--ema_model', action='store_true', help='exponentially weighted iterate averaging model via a CPU version')
parser.add_argument('--generate_freq', default=20, type=int, help='exponentially weighted iterate averaging model via a CPU version')
parser.add_argument('--clip_model_name', type=str, help='name of your pre-trained CLIP')
parser.add_argument('--clip_loss_weight', default = 0., type = float, help = 'CLIP loss weight')
parser.add_argument('--clip_loss_type', default = "Global", type = str, choices=["Global"])
parser.add_argument('--clip_loss_method', default = "abs", type = str, choices=["abs", "ce"])
parser.add_argument('--gumbel_tau', default = 1., type = float, help = 'temperature for gumbel softmax')
parser.add_argument('--straight_through', action='store_true', help = 'gumbel softmax relax to a hard distribution')
parser.add_argument('--inf_subset', default='val', type=str)
parser.add_argument('--text', default=None, type=str)
parser = distributed_utils.wrap_arg_parser(parser)

train_group = parser.add_argument_group('Training settings')
train_group.add_argument('--flops_profiler', dest = 'flops_profiler', action='store_true', help = 'Exits after printing detailed flops/runtime analysis of forward/backward')
train_group.add_argument('--epochs', default = 20, type = int, help = 'Number of epochs')
train_group.add_argument('--save_every_n_steps', default = 1000, type = int, help = 'Save a checkpoint every n steps')
train_group.add_argument('--keep_n_checkpoints', default = None, type = int, help = '(Careful) Deletes old deepspeed checkpoints if there are more than n')
train_group.add_argument('--batch_size', default = 4, type = int, help = 'Batch size')
train_group.add_argument('--ga_steps', default = 1, type = int, help = 'Number of steps to accumulate gradients across per each iteration. DeepSpeed only.')
train_group.add_argument('--learning_rate', default = 3e-4, type = float, help = 'Learning rate')
train_group.add_argument('--resume_learning_rate', default = None, type = float, help = 'Resume Learning rate')
train_group.add_argument('--weight_decay', default = 0.0, type = float, help = 'Weight decay')
train_group.add_argument('--warm_up_iters', default = 5000, type = int, help = 'the number of warm up iterations')
train_group.add_argument('--warm_up_factor', default = 0.1, type = float, help = 'the starting lr scaling for warm up')
train_group.add_argument('--lr_decay_patience', default = 50000, type = int, help = 'patience')
train_group.add_argument('--LAMB', default = False, action='store_true', help = 'LAMB optimizer')
train_group.add_argument('--adam_beta1', type=float, default=0.9)
train_group.add_argument('--adam_beta2', type=float, default=0.96)
train_group.add_argument('--clip_grad_norm', default = 4., type = float, help = 'Clip gradient norm')
train_group.add_argument('--lr_decay', default = None, type = str, choices = ["iter", "epoch"])
train_group.add_argument('--ff_dropout', default = 0.0, type = float, help = 'Feed forward dropout.')
train_group.add_argument('--attn_dropout', default = 0.0, type = float, help = 'Feed forward dropout.')
train_group.add_argument('--val_text',
                         default=["this colorful bird has a yellow breast, with a black crown and a black cheek patch",
                                  "this bird is tiny with greenish fur and feathers",
                                  "a woman and a man standing next to a bush bench.",
                                  "a bathroom with two sinks, a cabinet and a bathtub."
                                  ],
                         type = str, nargs = '+')
train_group.add_argument('--prefix', type = str, default = '')
train_group.add_argument('--num_images', type = int, default = 1)
train_group.add_argument('--generate_folder', type = str, default = None)
train_group.add_argument('--generate_text_suffix', type = str, default = '1000_test_sample.txt')

model_group = parser.add_argument_group('Model settings')
model_group.add_argument('--network_pipe', default = "base_edit", type = str, choices = ['base_edit', 'base_generate'],
    help = 'Model forward pipeline')
model_group.add_argument('--dim', default = 512, type = int, help = 'Model dimension')
model_group.add_argument('--text_seq_len', default = 128, type = int, help = 'Text sequence length')
model_group.add_argument('--depth', default = 2, type = int, help = 'Model depth')
model_group.add_argument('--heads', default = 8, type = int, help = 'Model number of heads')
model_group.add_argument('--dim_head', default = 64, type = int, help = 'Model head dimension')
model_group.add_argument('--reversible', dest = 'reversible', action='store_true')
model_group.add_argument('--loss_img_weight', default = 7, type = int, help = 'Image loss weight')
model_group.add_argument('--attn_types', default = 'full', type = str, help = 'comma separated list of attention types. attention type can be: full or sparse or axial_row or axial_col or conv_like.')
args = parser.parse_args()


def strip_path_str(file_path):
    if os.path.basename(file_path).strip(' ') == '': 
        file_path = file_path.strip(' ')[:-1] 
    return file_path

def exists(val):
    return val is not None

def get_trainable_params(model):
    return [params for params in model.parameters() if params.requires_grad]

def group_opt_params(model, weight_decay, skip_list=('ln', 'bn', 'bias', 'scale', 'emb', 'norm'), return_name=False):
    def is_in_skip_list(name, skipping_list):
        for key_word in skipping_list:
            if key_word in name:
                return True
        return False

    opt_params = []
    if skip_list is None:
        skip_list = []

    if return_name:
        name_list = []
        layer_list = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if is_in_skip_list(name, skip_list):
            opt_params.append({"params": param, "weight_decay": 0.0})
        else:
            opt_params.append({"params": param, "weight_decay": weight_decay})

        if return_name:
            name_list.append(name)
            if not is_in_skip_list(name, ['bias', 'scale']):
                layer_list.append(name)


    if return_name:
        return opt_params, (name_list, layer_list)
    else:
        return opt_params

def cp_path_to_dir(cp_path, tag):
    """Convert a checkpoint path to a directory with `tag` inserted.
    If `cp_path` is already a directory, return it unchanged.
    """
    if not isinstance(cp_path, Path):
        cp_path = Path(cp_path)
    if cp_path.is_dir():
        return cp_path
    path_sans_extension = cp_path.parent / cp_path.stem
    cp_dir = Path(f'{path_sans_extension}-{tag}-cp')
    return cp_dir


# constants
DALLE_OUTPUT_FILE_NAME = args.dalle_output_file_name + ".pt"
VAE_PATH = args.vae_path
VQGAN_MODEL_PATH = None
VQGAN_CONFIG_PATH = None

DALLE_PATH = args.dalle_path
RESUME = exists(DALLE_PATH) or exists(VAE_PATH)

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

LEARNING_RATE = args.learning_rate
RESUME_LEARNING_RATE = args.resume_learning_rate
GRAD_CLIP_NORM = args.clip_grad_norm
LR_DECAY = args.lr_decay
SAVE_EVERY_N_STEPS = args.save_every_n_steps
KEEP_N_CHECKPOINTS = args.keep_n_checkpoints

WEIGHT_DECAY = args.weight_decay

MODEL_DIM = args.dim
TEXT_SEQ_LEN = args.text_seq_len
DEPTH = args.depth
HEADS = args.heads
DIM_HEAD = args.dim_head
REVERSIBLE = args.reversible
LOSS_IMG_WEIGHT = args.loss_img_weight
FF_DROPOUT = args.ff_dropout
ATTN_DROPOUT = args.attn_dropout

ATTN_TYPES = tuple(args.attn_types.split(','))

DATA_LEN = args.restrict_data_len
EXP_PATH = args.exp_result_path

SIM_THRESHOLD = args.sim_threshold

SEED = args.seed
# set seed
torch.backends.cudnn.benchmark = False
# deterministic
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True

# Now TORCH DDP
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = args.local_rank
if WORLD_SIZE > 1:
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(LOCAL_RANK)
    IS_LOG = LOCAL_RANK == 0
else:
    IS_LOG = True
print(f"WORLD_SIZE: {WORLD_SIZE}  rank: {RANK}  gpu: {LOCAL_RANK}  IS_LOG: {IS_LOG}")

# tokenizer

if exists(args.bpe_path):
    klass = HugTokenizer if args.hug else YttmTokenizer
    tokenizer = klass(args.bpe_path)
elif args.chinese:
    tokenizer = ChineseTokenizer()


# models
if RESUME:
    if exists(VAE_PATH):
        vae_loaded = torch.load(str(VAE_PATH), map_location='cpu')
        vae_weights = vae_loaded["state_dict"]
        if exists(DALLE_PATH):
            dalle_loaded = torch.load(str(DALLE_PATH), map_location='cpu')
        else:
            dalle_loaded = torch.load(str(vae_loaded["pretrained_dalle_model_path"]), map_location="cpu")

        resume_epoch = vae_loaded.get('epoch', 0)
        resume_global_iter = vae_loaded.get('global_step', -1)

        if args.ema_model or 'cpu_model' in vae_loaded:
            cpu_vae_model = vae_loaded.get('cpu_model')
    else:
        dalle_path = Path(DALLE_PATH)
        assert dalle_path.exists(), 'DALL-E model file does not exist'
        dalle_loaded = torch.load(str(dalle_path), map_location="cpu")

        resume_epoch = dalle_loaded.get('epoch', 0)
        resume_global_iter = dalle_loaded.get('global_step', -1)

        if args.ema_model or 'cpu_model' in dalle_loaded:
            cpu_dalle_model = dalle_loaded.get('cpu_model')
    dalle_params, vae_params, dalle_weights = dalle_loaded['hparams'], dalle_loaded['vae_params'], dalle_loaded['state_dict']

    if vae_params is not None:
        raise RuntimeError
    else:
        if args.taming:
            vae = VQGanVAE(VQGAN_MODEL_PATH, VQGAN_CONFIG_PATH)
        else:
            vae = OpenAIDiscreteVAE()

    vae.image_size = args.img_size

    dalle_params = dict(
        **dalle_params
    )
    TEXT_ON_VISUAL_CONTEXT = dalle_loaded.get('text_on_visual_context', False)
else:
    raise RuntimeError
# configure OpenAI VAE for float16s

if isinstance(vae, OpenAIDiscreteVAE) and args.fp16:
    vae.enc.blocks.output.conv.use_float16 = True
from dalle_pytorch.dalle_pytorch import set_requires_grad
vae = vae.cuda().eval()
set_requires_grad(vae, False)
if exists(VAE_PATH):
    vae.load_state_dict(vae_weights)
    print('*' * 20, 'VAE MODEL STATES HAVE BEEN LOADED', '*' * 20)

# CLIP
clip_model = None
clip_img_preprocess = None
clip_txt_tokenize = None
patch_num = 16  # 224 // 14

# Entity Segmentation
import sys

sys.path.append('./EntitySeg')

from detectron2.checkpoint import DetectionCheckpointer
from train_net import setup, Trainer

args.config_file = "coco_val_b120_entity_swin_lw7_3x.yaml"
args.eval_only = True
args.opts = []
args.resume = False
cfg = setup(args)
entityseg_model = Trainer.build_model(cfg)
entityseg_model.eval()
DetectionCheckpointer(entityseg_model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    cfg.MODEL.WEIGHTS, resume=args.resume
)
set_requires_grad(entityseg_model, False)

if args.network_pipe == "base_generate":
    dalle = DALLE(vae=vae, **dalle_params)
    assert args.without_context
elif args.network_pipe == "base_edit":
    dalle = DALLE_CONTEXT(vae=vae, **dalle_params)
else:
    raise NotImplementedError

dalle = dalle.cuda()
if RESUME:
    print('*'*20, 'MODEL STATES HAVE BEEN LOADED', '*'*20)
    dalle.load_state_dict(dalle_weights, strict=False)
else:
    raise NotImplementedError

if args.amp and not args.fp16:
    opt_level = 'O1'
elif args.amp and args.fp16:
    opt_level = 'O2'
elif not args.amp and args.fp16:
    opt_level = 'O3'
else:
    opt_level = 'O0'

from apex import amp
[dalle, vae, clip_model, entityseg_model] = amp.initialize(
    [dalle, vae, clip_model, entityseg_model],
    patch_torch_functions=False,
    opt_level=opt_level,
    loss_scale="dynamic"  
)

# TODO. INSTANCE MASK SETTING
max_instance_num = 5
instance_confidence = 0.0

instances_masks = None

# TODO. Example Loder
IMG_SIZE = args.img_size
EXAMPLE_FILE = "{}/example_filenames.txt".format(args.dataset)
# "example_filenames.txt"
DATASET_BASE_FOLDER = args.dataset
IMAGE_TRANSFORMS = T.Compose([
    T.Scale(IMG_SIZE),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor()
])
SEG_TRANSFORMS = T.Compose([
            T.Scale(IMG_SIZE),
            T.CenterCrop(IMG_SIZE)
        ])
seg_size = vae.image_size // 16
def segment_preprocess_after_the_same_image_transformation(type, out_size):
    if type == 'resize':
        seg_preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize(out_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),  
    ])  # TODO. Resize + CenterCrop
    elif type == 'maxpool':
        seg_preprocess = partial(torch.nn.functional.adaptive_max_pool2d, output_size=out_size)
    elif type == 'nothing':
        seg_preprocess = lambda x: x
    else:
        raise NotImplementedError
    return seg_preprocess
seg_method = "resize"  
SEG_PROCESS = segment_preprocess_after_the_same_image_transformation(seg_method, seg_size)
SEG_DIRECT_PROCESS = segment_preprocess_after_the_same_image_transformation('resize', seg_size)
from PIL import Image


@torch.no_grad()
def example_loader_v2(instances_masks, instance_confidence, save_image=False, save_path=None):
    with open(EXAMPLE_FILE, 'r') as f:
        text_files = [line.strip('\n') for line in f]
    text_files = list(filter(lambda t: len(t) > 0, text_files))
    for name in text_files:
        img_name = name.replace("text", "images")
        cls_name = img_name.split('/')[-2]
        img_path = os.path.join(DATASET_BASE_FOLDER, img_name+'.jpg')
        img_basename = os.path.basename(img_path)
        image = IMAGE_TRANSFORMS(Image.open(img_path))
        if save_image:
            img_for_paste = image.clone().permute(1, 2, 0).numpy()
            img_for_paste = (img_for_paste[:, :, ::-1] * 255).astype(np.uint8)
            color_mask = copy.deepcopy(img_for_paste)
            masks_edge = np.zeros(color_mask.shape[:2], dtype=np.uint8)
            alpha = 0.4
            count = 0

        # Segmentation
        assert image.size(0) == 3
        input_seg = [{"height": image.size(-2), "width": image.size(-1), "image_id": 0, "file_name": img_path,
                      "image": (torch.from_numpy(image.numpy()[::-1].copy()) * 255.).to(torch.uint8)}]
        output_seg = entityseg_model(input_seg)
        assert len(output_seg) == 1
        instances = output_seg[0]["instances"]
        num_instance = len(instances)
        if num_instance == 0:
            raise NotImplemented
        scores = instances.scores.tolist()
        img_instances_mask = instances.pred_masks
        assert img_instances_mask.size(0) == num_instance

        seg_list = []
        for iim_id, iim in enumerate(img_instances_mask):
            seg = iim.cpu()
            if iim_id > 0 and scores[iim_id] < instance_confidence:
                continue
            seg = SEG_TRANSFORMS(seg.unsqueeze(0)).squeeze(0)
            seg_list.append(seg)
            if save_image:
                color_mask[seg.numpy() == 1] = COLORS[count]
                boundary = mask_to_boundary((seg.numpy() == 1).astype(np.uint8))
                masks_edge[boundary > 0] = 1
                count += 1

            if len(seg_list) >= max_instance_num:
                break
        # The size of the larger image edge is 256, which corresponds to the same resized original image.
        # Because of the image transformation of resize the shorter edge to 256 and center crop,
        # we need to transform the segmentation mask use the same way.
        seg_list = torch.stack(seg_list)
        seg_instance_num = seg_list.size(0)
        if save_image:
            img_wm = cv2.addWeighted(img_for_paste, alpha, color_mask, 1 - alpha, 0)
            img_wm[masks_edge == 1] = 0
            fvis = img_wm  
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(os.path.join(save_path, f'mask_{img_basename[:-4]}.png'), fvis)

        with open(os.path.join(DATASET_BASE_FOLDER, name+'.txt'), "r") as f:
            for sent_id, line in enumerate(f):  # entity1: guide1 | entity2: guide2
                if len(line.strip()) == 0:
                    continue

                descriptions = []
                entities = []

                guides = line.strip().split('|')
                guide_num = len(guides)
                for guide in guides:
                    try:
                        entity, gui = map(lambda x: x.strip(), guide.split(':'))
                    except ValueError:
                        print('LLL')
                        import pdb; pdb.set_trace()
                    entities.append(entity)
                    descriptions.append(gui)

                tokenized_text = tokenizer.tokenize(
                    descriptions,
                    TEXT_SEQ_LEN,
                    truncate_text=args.truncate_captions
                ) 

                yield tokenized_text, image, img_basename, cls_name, seg_list, seg_instance_num, \
                      (descriptions, entities, guide_num, sent_id)


def instance_confidence_computation(similarity, instance_num, instance_mask, method, obj_token_order_id_list):
    assert method in ["vote", "entire_sim", "token_sim"]
    assert (similarity == -9).sum() >= 1, "Similarity Matrix needs to be masked before computation confidence " \
                                          "or change the number(-9) to check in this assertion"
    img_seq = similarity.size(1)
    assert img_seq % 2 == 0, 'similarity must be excluded BOS before'
    clip_img_grid_size = int(math.sqrt(img_seq))

    if method == "token_sim":
        specific_token_sims = similarity[:, :, obj_token_order_id_list]  # (TXT_BSZ, IMG_TOKENS)
        if specific_token_sims.dim() == 3:
            specific_token_sims = specific_token_sims.max(dim=-1)[0]
            assert specific_token_sims.dtype == similarity.dtype
        specific_token_sims = specific_token_sims.view(specific_token_sims.size(0), clip_img_grid_size, clip_img_grid_size)
        obj_sims = specific_token_sims
    elif method == "entire_sim":
        text_sims = similarity.max(dim=-1)[0]  # (TXT_BSZ, IMG_TOKENS)
        text_sims = text_sims.view(text_sims.size(0), clip_img_grid_size, clip_img_grid_size)
        obj_sims = text_sims

    sims_list = []
    for in_id in range(instance_num):
        cur_instance_0pad = instance_mask[in_id]
        cur_instance_sims = obj_sims[..., cur_instance_0pad]  # (TXT_BSZ, ...)
        cur_instance_sims = cur_instance_sims.mean(dim=-1)  # (TXT_BSZ)
        sims_list.append(cur_instance_sims)
    return sims_list


def image_text_similarity(sims, instance_mask, text_tokens, prompt_token_list):
    """
    :param sims: Tensor(IMG_BSZ==1 to squeeze, TXT_BSZ, CLS_TOKEN+IMG_TOKENS, BOS_TOKEN+TEXT_TOKENS+EOS_TOKEN+PAD_TOKEN)
    :param instance_mask: Tensor(IMG_BSZ==1 to squeeze, instance_num, H, W)
    :param text_tokens: Tensor(TXT_BSZ, BOS_TOKEN+TEXT_TOKENS+EOS_TOKEN+PAD_TOKEN)
    :param prompt_token_list: LIST
    """
    assert sims.dim() == 3

    non_padding_txt_token_num = sims.size(-1) - (text_tokens == 0).sum(dim=1)  # (TXT_BSZ, )
    to_mask = (torch.arange(sims.size(-1)).expand_as(sims).to(sims)
               <
               non_padding_txt_token_num[..., None, None].expand_as(sims))
    sims.masked_fill_(~to_mask, -9.)

    token_sims = instance_confidence_computation(
        sims, instance_mask.size(0), instance_mask, "token_sim", prompt_token_list
    )
    return token_sims


def print_img_segment(original_image, instance_mask, img_name, cnt, save_path):
    patch_num = instance_mask.size(-1)
    h = w = original_image.size(-1)
    original_image = original_image.clone()
    patch_size = h // patch_num

    for row, col in zip(*torch.nonzero(instance_mask, as_tuple=True)):
        original_image[..., row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size] = 0
    torchvision.utils.save_image(original_image, os.path.join(save_path, f'{cnt}th_mask_'+img_name))


outputs_dir = Path(args.outputs_dir) / 'example' / 'Thre{}'.format(args.filter_thres)
seg_mask_outputs_dir = Path(args.outputs_dir) / 'example' / 'entity_seg'
seg_mask_outputs_dir.mkdir(parents=True, exist_ok=True)
cub_loader = example_loader_v2(instances_masks, instance_confidence, save_image=True, save_path=str(seg_mask_outputs_dir))
MASK_COLOR = torch.tensor([1., 1., 1.])


with torch.no_grad():
    idx = 0
    for i, (
            tokenized_text, images, image_name, cls_name, seg_list, seg_instance_num,
            (descriptions, entities, guide_num, sent_id)
    ) \
            in enumerate(tqdm(cub_loader)):

        for k in range(args.num_images):  # For each Image
            tokenized_text = tokenized_text.cuda()

            if images.dim() == 3:
                images = images.unsqueeze(0).cuda()  # 256
            # TODO. seg_list are candidates to be ranked
            # seg_list: 256
            seg_list = SEG_PROCESS(seg_list.float()).bool()  # (seg_num, H, W)
            # clip similarity
            clip_img = clip_img_preprocess(images)  # resize to 224
            clip_img_features = clip_model.encode_image(clip_img)  # (img_num=1, img_seq, dim)
            clip_img_features = clip_img_features / clip_img_features.norm(dim=-1,
                                                                           keepdim=True)  # (img_num=1, img_seq, dim)
            cur_outputs_dir = outputs_dir / image_name.split('.')[0]
            cur_outputs_dir.mkdir(parents=True, exist_ok=True)

            pre_manipulated_instance = None
            try:
                assert len(entities) == guide_num == tokenized_text.size(0)
            except AssertionError:
                print('....')
                import pdb; pdb.set_trace()
            has_been_modified = {}  # seg_id: {"entity": , "score":}

            pre_masked_patch = torch.zeros_like(seg_list[0])
            for en_id, cur_prompt_label in enumerate(entities):  # For each entity
                cur_tokenized_text = tokenized_text[en_id:en_id+1]
                cur_text_num = cur_tokenized_text.size(0)  # (txt_num, txt_seq)

                clip_text = clip_txt_tokenize(["a photo of a {}.".format(cur_prompt_label)]).cuda()
                clip_txt_features = clip_model.encode_text(clip_text)  # (txt_num=1, txt_seq, dim)
                clip_txt_features = clip_txt_features / clip_txt_features.norm(dim=-1, keepdim=True)  # (txt_num=1, txt_seq, dim)

                sims = torch.matmul(
                    clip_img_features,  # (1, img_seq, dim)
                    clip_txt_features.transpose(-1, -2)  # (1, dim, txt_seq)
                )  # (1, img_seq, txt_seq)

                eot_pos = torch.nonzero(clip_text == 49407)
                assert eot_pos.size(0) == 1
                assert eot_pos[0, 0] == 0
                eot_pot = eot_pos[0, 1].item()
                prompt_token_list = list(range(5, eot_pot - 1))  # -1 for the last full stop
                assert prompt_token_list[0] == 5
                sims = image_text_similarity(sims, seg_list, clip_text, prompt_token_list)  

                # Todo. RESIZE SIM FEATURE OR RESIZE SEG_LIST
                assert len(sims) == len(seg_list)
                aggregated_sims = torch.stack(sims)  # Instance_num * txt_num=1
                aggregated_sims = aggregated_sims.squeeze(1)
                assert aggregated_sims.dim() == 1

                instance_to_manipulate_cnt = 0

                for seg_id_sim, seg_id in zip(*torch.sort(aggregated_sims, descending=True)):
                    if instance_to_manipulate_cnt >= 1:
                        if seg_id_sim.item() >= SIM_THRESHOLD:
                            if cur_prompt_label in entities[en_id+1:]:  # some same entities left for editing
                                break
                            else:   # no more same entity to be manipulate
                                pass
                        else:
                            break

                    to_search_seg_id = seg_id.item()
                    if to_search_seg_id in has_been_modified:
                        if has_been_modified[to_search_seg_id]["entity"] == cur_prompt_label:
                            continue
                        elif has_been_modified[to_search_seg_id]["score"] > seg_id_sim.item():
                            continue

                    instances_mask = seg_list[to_search_seg_id]  # (MH, MW)
                    if instances_mask.sum() < 4:  # the aspect of mask is too small to search the next mask
                        continue
                    has_been_modified[to_search_seg_id] = {"entity": cur_prompt_label, "score": seg_id_sim.item()}

                    instance_to_manipulate_cnt += 1
                    print_img_segment(
                        images, instances_mask, image_name,
                        '{}_s_{}_en{}_{}'.format(seg_method, sent_id, en_id, seg_id),
                        str(cur_outputs_dir)
                    )
                    if en_id >= 1:
                        pre_masked_patch = pre_masked_patch | instances_mask
                        print_img_segment(images, pre_masked_patch, image_name,
                                          '{}_s_{}_total'.format(seg_method, sent_id, en_id, seg_id),
                                          str(cur_outputs_dir))

                    patch_size = int(images.size(-1) // instances_mask.size(-1))
                    labels = vae.get_codebook_indices(images)  # (1, MH*MW)
                    cam_mask = instances_mask.view(-1)  # (MH*MW)

                    if pre_manipulated_instance is None:
                        labels = repeat(labels, '() n -> b n', b = cur_text_num)  # (text_num, MH*MW)
                    else:
                        labels = pre_manipulated_instance
                    cam_mask = repeat(cam_mask, 'n -> b n', b = cur_text_num)  # (text_num, MH*MW)

                    if not args.without_context:
                        use_context = torch.ones(cur_tokenized_text.size(0), device=cur_tokenized_text.device, dtype=torch.bool)
                        if images.dim() == 4:
                            visual_context = T.functional.rgb_to_grayscale(images, num_output_channels=3)
                            if args.fp16:
                                visual_context = vae.get_codebook_indices(visual_context.half())
                            else:
                                visual_context = vae.get_codebook_indices(visual_context)
                            visual_context = repeat(visual_context, '() n -> b n', b = cur_text_num)

                            vc_seq = visual_context.shape[-1]
                            txt_seq = cur_tokenized_text.shape[-1]
                            seq_len = vc_seq + 2 + txt_seq + vc_seq - 1
                            visual_context_mask_0pad = ~torch.ones(seq_len, seq_len).triu_(seq_len - seq_len + 1).bool().to(
                                cur_tokenized_text.device)
                            visual_context_mask_0pad[vc_seq + 1: vc_seq + 1 + txt_seq, :vc_seq + 1] = 0  # 2:4

                            row_cam = torch.nonzero(cam_mask[0], as_tuple=True)[0]  # ()
                            for ri in row_cam[:1]:
                                visual_context_mask_0pad[
                                    vc_seq+2+txt_seq-1+ri,
                                    vc_seq+2+txt_seq:vc_seq+2+txt_seq+ri
                                ] = 0
                            visual_context_mask_0pad = visual_context_mask_0pad.unsqueeze(0)
                            if TEXT_ON_VISUAL_CONTEXT:
                                transformer_attn_kwargs = {}
                            else:
                                transformer_attn_kwargs = {"mask": visual_context_mask_0pad}
                        else:
                            raise RuntimeError
                    else:
                        visual_context = None
                        use_context = None
                        transformer_attn_kwargs = {}

                    if args.network_pipe == 'base_edit':
                        outputs, cur_manipulated_instance = dalle.generate_images(
                            vae,
                            cur_tokenized_text,  # (text_num, text_seq)
                            visual_context=visual_context,
                            context_mask_0pad=use_context,
                            transformer_attn_kwargs=transformer_attn_kwargs,
                            filter_thres=args.filter_thres,
                            labels=labels,  # (text_num, MH*MW)
                            cam_mask=cam_mask,  # (text_num, MH*MW)
                            return_img_seq=True
                        )
                    else:
                        outputs, cur_manipulated_instance = dalle.generate_images(
                            vae,
                            cur_tokenized_text,  # (text_num, text_seq)
                            filter_thres=args.filter_thres,
                            labels=labels,  # (text_num, MH*MW)
                            cam_mask=cam_mask,  # (text_num, MH*MW)
                            return_img_seq=True
                        )
                    if pre_manipulated_instance is not None:
                        assert (pre_manipulated_instance[~cam_mask] == cur_manipulated_instance[~cam_mask]).all()
                    pre_manipulated_instance = cur_manipulated_instance

            assert outputs.size(0) == 1
            save_image(outputs[0], cur_outputs_dir / f'{k}_s_{sent_id}_g{0}.png', normalize=True)
        save_image(images[0], cur_outputs_dir / 'None_s_None_SR.png', normalize=True)

