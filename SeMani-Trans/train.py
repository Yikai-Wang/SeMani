import argparse
import copy
from pathlib import Path
import time
import os
import random
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# TODO. VAEs from dalle_pytorch have been modified; the old version is saved in vae_before.py
from dalle_pytorch.vae import OpenAIDiscreteVAE, VQGanVAE

from dalle_pytorch import DALLE
from dalle_pytorch import distributed_utils
from dalle_pytorch.loader import *  # TextImageDataset
from dalle_pytorch.tokenizer import tokenizer, HugTokenizer, ChineseTokenizer, YttmTokenizer

# Adding
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from datetime import datetime
from dalle_pytorch.lr_scheduler import warm_up_lr
from dalle_pytorch.val_engine import validation

from dalle_pytorch.generate_engine import generate_image_from_loader, TextDataset, moving_files
from collections import defaultdict
from os.path import join
import json

from dalle_pytorch.clip_loss import global_clip_loss, fetch_soft_one_hot_map

# argument parsing

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=False)
parser.add_argument('--seed', type=int, default=110, help='seed')
parser.add_argument('--img_size', type=int, help='input image size')
group.add_argument('--vae_path', type=str, help='path to your trained discrete VAE')
group.add_argument('--dalle_path', type=str, help='path to your partially trained DALL-E')
parser.add_argument('--vqgan_model_path', type=str, default = None,
                   help='path to your trained VQGAN weights. This should be a .ckpt file. (only valid when taming option is enabled)')
parser.add_argument('--vqgan_config_path', type=str, default = None,
                   help='path to your trained VQGAN config. This should be a .yaml file. (only valid when taming option is enabled)')
parser.add_argument('--dataset', type=str, required=True, help='List of dataset to train, separated with comma')
parser.add_argument('--restrict_data_len', type=int, default=None, help='training number')
parser.add_argument('--text_folder', type=str, required=True,
                    help='List of paths to your folder of texts for args.dataset, separated with comma')
parser.add_argument('--img_folder', type=str, required=True,
                    help='List of paths to your folder of images for args.dataset, separated with comma')
parser.add_argument('--val_folder', type=str, default = "/home/work/user-job-dir/dataset/CUB-200",
                    help='folder of cub dataset')
parser.add_argument('--truncate_captions', dest='truncate_captions', action='store_true',
                    help='Captions passed in which exceed the max token length will be truncated if this is set.')
parser.add_argument('--random_resize_crop_lower_ratio', dest='resize_ratio', type=float, default=0.75,
                    help='Random resized crop lower ratio')
parser.add_argument('--chinese', dest='chinese', action='store_true')
parser.add_argument('--taming', dest='taming', action='store_true')
parser.add_argument('--hug', dest='hug', action='store_true')
parser.add_argument('--bpe_path', type=str, help='path to your BPE json file')
parser.add_argument('--dalle_output_file_name', type=str, default="dalle", help='output_file_name')
parser.add_argument('--fp16', action='store_true', help='(experimental) - Enable DeepSpeed 16 bit precision. Reduces VRAM.')
parser.add_argument('--amp', action='store_true',
	help='Apex "O1" automatic mixed precision. More stable than 16 bit precision. Can\'t be used in conjunction with deepspeed zero stages 1-3.')
parser.add_argument('--wandb_name', default='dalle_train_transformer',
                    help='Name W&B will use when saving results.\ne.g. `--wandb_name "coco2017-full-sparse"`')
parser.add_argument('--exp_result_path', default='dalle_experiments', help='path for storing the experiment results')
parser.add_argument('--save_token', action='store_true', help='save tokens for similar experiment')
parser.add_argument('--ema_model', action='store_true', help='exponentially weighted iterate averaging model via a CPU version')
parser.add_argument('--ema_step', default=250, type=int)
parser.add_argument('--generate_freq', default=1000000, type=int, help='exponentially weighted iterate averaging model via a CPU version')
parser.add_argument('--clip_trainer', default = "openai", type = str, choices=["openai"]) # "openai"
parser.add_argument('--clip_model_name', type=str, help='name of your pre-trained CLIP', default='ViT-B/32')
parser.add_argument('--clip_loss_weight', default = 0., type = float, help = 'CLIP loss weight')
parser.add_argument('--clip_loss_type', default = "Global", type = str, choices=["Global"])
parser.add_argument('--clip_loss_method', default = "abs", type = str, choices=["abs", "ce"])
parser.add_argument('--gumbel_tau', default = 1., type = float, help = 'temperature for gumbel softmax')
parser.add_argument('--straight_through', action='store_true', help = 'gumbel softmax relax to a hard distribution')
parser = distributed_utils.wrap_arg_parser(parser)

train_group = parser.add_argument_group('Training settings')
train_group.add_argument('--flops_profiler', dest = 'flops_profiler', action='store_true',
                         help = 'Exits after printing detailed flops/runtime analysis of forward/backward')
train_group.add_argument('--epochs', default = 20, type = int, help = 'Number of epochs')
train_group.add_argument('--save_every_n_steps', default = 1000, type = int, help = 'Save a checkpoint every n steps')
train_group.add_argument('--save_every_n_epochs', default = 100, type = int, help = 'Save a checkpoint every n epochs')
train_group.add_argument('--keep_n_checkpoints', default = None, type = int, help = '(Careful) Deletes old deepspeed checkpoints if there are more than n')
train_group.add_argument('--batch_size', default = 4, type = int, help = 'Batch size')
train_group.add_argument('--ga_steps', default = 1, type = int, help = 'Number of steps to accumulate gradients across per each iteration. DeepSpeed only.')
train_group.add_argument('--learning_rate', default = 3e-4, type = float, help = 'Learning rate')
train_group.add_argument('--resume_learning_rate', default = None, type = float, help = 'Resume Learning rate')
train_group.add_argument('--weight_decay', default = 0.0, type = float, help = 'Weight decay')
train_group.add_argument('--warm_up_iters', default = 5000, type = int, help = 'the number of warm up iterations')
train_group.add_argument('--warm_up_factor', default = 0.1, type = float, help = 'the starting lr scaling for warm up')
train_group.add_argument('--lr_decay_patience', default = 50000, type = int, help = 'patience')
train_group.add_argument('--adam_beta1', type=float, default=0.9)
train_group.add_argument('--adam_beta2', type=float, default=0.96)
train_group.add_argument('--mu', default = 1000., type = float, help = 'DESSILBI optimizer')
train_group.add_argument('--fc_lambda', default = 0.5, type = float, help = 'DESSILBI optimizer')
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
train_group.add_argument('--num_images', type = int, default = 128)
train_group.add_argument('--generate_folder', type = str, default = None)
train_group.add_argument('--generate_text_suffix', type = str, default = '1000_test_sample.txt')

model_group = parser.add_argument_group('Model settings')
model_group.add_argument('--network_pipe', default = "base", type = str, choices = ['base', 'fast'],
    help = 'Model forward pipeline')
model_group.add_argument('--dim', default = 512, type = int, help = 'Model dimension')
model_group.add_argument('--text_seq_len', default = 256, type = int, help = 'Text sequence length')
model_group.add_argument('--depth', default = 24, type = int, help = 'Model depth')
model_group.add_argument('--heads', default = 8, type = int, help = 'Model number of heads')
model_group.add_argument('--dim_head', default = 64, type = int, help = 'Model head dimension')
model_group.add_argument('--reversible', dest = 'reversible', action='store_true')
model_group.add_argument('--loss_img_weight', default = 7, type = int, help = 'Image loss weight')
model_group.add_argument('--attn_types', default = 'full', type = str,
                         help = 'comma separated list of attention types. attention type can be: full or sparse or axial_row or axial_col or conv_like.')
args = parser.parse_args()

# quit early if you used the wrong folder name
# training dataset
dataset_type_list = args.dataset.split(",")
text_folder_list = args.text_folder.split(",")
image_folder_list = args.img_folder.split(",")
assert len(dataset_type_list) == len(text_folder_list) and len(dataset_type_list) == len(image_folder_list), \
    "length of list of dataset, text_folder, image_folder is not the same"
for text_folder, image_folder in zip(text_folder_list, image_folder_list):
    assert Path(text_folder).exists(), f'The path {text_folder} was not found.'
    assert Path(image_folder).exists(), f'The path {image_folder} was not found.'

# helpers
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
DALLE_SAVE_ITER_NAME = args.dalle_output_file_name + "_iter.pt"
DALLE_SAVE_EPOCH_NAME = args.dalle_output_file_name + "_epoch{}.pt"
VAE_PATH = args.vae_path
VQGAN_MODEL_PATH = args.vqgan_model_path
VQGAN_CONFIG_PATH = args.vqgan_config_path
DALLE_PATH = args.dalle_path
RESUME = exists(DALLE_PATH)

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

LEARNING_RATE = args.learning_rate
RESUME_LEARNING_RATE = args.resume_learning_rate
GRAD_CLIP_NORM = args.clip_grad_norm
LR_DECAY = args.lr_decay
SAVE_EVERY_N_STEPS = args.save_every_n_steps
SAVE_EVERY_N_EPOCHS = args.save_every_n_epochs
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

DATASET = args.dataset
DATA_LEN = args.restrict_data_len
EXP_PATH = args.exp_result_path
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
    IS_LOG = LOCAL_RANK == 0 # and RANK == 0
else:
    IS_LOG = True
print(f"WORLD_SIZE: {WORLD_SIZE}  rank: {RANK}  gpu: {LOCAL_RANK}  IS_LOG: {IS_LOG}")

""" Training Logs """
if RESUME_LEARNING_RATE is not None:
    assert RESUME_LEARNING_RATE == LEARNING_RATE
TMP_SAVE_PATH = '../{}/{}'.format(EXP_PATH, DATASET)
prefix = '{:.1f}M'.format(args.restrict_data_len / 10**6) if args.restrict_data_len is not None else ''
args.name = "RESUME_" * RESUME + 'SCRATCH_' * (1 - RESUME) + \
            f'DownStreamOn{DATASET}_' * (RESUME_LEARNING_RATE is not None) + \
            args.prefix + '_' * (args.prefix != '') + \
            prefix + \
            'AMP' * args.amp + 'FP16' * args.fp16 + '_' + \
            'Lr{}_Wd{}_B{}_Epoch{}_Depth{}_Dim{}'.format(
                LEARNING_RATE if not RESUME else RESUME_LEARNING_RATE, WEIGHT_DECAY,
                BATCH_SIZE*args.ga_steps*WORLD_SIZE, EPOCHS, DEPTH, DIM_HEAD*HEADS
            ) + \
            '_' + args.clip_loss_type + args.clip_loss_method + \
            '{}CLIP{}{}_'.format(args.clip_trainer, args.clip_model_name.replace('/', '-'), args.clip_loss_weight) + \
            'STE' * args.straight_through + \
            'Tau{}_'.format(args.gumbel_tau) * (1 - args.straight_through)

args.ckpt_path = os.path.join(TMP_SAVE_PATH, args.name, 'checkpoints')
args.log_path = os.path.join(TMP_SAVE_PATH, args.name, 'logs')
args.outputs_dir = os.path.join(TMP_SAVE_PATH, args.name, 'images')
if IS_LOG:
    print(f"For this time, results are saved in :  {os.path.join(TMP_SAVE_PATH, args.name)}")
    os.makedirs(args.ckpt_path, exist_ok=True)
    with open(join(args.ckpt_path, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=4)
    print(args)
    writer = SummaryWriter(args.log_path)

# tokenizer

if exists(args.bpe_path):
    klass = HugTokenizer if args.hug else YttmTokenizer
    tokenizer = klass(args.bpe_path)
elif args.chinese:
    tokenizer = ChineseTokenizer()

""" create dataset and dataloader """

def get_dataset(dataset_name, text_folder, image_folder):
    print("Construct [%s] Dataset" % dataset_name)
    if dataset_name == 'cub':
        ds = CUB_TextAndImageDataset(  
            text_folder,
            image_folder,
            phase="training",
            text_len=TEXT_SEQ_LEN,
            image_size=args.img_size,
            resize_ratio=args.resize_ratio,
            truncate_captions=args.truncate_captions,
            tokenizer=tokenizer,
            shuffle=False,
            subset='train'
        )
    elif dataset_name == 'coco':
        ds = COCO_TextAndImageDataset(
            image_folder, # the root folder actually
            phase="training",
            text_len=TEXT_SEQ_LEN,
            image_size=args.img_size,
            resize_ratio=args.resize_ratio,
            truncate_captions=args.truncate_captions,
            tokenizer=tokenizer,
            shuffle=False,
            subset='train'
        )
    elif dataset_name == 'oxford102':
        ds = Oxford102_TextAndImageDataset(
            image_folder, # the root folder actually
            phase="training",
            text_len=TEXT_SEQ_LEN,
            image_size=args.img_size,
            resize_ratio=args.resize_ratio,
            truncate_captions=args.truncate_captions,
            tokenizer=tokenizer,
            shuffle=False,
            subset='train'
        )
    else:
        raise Exception("dataset_name[%s] not recognized" % dataset_name)
    print("Finish construct [%s] Dataset, length: %d" % (dataset_name, len(ds)))
    return ds
dataset_list = [get_dataset(dataset_name, text_folder, image_folder)
                for dataset_name, text_folder, image_folder in zip(dataset_type_list, text_folder_list, image_folder_list)]
ds = torch.utils.data.ConcatDataset(dataset_list)
print("length of the whole dataset[%s]: %d" % (args.dataset, len(ds)))

def get_val_dataset(dataset_name, text_folder, image_folder):
    if dataset_name == 'cub':
        val_db = CUB_TextAndImageDataset(
                text_folder,
                image_folder,
                phase="inference",
                text_len=TEXT_SEQ_LEN,
                image_size=args.img_size,
                resize_ratio=args.resize_ratio,
                truncate_captions=args.truncate_captions,
                tokenizer=tokenizer,
                shuffle=False,
                subset='test'
            )
        val_name = 'CUB'
    elif dataset_name == 'coco':
        val_db = COCO_TextAndImageDataset(
            image_folder, # the root folder actually
            phase="inference",
            text_len=TEXT_SEQ_LEN,
            image_size=args.img_size,
            resize_ratio=args.resize_ratio,
            truncate_captions=args.truncate_captions,
            tokenizer=tokenizer,
            shuffle=False,
            subset='test'
        )
        val_name = 'COCO'
    elif dataset_name == 'oxford102':
        val_db = Oxford102_TextAndImageDataset(
            image_folder, # the root folder actually
            phase="inference",
            text_len=TEXT_SEQ_LEN,
            image_size=args.img_size,
            resize_ratio=args.resize_ratio,
            truncate_captions=args.truncate_captions,
            tokenizer=tokenizer,
            shuffle=False,
            subset='test'
        )
        val_name = 'Oxford102'
    else:
        raise NotImplementedError
    return val_db, val_name
dataset_list = [get_val_dataset(dataset_name, text_folder, image_folder)
                for dataset_name, text_folder, image_folder in zip(dataset_type_list, text_folder_list, image_folder_list)]
val_db = torch.utils.data.ConcatDataset([tup[0] for tup in dataset_list])
val_name = ",".join([tup[1] for tup in dataset_list])
print("length of the whole val dataset[%s]: %d" % (args.dataset, len(val_db)))

assert len(ds) > 0, 'training dataset is empty'
assert len(val_db) > 0, 'validation dataset is empty'

if WORLD_SIZE > 1:
    data_sampler = torch.utils.data.distributed.DistributedSampler(
        ds,
        num_replicas=WORLD_SIZE,
        rank=RANK
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_db,
        num_replicas=WORLD_SIZE,
        rank=RANK
    )
else:
    data_sampler = None
    val_sampler = None

if dataset_type_list[0] == 'feat':
    training_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=1)
else:
    training_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, sampler=data_sampler, num_workers=4)
val_loader = DataLoader(val_db, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, sampler=val_sampler, num_workers=4)

if args.generate_folder is not None:
    raise NotImplementedError
else:
    GENERATE_TEXT_FROM_LOADER = False
    gen_db = []

if IS_LOG:
    if args.generate_folder is not None:
        description_dict = defaultdict(list)
        for sentence, dataset, img_id in gen_db.descriptions:
            description_dict[dataset].append((sentence, dataset, img_id))
        for dataset, sentence_dataset_id_list in description_dict.items():
            description_dict[dataset] = sorted(sentence_dataset_id_list, key=lambda x: x[2])

# models
if RESUME:
    dalle_path = Path(DALLE_PATH)
    assert dalle_path.exists(), 'DALL-E model file does not exist'

    loaded_obj = torch.load(str(dalle_path), map_location='cpu')

    dalle_params, vae_params, dalle_weights = loaded_obj['hparams'], loaded_obj['vae_params'], loaded_obj['state_dict']
    opt_state = loaded_obj.get('opt_state')
    scheduler_state = loaded_obj.get('scheduler_state')

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
    resume_epoch = loaded_obj.get('epoch', 0)
    resume_global_iter = loaded_obj.get('global_step', -1)
    training_data_num = loaded_obj.get('training_data_num')

    if args.ema_model:
        if 'cpu_model' in loaded_obj:
            cpu_model = loaded_obj.get('cpu_model')
        else:
            cpu_model = copy.deepcopy(dalle_weights)
        cpu_model_factor = 0.01
else:
    # vae
    if exists(VAE_PATH):
        raise RuntimeError
    else:
        if IS_LOG:
            print('using pretrained VAE for encoding images to tokens')
        vae_params = None

        if args.taming:
            vae = VQGanVAE(VQGAN_MODEL_PATH, VQGAN_CONFIG_PATH)
        else:
            vae = OpenAIDiscreteVAE()

    vae.image_size = args.img_size

    resume_epoch = 0

    # DALL_E
    dalle_params = dict(
        num_text_tokens=tokenizer.vocab_size,
        text_seq_len=TEXT_SEQ_LEN,
        dim=MODEL_DIM,  # out dim
        depth=DEPTH,
        heads=HEADS,  # inner dim for qkv
        dim_head=DIM_HEAD,
        reversible=REVERSIBLE,
        loss_img_weight=LOSS_IMG_WEIGHT,
        attn_types=ATTN_TYPES,
        ff_dropout=FF_DROPOUT,
        attn_dropout=ATTN_DROPOUT,
    )

# configure OpenAI VAE for float16s
if isinstance(vae, OpenAIDiscreteVAE) and args.fp16:
    vae.enc.blocks.output.conv.use_float16 = True
from dalle_pytorch.dalle_pytorch import set_requires_grad
vae = vae.cuda().eval()
set_requires_grad(vae, False)

# CLIP
import clip_openai
from clip_openai import tokenize as clip_tokenize
clip_model, clip_preprocess = clip_openai.load(args.clip_model_name)
clip_model.eval()
set_requires_grad(clip_model, False)
clip_img_preprocess = T.Compose(
    clip_preprocess.transforms[:2] +
    clip_preprocess.transforms[4:]
)
del clip_preprocess

if args.network_pipe == "fast":
    raise NotImplementedError
elif args.network_pipe == "base":
    dalle = DALLE(vae=vae, **dalle_params)
else:
    raise NotImplementedError

num_params = sum(p.numel() for p in dalle.parameters() if p.requires_grad)
num_B = num_params // 1000000000
num_params %= 1000000000
num_M = num_params // 1000000
num_params %= 1000000
num_K = num_params // 1000
num_params %= 1000
print("number of parameters: %dB,%dM,%dK,%d" % (num_B, num_M, num_K, num_params))
dalle = dalle.cuda()
if RESUME:
    print('*'*20, 'MODEL STATES HAVE BEEN LOADED', '*'*20)
    if VQGAN_MODEL_PATH.find(str(dalle_weights["image_emb.weight"].size(0))) == -1:
        dalle_weights.pop("image_emb.weight")
        dalle_weights.pop("to_im_logits.0.weight")
        dalle_weights.pop("to_im_logits.0.bias")
        dalle_weights.pop("to_im_logits.1.weight")
        dalle_weights.pop("to_im_logits.1.bias")
        print('\"image_emb\" and \"to_im_logits\"\'s weights are re-initialized')
    dalle.load_state_dict(dalle_weights, strict=False)

# optimizer
optimizer = AdamW(group_opt_params(dalle, WEIGHT_DECAY), lr=LEARNING_RATE, betas=(args.adam_beta1, args.adam_beta2))

if args.amp and not args.fp16:
    opt_level = 'O1'
elif args.amp and args.fp16:
    opt_level = 'O2'
elif not args.amp and args.fp16:
    opt_level = 'O3'
else:
    opt_level = 'O0'

from apex import amp
amp.register_half_function(vae.model, "encode")
amp.register_half_function(vae.model, "decode")
amp.register_half_function(vae, "get_codebook_indices")
amp.register_half_function(vae, "decode")
[dalle, vae], optimizer = amp.initialize(
    [dalle, vae], optimizer,
    patch_torch_functions=False,
    opt_level=opt_level,
    loss_scale="dynamic"  # args.loss_scale
)

if WORLD_SIZE > 1:
    from apex.parallel import DistributedDataParallel as ApexDDP

    distr_dalle = ApexDDP(dalle)
else:
    distr_dalle = dalle

# if args.amp:
if RESUME and RESUME_LEARNING_RATE is None:
    print('*'*20, 'OPTIMIZER STATES HAVE BEEN LOADED', '*'*20)
    optimizer.load_state_dict(opt_state)

if LR_DECAY is not None:
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.lr_decay_patience,
        min_lr=1e-6,
        verbose=True,
    )
    if RESUME and RESUME_LEARNING_RATE is None and scheduler_state:
        print('*'*20, 'SCHEDULER STATES HAVE BEEN LOADED', '*'*20)
        scheduler.load_state_dict(scheduler_state)
else:
    scheduler = None


def reduce_all_tensor(tensor):
    if WORLD_SIZE == 1:
        return tensor
    averaged = tensor.detach().clone()
    torch.distributed.all_reduce(averaged, torch.distributed.ReduceOp.SUM)
    return averaged / WORLD_SIZE


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_model(path, epoch, global_step):

    if WORLD_SIZE > 1:
        state_dict = distr_dalle.module.state_dict()
    else:
        state_dict =  distr_dalle.state_dict()

    save_obj = {
        'hparams': dalle_params,
        'vae_params': vae_params,
        'epoch': epoch,
        'global_step': global_step,
        'state_dict': state_dict,
        'training_data_num': DATA_LEN,
        'opt_state': optimizer.state_dict(),
    }
    if args.ema_model and 'cpu_model' in locals():
        save_obj['cpu_model'] = cpu_model  

    if scheduler is not None:
        save_obj.update({'scheduler_state': scheduler.state_dict()})

    if RESUME:
        save_obj.update({'resume_model': DALLE_PATH})

    torch.save(save_obj, path)
    print("save model at %d epoch, %d iter" % (epoch, global_step))


def save_model_gradient(path, model, global_step):
    state_gradient_dict = {}

    if WORLD_SIZE > 1:
        check_toy = model.module
    else:
        check_toy = model

    for n, p in check_toy.named_parameters():
        state_gradient_dict[n] = p.grad

    save_obj = {
        'global_step': global_step,
        'state_gradient_dict': state_gradient_dict,
    }

    torch.save(save_obj, path)

# training
if RESUME and RESUME_LEARNING_RATE is None:
    global_iter = resume_global_iter + 1
else:
    global_iter = 0
    resume_epoch = 0
ga_step_cnt = 0
skip_step_cnt = 0

epoch_loss = AverageMeter()
epoch_loss_text = AverageMeter()
epoch_loss_image = AverageMeter()
epoch_loss_meaningful_text = AverageMeter()
epoch_loss_clip = AverageMeter()

iter_loss = AverageMeter()
iter_loss_text = AverageMeter()
iter_loss_image = AverageMeter()
iter_loss_meaningful_text = AverageMeter()
iter_loss_clip = AverageMeter()

if IS_LOG and not RESUME and RANK == 0:
    save_model(os.path.join(args.ckpt_path, DALLE_SAVE_ITER_NAME), epoch=0, global_step=0)
for epoch in range(resume_epoch, EPOCHS):
    epoch_loss.reset()
    epoch_loss_text.reset()
    epoch_loss_image.reset()
    epoch_loss_meaningful_text.reset()
    epoch_loss_clip.reset()

    for i_datatype, i_dataset in zip(dataset_type_list, dataset_list):
        if i_datatype == 'feat':
            i_dataset.get_feat(epoch)
    if data_sampler:
        data_sampler.set_epoch(epoch)            

    for i, (text, images, original_text, image_name) in enumerate(training_loader):
        # warmup learning rate
        warm_up_lr(optimizer, global_iter, args.warm_up_iters, LEARNING_RATE, args.warm_up_factor)  # iteration beginning from 0

        # inputs
        text, images = map(lambda t: t.cuda(), (text, images))
        # img tokens
        if images.dim() == 4:
            if args.fp16:
                images = images.half()
            images = vae.get_codebook_indices(images)
        else:
            images = images

        distr_dalle.train()
        # loss; gradient descent

        loss, (loss_text, loss_image), loss_meaningful_text, img_logits = distr_dalle(text, images, return_loss=True)

        # CLIP LOSS
        clip_tokenized_texts = clip_tokenize(original_text, truncate=True).to(images.device)
        if clip_tokenized_texts.size(-1) == text.size(-1):
            if (clip_tokenized_texts == text).all():
                clip_tokenized_texts = text
        soft_one_hot = fetch_soft_one_hot_map(
            logits=img_logits, tau=args.gumbel_tau, hard_one_hot=args.straight_through,
            labels=None
        )
        new_img = vae(soft_one_hot)
        loss_clip = global_clip_loss(
            clip_model, clip_img_preprocess, new_img, clip_tokenized_texts,
            clip_loss_method=args.clip_loss_method, clip_trainer=args.clip_trainer
        )
        loss = loss + loss_clip * args.clip_loss_weight

        ga_loss = loss / float(args.ga_steps)

        if reduce_all_tensor(ga_loss).isnan().any() or reduce_all_tensor(ga_loss).isinf().any():
            skip_step_cnt += 1
            print(
                '{}, Epoch {}, GlobalIter/SkipIter {}/{}, Rank {} Loss nan/inf:.'.format(
                    datetime.now(), epoch, global_iter, skip_step_cnt, RANK
                )
            )
            continue

        with amp.scale_loss(ga_loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        ga_step_cnt += 1  # steps to accumulate scaled gradients
        if ga_step_cnt >= args.ga_steps:
            # time to update parameters
            ga_step_cnt = 0

            clip_grad_norm_(amp.master_params(optimizer), GRAD_CLIP_NORM)
            optimizer.step()
            optimizer.zero_grad()

            global_iter += 1

        # iteration losses
        if WORLD_SIZE > 1:
            # Collective loss, averaged
            iter_loss.update(reduce_all_tensor(loss).item())
            iter_loss_text.update(reduce_all_tensor(loss_text).item())
            iter_loss_image.update(reduce_all_tensor(loss_image).item())
            iter_loss_meaningful_text.update(reduce_all_tensor(loss_meaningful_text).item())
            iter_loss_clip.update(reduce_all_tensor(loss_clip).item())
        else:
            iter_loss.update(loss.item())
            iter_loss_text.update(loss_text.item())
            iter_loss_image.update(loss_image.item())
            iter_loss_meaningful_text.update(loss_meaningful_text.item())
            iter_loss_clip.update(loss_clip.item())

        # epoch losses
        epoch_loss.update(iter_loss.val)
        epoch_loss_text.update(iter_loss_text.val)
        epoch_loss_image.update(iter_loss_image.val)
        epoch_loss_meaningful_text.update(iter_loss_meaningful_text.val)
        epoch_loss_clip.update(iter_loss_clip.val)

        # iteration logs
        if global_iter % 50 == 0 and ga_step_cnt == 0:
            if IS_LOG:

                torch.cuda.synchronize()
                print(
                    '{}, Epoch {}, Iter {}/{} Loss: {:.4f}, Loss_Text: {:.4f}, Loss_Image: {:.4f}, '
                    'Loss_Meaningful_Text: {:.4f}, Loss_CLIP: {:.4f}'.format(
                        datetime.now(), epoch, i+1, len(training_loader), iter_loss.avg,
                        iter_loss_text.avg, iter_loss_image.avg, iter_loss_meaningful_text.avg,
                        iter_loss_clip.avg
                    )
                )

                writer.add_scalar("Training_Step/loss", iter_loss.avg, global_iter)
                writer.add_scalar("Training_Step/loss_text", iter_loss_text.avg, global_iter)
                writer.add_scalar("Training_Step/loss_image", iter_loss_image.avg, global_iter)
                writer.add_scalar("Training_Step/learning_rate_per_sample",
                                  optimizer.param_groups[0]['lr'] / images.size(0) / WORLD_SIZE / args.ga_steps,
                                  global_iter)
                writer.add_scalar("Training_Step/loss_meaningful_text", iter_loss_meaningful_text.avg, global_iter)
                writer.add_scalar("Training_Step/loss_clip", iter_loss_clip.avg, global_iter)

        if global_iter % args.ema_step == 0 and args.ema_model and ga_step_cnt == 0:
            # print("ema_model update")
            if 'cpu_model' in locals() and cpu_model is not None:
                cpu_model = {key: cpu_model[key].add_(val.cpu().sub_(cpu_model[key]), alpha=cpu_model_factor) for key, val in distr_dalle.module.state_dict().items() if key in cpu_model.keys()}
            else:
                cpu_model = {key: val.cpu() for key, val in distr_dalle.module.state_dict().items() if not key.startswith('vae')}
                cpu_model_factor = 0.01

        # iteration save
        if (global_iter % SAVE_EVERY_N_STEPS == 0 or global_iter == 81000) and ga_step_cnt == 0 and IS_LOG and RANK == 0:
            save_model(os.path.join(args.ckpt_path, DALLE_SAVE_ITER_NAME),
                       epoch=epoch, global_step=global_iter)

        # iteration lr decay
        if LR_DECAY == 'iter' and global_iter >= args.warm_up_iters and ga_step_cnt == 0:  # iteration from 0 because of warm_up_lr()
            scheduler.step(iter_loss.avg)

        # iteration meter reset
        if ga_step_cnt == 0:
            iter_loss.reset()
            iter_loss_text.reset()
            iter_loss_image.reset()
            iter_loss_meaningful_text.reset()
            iter_loss_clip.reset()

    # epoch lr decay
    if LR_DECAY == 'epoch' and epoch >= args.warm_up_iters // len(training_loader):
        scheduler.step(epoch_loss.avg)

    # epoch save model
    if (epoch % SAVE_EVERY_N_EPOCHS == 0 or epoch == EPOCHS-1) and IS_LOG and RANK == 0:
        save_model(os.path.join(args.ckpt_path, DALLE_SAVE_EPOCH_NAME.format(epoch)), epoch=epoch, global_step=global_iter)

    # epoch validation on CUB-200
    torch.cuda.synchronize()
    val_time_start = time.time()
    val_loss, (val_loss_text, val_loss_image), val_loss_meaningful_text, val_loss_clip = \
        validation(distr_dalle, val_loader, args, WORLD_SIZE, vae, clip_model, clip_img_preprocess)
    torch.cuda.synchronize()
    val_time_end = time.time()

    # epoch image generation on COCO and CUB under zero-shot
    if args.ema_model and 'cpu_model' in locals():
        # Use a barrier() to make sure that the process loads the model after process 0 saves it.
        dist.barrier()
        training_model = {key: val.cpu() for key, val in distr_dalle.module.state_dict().items()}
        distr_dalle.module.load_state_dict(cpu_model, strict=False)

    if args.ema_model and 'cpu_model' in locals():
        dist.barrier()
        distr_dalle.module.load_state_dict(training_model)

    # epoch logs
    if IS_LOG:
        writer.add_scalar("Training_Epoch/loss", epoch_loss.avg, epoch)
        writer.add_scalar("Training_Epoch/loss_text", epoch_loss_text.avg, epoch)
        writer.add_scalar("Training_Epoch/loss_image", epoch_loss_image.avg, epoch)
        writer.add_scalar("Training_Epoch/learning_rate_per_sample",
                          optimizer.param_groups[0]['lr'] / images.size(0) / WORLD_SIZE / args.ga_steps,
                          epoch)
        writer.add_scalar("Training_Epoch/loss_meaningful_text", epoch_loss_meaningful_text.avg, epoch)
        writer.add_scalar("Training_Epoch/loss_clip", epoch_loss_clip.avg, epoch)
        # val logs
        print(
            '{}, Epoch {} Validation Loss: {:.4f}, Loss_Text: {:.4f}, Loss_Image: {:.4f}, '
            'Loss_Meaningful_Text: {:.4f}, Loss_CLIP: {:.4f}, '
            'consuming {:.2f} s.'.format(
                datetime.now(), epoch, val_loss, val_loss_text, val_loss_image, val_loss_meaningful_text, val_loss_clip,
                val_time_end - val_time_start)
        )
        writer.add_scalar(f"Val_Epoch_on_{val_name}/loss", val_loss, epoch)
        writer.add_scalar(f"Val_Epoch_on_{val_name}/loss_text", val_loss_text, epoch)
        writer.add_scalar(f"Val_Epoch_on_{val_name}/loss_image", val_loss_image, epoch)
        writer.add_scalar(f"Val_Epoch_on_{val_name}/loss_meaningful_text", val_loss_meaningful_text, epoch)
        writer.add_scalar(f"Val_Epoch_on_{val_name}/loss_clip", val_loss_clip, epoch)

if IS_LOG:
    writer.close()
