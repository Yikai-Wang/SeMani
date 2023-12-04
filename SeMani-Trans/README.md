# ManiTrans: Entity-Level Text-Guided Image Manipulation via Token-wise Semantic Alignment and Generation

\[[CVPR 2022 (oral)](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_ManiTrans_Entity-Level_Text-Guided_Image_Manipulation_via_Token-Wise_Semantic_Alignment_and_CVPR_2022_paper.pdf)\]

## Overview

Semani follows the pipeline of Segment -> Align -> Manipulate. 
In particular, SeMani-Trans uses entity segmentation [1] to segment the image, uses FILIP [2] to align the text and entites, and uses a transformer model to manipulate the image.


This repo contains only the training and inference code for SeMani-Trans. You could find the code for [EntitySeg](https://github.com/qqlu/Entity) [1] in its repo, while the code for FILIP [2] is not open-sourced.

[1]: Qi, Lu, et al. "Open world entity segmentation." IEEE Transactions on Pattern Analysis and Machine Intelligence (2022).

[2]: Yao, Lewei, et al. "FILIP: Fine-grained Interactive Language-Image Pre-Training." International Conference on Learning Representations. 2021.

## Training
`train.py` is to train SeMani-Trans for capturing the
correlation between text and image.

```shell
datasets="cub,flower"
text_folder="$path_cub,$path_flower"
img_folder="$path_cub,$path_flower"
vqgan_model_path=PATH_TO_VQGAN_MODEL
vqgan_config_path=PATH_TO_VQGAN_CONFIG

python -m torch.distributed.launch --nproc_per_node=8 --master_port=9741 train.py --move_results_to_s3 --truncate_captions --dataset $datasets --taming --vqgan_model_path $vqgan_model_path --vqgan_config_path $vqgan_config_path --img_size 256 --text_folder $text_folder --img_folder $img_folder --epochs 1000 --save_every_n_steps 1000 --warm_up_iters 1000 --lr_decay_patience 100 --batch_size 4 --ga_steps 3 --learning_rate 5e-4 --resume_learning_rate 5e-4 --weight_decay 1e-3 --lr_decay "epoch" --text_seq_len 128 --heads 12 --dim 768 --depth 12 --exp_result_path "exp_pretrain" --amp --clip_model_name 'ViT-B/32' --clip_loss_weight 5.0 --clip_loss_method 'abs' --straight_through
```

`finetune.py` is to finetune SeMani-Trans with visual context for local manipulation.

```shell
trained_model=PATH_TO_TRAINED_MODEL
datasets="cub,flower"
text_folder="$path_cub,$path_flower"
img_folder="$path_cub,$path_flower"
vqgan_model_path=PATH_TO_VQGAN_MODEL
vqgan_config_path=PATH_TO_VQGAN_CONFIG

python -m torch.distributed.launch --nnodes=4 --node_rank=3 --nproc_per_node=8 --master_addr=ADDR --master_port=6345 finetune.py --truncate_captions --dataset $datasets --taming --vqgan_model_path $vqgan_model_path --vqgan_config_path $vqgan_config_path --img_size 256 --text_folder $text_folder --img_folder $img_folder --epochs 1000 --save_every_n_steps 1000 --warm_up_iters 1000 --lr_decay_patience 10 --batch_size 3 --learning_rate 5e-4 --resume_learning_rate 5e-4 --weight_decay 1e-3 --lr_decay "epoch" --text_seq_len 128 --ff_dropout 0.1 --attn_dropout 0.1 --depth 24 --exp_result_path "exp_finetune" --dalle_path $trained_model --clip_model_name 'ViT-B/32' --clip_loss_weight 5.0 --clip_loss_method 'abs' --straight_through

```

# Inference

`evaluate.py` is to support to edit an image with mutli-entities of multi-guides.

Note that the evaluate.py relies on the [EntitySeg](https://github.com/qqlu/Entity) [1] and FILIP [2] to run successfully. In this file we set None for FILIP model and assume the EntitySeg model is in the folder *EntitySeg*. You should modify the code to run it with your own segment and align model.

The input text file, which can be found in `example_filenames.txt`, has the format as following:

```
horse: a brown horse
ground: a dirt field
horse: a brown horse | ground: a dirt field

bird: a blue bird
bird: a red bird
bird: a blue bird | bird: a red bird
```


```shell
trained_model=PATH_TO_TRAINED_MODEL
vqgan_model_path=PATH_TO_VQGAN_MODEL
vqgan_config_path=PATH_TO_VQGAN_CONFIG
CUDA_VISIBLE_DEVICES=0 python evaluate.py --dataset coco --network_pipe "base_edit" --dalle_path $trained_model --taming --vqgan_model_path $vqgan_model_path --vqgan_config_path $vqgan_config_path --img_size 256 --text_seq_len 128 --outputs_dir "evaluate" --amp --filter_thres 0.99 --num_images 10
```

## Citation

If you found the provided code useful, please cite our work.

```
@inproceedings{wang2022manitrans,
title={ManiTrans: Entity-Level Text-Guided Image Manipulation via Token-wise Semantic Alignment and Generation},
author={Wang, Jianan and Lu, Guansong and Xu, Hang and Li, Zhenguo and Xu, Chunjing and Fu, Yanwei},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
pages={10707--10717},
year={2022}
}
```