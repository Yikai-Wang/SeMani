# Entity-Level Text-Guided Image Manipulation

\[[CVPR 2022 (oral)](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_ManiTrans_Entity-Level_Text-Guided_Image_Manipulation_via_Token-Wise_Semantic_Alignment_and_CVPR_2022_paper.pdf)\], \[[Journal extension](https://arxiv.org/abs/2302.11383)\]

## Overview
This is the official repo for our papers "[ManiTrans: Entity-Level Text-Guided Image Manipulation via Token-wise Semantic Alignment and Generation](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_ManiTrans_Entity-Level_Text-Guided_Image_Manipulation_via_Token-Wise_Semantic_Alignment_and_CVPR_2022_paper.pdf)" and "[Entity-Level Text-Guided Image Manipulation]((https://arxiv.org/abs/2302.11383))".

> We introduce a new task, entity-Level Text-Guided Image Manipulation (eL-TGIM) which aims to manipulate entities of the image with only text descriptions.

> To solve eL-TGIM, we propose an elegant SeMani framework, that decomposes the eL-TGIM into the semantic alignment phase and image manipulation phase.

> We propose a transformer-based framework with discrete token-wise semantic alignment and generation, named SeMani-Trans, and a diffusion-based framework with continuous semantic alignment and generation, named SeMani-Diff.

## Reminder

The code for SeMani-Trans is now available in the **SeMani-Trans** folder. 

We are actively enhancing SeMani-Diff; its code will be made available upon completion of these improvements.



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
@article{wang2023entitylevel,
title={Entity-Level Text-Guided Image Manipulation},
author={Wang, Yikai and Wang, Jianan and Lu, Guansong and Xu, Hang and Li, Zhenguo and Zhang, Wei and Fu, Yanwei},
year={2023},
journal={arXiv preprint arXiv:2302.11383},
}
```