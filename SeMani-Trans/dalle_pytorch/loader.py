from pathlib import Path
from random import randint, choice
import json
import os
import PIL
import pickle
import numpy as np
import glob
import torch
import pycocotools.mask as mask_util
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms as T
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# class Instance_CUB_TextAndImageDataset(Dataset):
#     def __init__(self,
#                  text_folder,
#                  img_folder,
#                  instance_folder,
#                  max_instance_num,
#                  instance_confidence,
#                  segment_resize_func,
#                  phase,
#                  text_len=256,
#                  image_size=256,
#                  truncate_captions=False,
#                  resize_ratio=0.75,
#                  tokenizer=None,
#                  shuffle=False,
#                  subset='train',
#                  ):
#         """
#         @param folder: Folder containing images and text files matched by their paths' respective "stem"
#         @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
#         """
#         super().__init__()
#         assert subset in ['train', 'val', 'test']
#         self.shuffle = shuffle
#
#         all_text_path = os.path.join(text_folder, 'text')
#         all_img_path = img_folder
#         with open(os.path.join(text_folder, subset if subset != 'val' else 'train', 'filenames.pickle'), 'rb') as f:
#             self.obj_texts = pickle.load(f)  # '001.Black_footed_Albatross/Black_Footed_Albatross_0046_18'
#             print('*'*20, 'The Split of dataset has been recalled back', '*'*20)
#             # if subset == 'val':
#             #     self.obj_texts = self.obj_texts[-1000:]
#             # elif subset == 'train':
#             #     self.obj_texts = self.obj_texts[:-1000]
#
#         self.text_files = {}
#         self.image_files = {}
#         for ot in self.obj_texts:
#             txt_file = os.path.join(all_text_path, ot+'.txt')
#             assert Path(txt_file).exists()
#             self.text_files[ot] = txt_file
#
#             img_file = os.path.join(all_img_path, ot+'.jpg')  # os.path.join(all_img_path, ot.split('/')[-1] +'.jpg')  # os.path.join(all_img_path, ot+'.jpg')
#             assert Path(img_file).exists()
#             self.image_files[ot] = img_file
#
#         assert len(self.text_files) == len(self.obj_texts)
#         self.text_len = text_len
#         self.truncate_captions = truncate_captions
#         self.resize_ratio = resize_ratio
#         self.tokenizer = tokenizer
#         self.image_transform = T.Compose([
#             T.Lambda(lambda img: img.convert('RGB')
#             if img.mode != 'RGB' else img),
#             T.Resize(image_size),
#             T.CenterCrop(image_size),
#             T.ToTensor()
#         ])
#         self.phase = phase
#         assert phase in ["training", "inference"]
#
#         instances_masks = torch.load(instance_folder, map_location="cpu")
#         name_mask_dict = {}
#         for im in instances_masks:
#             name_mask_dict[im["file_name"]] = im
#         del instances_masks
#         self.instances_masks = name_mask_dict
#
#         self.max_instance_num = max_instance_num
#         self.instance_confidence = instance_confidence
#         self.segment_resize_func = segment_resize_func if segment_resize_func is not None else lambda x: x
#         if segment_resize_func is None:
#             print('*'*20, 'Segment has only been resized and center-cropped to {}'.format(image_size), '*'*20)
#         self.segment_transform = T.Compose([
#             T.Resize(image_size),
#             T.CenterCrop(image_size)
#         ])
#
#     def __len__(self):
#         return len(self.obj_texts)
#
#     def random_sample(self):
#         return self.__getitem__(randint(0, self.__len__() - 1))
#
#     def sequential_sample(self, ind):
#         if ind >= self.__len__() - 1:
#             return self.__getitem__(0)
#         return self.__getitem__(ind + 1)
#
#     def skip_sample(self, ind):
#         if self.shuffle:
#             return self.random_sample()
#         return self.sequential_sample(ind=ind)
#
#     def __getitem__(self, ind):
#         key = self.obj_texts[ind]
#
#         text_file = self.text_files[key]
#         image_file = self.image_files[key]
#         image_name = os.path.basename(image_file)
#         cls_name = image_file.split('/')[-2]
#
#         with open(text_file, 'r') as f:
#             descriptions = [line.strip('\n') for line in f]
#         descriptions = list(filter(lambda t: len(t) > 0, descriptions))
#
#         if self.phase == 'training':
#             assert len(descriptions) == 10
#             try:
#                 description = choice(descriptions)
#             except IndexError as zero_captions_in_file_ex:
#                 print(f"An exception occurred trying to load file {text_file}.")
#                 print(f"Skipping index {ind}")
#                 return self.skip_sample(ind)
#         else:
#             description = descriptions[0]
#
#         tokenized_text = self.tokenizer.tokenize(
#             description,
#             self.text_len,
#             truncate_text=self.truncate_captions
#         ).squeeze(0)
#         try:
#             image_tensor = self.image_transform(PIL.Image.open(image_file))
#         except OSError as corrupt_image_exceptions:
#             print(f"An exception occurred trying to load file {image_file}.")
#             print(f"Skipping index {ind}")
#             return self.skip_sample(ind)
#
#         # Segmentation
#         img_instances_mask = self.instances_masks[image_name]["instances"]
#         seg_list = []
#         for iim_id, iim in enumerate(img_instances_mask):
#             seg = iim["segmentation"]
#             if iim_id > 0 and iim["score"] < self.instance_confidence:
#                 continue
#
#             seg = torch.from_numpy(mask_util.decode(seg))  # (1, H, W)
#             seg_list.append(seg)
#             if len(seg_list) >= self.max_instance_num:
#                 break
#
#         seg_list = self.segment_transform(torch.stack(seg_list))  # .bool()  # (S, H, W)
#         seg_list = self.segment_resize_func(seg_list).bool()
#         seg_instance_num = seg_list.size(0)
#         if seg_instance_num < self.max_instance_num:
#             seg_list = torch.cat([
#                 seg_list,
#                 torch.ones(self.max_instance_num - seg_instance_num, seg_list.size(1), seg_list.size(2)).to(
#                     seg_list)
#             ], dim=0)
#         return tokenized_text, image_tensor, description, image_name, cls_name, seg_list, seg_instance_num


class Instance_CUB_OneTextAndImageDataset(Dataset):
    def __init__(self,
                 text_folder,
                 img_folder,
                 instance_folder,
                 max_instance_num,
                 instance_confidence,
                 segment_resize_func,
                 phase,
                 text_len=256,
                 image_size=256,
                 truncate_captions=False,
                 resize_ratio=0.75,
                 tokenizer=None,
                 shuffle=False,
                 subset='train',
                 specified_text=None
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        assert subset in ['train', 'val', 'test']
        self.shuffle = shuffle

        all_text_path = os.path.join(text_folder, 'text')
        all_img_path = img_folder
        with open(os.path.join(text_folder, subset if subset != 'val' else 'train', 'filenames.pickle'), 'rb') as f:
            self.obj_texts = pickle.load(f)  # '001.Black_footed_Albatross/Black_Footed_Albatross_0046_18'
            print('*'*20, 'The Split of dataset has been recalled back', '*'*20)

        self.text_files = {}
        self.image_files = {}
        for ot in self.obj_texts:
            txt_file = os.path.join(all_text_path, ot+'.txt')
            assert Path(txt_file).exists()
            self.text_files[ot] = txt_file

            img_file = os.path.join(all_img_path, ot+'.jpg')  # os.path.join(all_img_path, ot.split('/')[-1] +'.jpg')  # os.path.join(all_img_path, ot+'.jpg')
            assert Path(img_file).exists()
            self.image_files[ot] = img_file

        assert len(self.text_files) == len(self.obj_texts)
        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')
            if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])
        self.phase = phase
        assert phase in ["training", "inference"]

        instances_masks = torch.load(instance_folder, map_location="cpu")
        name_mask_dict = {}
        for im in instances_masks:
            name_mask_dict[im["file_name"]] = im
        del instances_masks
        self.instances_masks = name_mask_dict

        self.max_instance_num = max_instance_num
        self.instance_confidence = instance_confidence
        # if segment_resize_func is not None:
        #     print('Segment Preprocess Transformation is determined when the image size is decided')
        self.segment_resize_func = segment_resize_func if segment_resize_func is not None else lambda x: x
        if segment_resize_func is None:
            print('*'*20, 'Segment has only been resized and center-cropped to {}'.format(image_size), '*'*20)
        self.segment_transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size)
        ])
        if specified_text is not None:
            with open(specified_text, 'r') as f:
                self.specified_text = f.readlines()
                self.specified_text = [st.strip() for st in self.specified_text]
        else:
            self.specified_text = None

    def __len__(self):
        return len(self.obj_texts)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        key = self.obj_texts[ind]

        text_file = self.text_files[key]
        image_file = self.image_files[key]
        image_name = os.path.basename(image_file)
        cls_name = image_file.split('/')[-2]

        text_file = self.text_files[key]
        image_file = self.image_files[key]

        with open(text_file, 'r') as f:
            descriptions = [line.strip('\n') for line in f]
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        description = descriptions[0]

        tokenized_text = self.tokenizer.tokenize(
            description,
            self.text_len,
            truncate_text=self.truncate_captions
        ).squeeze(0)

        try:
            image_tensor = self.image_transform(PIL.Image.open(image_file))
        except OSError as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        # Segmentation
        img_instances_mask = self.instances_masks[image_name]["instances"]
        seg_list = []
        for iim_id, iim in enumerate(img_instances_mask):
            seg = iim["segmentation"]
            if iim_id > 0 and iim["score"] < self.instance_confidence:
                continue

            seg = torch.from_numpy(mask_util.decode(seg))  # (1, H, W)
            seg_list.append(seg)
            if len(seg_list) >= self.max_instance_num:
                break
        seg_list = self.segment_transform(torch.stack(seg_list))  # .bool()  # (S, H, W)
        seg_list = self.segment_resize_func(seg_list).bool()
        seg_instance_num = seg_list.size(0)
        if seg_instance_num < self.max_instance_num:
            seg_list = torch.cat([
                seg_list,
                torch.ones(self.max_instance_num - seg_instance_num, seg_list.size(1), seg_list.size(2)).to(
                    seg_list)
            ], dim=0)
        return tokenized_text, image_tensor, description, image_name, cls_name, seg_list, seg_instance_num


class Instance_Oxford102_OneTextAndImageDataset(Dataset):
    def __init__(self,
                 root_path,
                 instance_folder,
                 max_instance_num,
                 instance_confidence,
                 segment_resize_func,
                 phase,
                 text_len=256,
                 image_size=256,
                 truncate_captions=False,
                 resize_ratio=0.75,
                 tokenizer=None,
                 shuffle=False,
                 subset='train',
                 specified_text=None
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        # assert subset in ['train', 'val', 'test']
        assert subset in ['train', 'test']
        self.shuffle = shuffle
        self.img_folder = os.path.join(root_path, "jpg")
        text_json_path = os.path.join(root_path, "train_img2caption.json" if subset == "train" else "test_img2caption.json")
        assert os.path.exists(self.img_folder) and os.path.exists(text_json_path), self.img_folder + " || " + text_json_path
        self.img2caption_dict = json.load(open(text_json_path, "r"))
        self.img_list = sorted(list(self.img2caption_dict.keys()))

        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')
            if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])
        self.phase = phase
        assert phase in ["training", "inference"]

        instances_masks = torch.load(instance_folder, map_location="cpu")
        name_mask_dict = {}
        for im in instances_masks:
            name_mask_dict[im["file_name"]] = im
        del instances_masks
        self.instances_masks = name_mask_dict

        self.max_instance_num = max_instance_num
        self.instance_confidence = instance_confidence
        # if segment_resize_func is not None:
        #     print('Segment Preprocess Transformation is determined when the image size is decided')
        self.segment_resize_func = segment_resize_func if segment_resize_func is not None else lambda x: x
        if segment_resize_func is None:
            print('*' * 20, 'Segment has only been resized and center-cropped to {}'.format(image_size), '*' * 20)
        self.segment_transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size)
        ])

        if specified_text is not None:
            with open(specified_text, 'r') as f:
                self.specified_text = f.readlines()
                self.specified_text = [st.strip() for st in self.specified_text]
        else:
            self.specified_text = None

    def __len__(self):
        return len(self.img_list)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        image_name = self.img_list[ind]
        image_file = os.path.join(self.img_folder, image_name)
        descriptions = self.img2caption_dict[image_name]
        description = descriptions[0]

        tokenized_text = self.tokenizer.tokenize(
            description,
            self.text_len,
            truncate_text=self.truncate_captions
        ).squeeze(0)
        try:
            image_tensor = self.image_transform(PIL.Image.open(image_file))
        except OSError as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        # Segmentation
        img_instances_mask = self.instances_masks[image_name]["instances"]
        seg_list = []
        for iim_id, iim in enumerate(img_instances_mask):
            seg = iim["segmentation"]
            if iim_id > 0 and iim["score"] < self.instance_confidence:
                continue

            seg = torch.from_numpy(mask_util.decode(seg))  # (1, H, W)
            seg_list.append(seg)
            if len(seg_list) >= self.max_instance_num:
                break
        seg_list = self.segment_transform(torch.stack(seg_list))  # .bool()  # (S, H, W)
        seg_list = self.segment_resize_func(seg_list).bool()
        seg_instance_num = seg_list.size(0)
        if seg_instance_num < self.max_instance_num:
            seg_list = torch.cat([
                seg_list,
                torch.ones(self.max_instance_num - seg_instance_num, seg_list.size(1), seg_list.size(2)).to(
                    seg_list)
            ], dim=0)
        return tokenized_text, image_tensor, description, image_name, "flower", seg_list, seg_instance_num


class Instance_COCO_OneTextAndImageDataset(Dataset):
    def __init__(self,
                 root_path,
                 instance_folder,
                 max_instance_num,
                 instance_confidence,
                 segment_resize_func,
                 phase,
                 text_len=256,
                 image_size=256,
                 truncate_captions=False,
                 resize_ratio=0.75,
                 tokenizer=None,
                 shuffle=False,
                 subset='train',
                 specified_text=None
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        # assert subset in ['train', 'val', 'test']
        assert subset in ['train', 'test']
        self.shuffle = shuffle
        self.img_folder = os.path.join(root_path, "train2014" if subset == "train" else "val2014")
        text_json_path = os.path.join(root_path, "train2014_img2caption.json" if subset == "train" else "val2014_img2caption.json")
        assert os.path.exists(self.img_folder) and os.path.exists(text_json_path)
        self.img2caption_dict = json.load(open(text_json_path, "r"))
        self.img_list = sorted(list(self.img2caption_dict.keys()))

        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')
            if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])
        self.phase = phase
        assert phase in ["training", "inference"]

        instances_masks = torch.load(instance_folder, map_location="cpu")
        name_mask_dict = {}
        for im in instances_masks:
            name_mask_dict[im["file_name"]] = im
        del instances_masks
        self.instances_masks = name_mask_dict

        self.max_instance_num = max_instance_num
        self.instance_confidence = instance_confidence
        # if segment_resize_func is not None:
        #     print('Segment Preprocess Transformation is determined when the image size is decided')
        self.segment_resize_func = segment_resize_func if segment_resize_func is not None else lambda x: x
        if segment_resize_func is None:
            print('*' * 20, 'Segment has only been resized and center-cropped to {}'.format(image_size), '*' * 20)
        self.segment_transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size)
        ])
        if specified_text is not None:
            with open(specified_text, 'r') as f:
                self.specified_text = f.readlines()
                self.specified_text = [st.strip() for st in self.specified_text]
        else:
            self.specified_text = None

    def __len__(self):
        return len(self.img_list)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        image_name = self.img_list[ind]
        image_file = os.path.join(self.img_folder, image_name)
        descriptions = self.img2caption_dict[image_name]

        description = descriptions[0]

        tokenized_text = self.tokenizer.tokenize(
            description,
            self.text_len,
            truncate_text=self.truncate_captions
        ).squeeze(0)
        try:
            image_tensor = self.image_transform(PIL.Image.open(image_file))
        except OSError as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        # Segmentation
        img_instances_mask = self.instances_masks[image_name]["instances"]
        seg_list = []
        for iim_id, iim in enumerate(img_instances_mask):
            seg = iim["segmentation"]
            if iim_id > 0 and iim["score"] < self.instance_confidence:
                continue

            seg = torch.from_numpy(mask_util.decode(seg))  # (1, H, W)
            seg_list.append(seg)
            if len(seg_list) >= self.max_instance_num:
                break
        seg_list = self.segment_transform(torch.stack(seg_list))  # .bool()  # (S, H, W)
        seg_list = self.segment_resize_func(seg_list).bool()
        seg_instance_num = seg_list.size(0)
        if seg_instance_num < self.max_instance_num:
            seg_list = torch.cat([
                seg_list,
                torch.ones(self.max_instance_num - seg_instance_num, seg_list.size(1), seg_list.size(2)).to(
                    seg_list)
            ], dim=0)
        return tokenized_text, image_tensor, description, image_name, "coco", seg_list, seg_instance_num


class Instance_CUB_WrongTextAndImageDataset(Dataset):
    def __init__(self,
                 text_folder,
                 img_folder,
                 instance_folder,
                 max_instance_num,
                 instance_confidence,
                 segment_resize_func,
                 phase,
                 text_len=256,
                 image_size=256,
                 truncate_captions=False,
                 resize_ratio=0.75,
                 tokenizer=None,
                 shuffle=False,
                 subset='train',
                 specified_text=None
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        assert subset in ['train', 'val', 'test']
        self.shuffle = shuffle

        all_text_path = os.path.join(text_folder, 'text')
        all_img_path = img_folder
        with open(os.path.join(text_folder, subset if subset != 'val' else 'train', 'filenames.pickle'), 'rb') as f:
            self.obj_texts = pickle.load(f)  # '001.Black_footed_Albatross/Black_Footed_Albatross_0046_18'
            print('*'*20, 'The Split of dataset has been recalled back', '*'*20)

        self.text_files = {}
        self.image_files = {}
        for ot in self.obj_texts:
            txt_file = os.path.join(all_text_path, ot+'.txt')
            assert Path(txt_file).exists()
            self.text_files[ot] = txt_file

            img_file = os.path.join(all_img_path, ot+'.jpg')  # os.path.join(all_img_path, ot.split('/')[-1] +'.jpg')  # os.path.join(all_img_path, ot+'.jpg')
            assert Path(img_file).exists()
            self.image_files[ot] = img_file

        assert len(self.text_files) == len(self.obj_texts)
        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')
            if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])
        self.phase = phase
        assert phase in ["training", "inference"]

        instances_masks = torch.load(instance_folder, map_location="cpu")
        name_mask_dict = {}
        for im in instances_masks:
            name_mask_dict[im["file_name"]] = im
        del instances_masks
        self.instances_masks = name_mask_dict

        self.max_instance_num = max_instance_num
        self.instance_confidence = instance_confidence
        # if segment_resize_func is not None:
        #     print('Segment Preprocess Transformation is determined when the image size is decided')
        self.segment_resize_func = segment_resize_func if segment_resize_func is not None else lambda x: x
        if segment_resize_func is None:
            print('*'*20, 'Segment has only been resized and center-cropped to {}'.format(image_size), '*'*20)
        self.segment_transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size)
        ])
        if specified_text is not None:
            with open(specified_text, 'r') as f:
                self.specified_text = f.readlines()
                self.specified_text = [st.strip() for st in self.specified_text]
        else:
            self.specified_text = None

    def __len__(self):
        return len(self.obj_texts)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        key = self.obj_texts[ind]

        # text_file = self.text_files[key]
        image_file = self.image_files[key]
        image_name = os.path.basename(image_file)
        cls_name = image_file.split('/')[-2]

        if self.specified_text is None:
            wrong_img_idx = ind
            while wrong_img_idx == ind:
                wrong_img_idx = random.randint(0, len(self.obj_texts)-1)
            wrong_text_file = self.text_files[self.obj_texts[wrong_img_idx]]
            with open(wrong_text_file, 'r') as f:
                wrong_descriptions = [line.strip('\n') for line in f]
            wrong_descriptions = list(filter(lambda t: len(t) > 0, wrong_descriptions))
            wrong_description = choice(wrong_descriptions)
        else:
            wrong_description = self.specified_text

        wrong_tokenized_text = self.tokenizer.tokenize(
            wrong_description,
            self.text_len,
            truncate_text=self.truncate_captions
        ).squeeze(0)

        try:
            image_tensor = self.image_transform(PIL.Image.open(image_file))
        except OSError as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        # Segmentation
        img_instances_mask = self.instances_masks[image_name]["instances"]
        seg_list = []
        for iim_id, iim in enumerate(img_instances_mask):
            seg = iim["segmentation"]
            if iim_id > 0 and iim["score"] < self.instance_confidence:
                continue

            seg = torch.from_numpy(mask_util.decode(seg))  # (1, H, W)
            seg_list.append(seg)
            if len(seg_list) >= self.max_instance_num:
                break
        seg_list = self.segment_transform(torch.stack(seg_list))  # .bool()  # (S, H, W)
        seg_list = self.segment_resize_func(seg_list).bool()
        seg_instance_num = seg_list.size(0)
        if seg_instance_num < self.max_instance_num:
            seg_list = torch.cat([
                seg_list,
                torch.ones(self.max_instance_num - seg_instance_num, seg_list.size(1), seg_list.size(2)).to(
                    seg_list)
            ], dim=0)
        return wrong_tokenized_text, image_tensor, wrong_description, image_name, cls_name, seg_list, seg_instance_num


class Instance_Oxford102_WrongTextAndImageDataset(Dataset):
    def __init__(self,
                 root_path,
                 instance_folder,
                 max_instance_num,
                 instance_confidence,
                 segment_resize_func,
                 phase,
                 text_len=256,
                 image_size=256,
                 truncate_captions=False,
                 resize_ratio=0.75,
                 tokenizer=None,
                 shuffle=False,
                 subset='train',
                 specified_text=None
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        # assert subset in ['train', 'val', 'test']
        assert subset in ['train', 'test']
        self.shuffle = shuffle
        self.img_folder = os.path.join(root_path, "jpg")
        text_json_path = os.path.join(root_path, "train_img2caption.json" if subset == "train" else "test_img2caption.json")
        assert os.path.exists(self.img_folder) and os.path.exists(text_json_path), self.img_folder + " || " + text_json_path
        self.img2caption_dict = json.load(open(text_json_path, "r"))
        self.img_list = sorted(list(self.img2caption_dict.keys()))

        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')
            if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])
        self.phase = phase
        assert phase in ["training", "inference"]

        instances_masks = torch.load(instance_folder, map_location="cpu")
        name_mask_dict = {}
        for im in instances_masks:
            name_mask_dict[im["file_name"]] = im
        del instances_masks
        self.instances_masks = name_mask_dict

        self.max_instance_num = max_instance_num
        self.instance_confidence = instance_confidence
        # if segment_resize_func is not None:
        #     print('Segment Preprocess Transformation is determined when the image size is decided')
        self.segment_resize_func = segment_resize_func if segment_resize_func is not None else lambda x: x
        if segment_resize_func is None:
            print('*' * 20, 'Segment has only been resized and center-cropped to {}'.format(image_size), '*' * 20)
        self.segment_transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size)
        ])

        if specified_text is not None:
            with open(specified_text, 'r') as f:
                self.specified_text = f.readlines()
                self.specified_text = [st.strip() for st in self.specified_text]
        else:
            self.specified_text = None

    def __len__(self):
        return len(self.img_list)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        image_name = self.img_list[ind]
        image_file = os.path.join(self.img_folder, image_name)
        # descriptions = self.img2caption_dict[image_name]

        if self.specified_text is None:
            wrong_img_idx = ind
            while wrong_img_idx == ind:
                wrong_img_idx = random.randint(0, len(self.img_list)-1)
            wrong_image_name = self.img_list[wrong_img_idx]
            wrong_descriptions = self.img2caption_dict[wrong_image_name]
            wrong_description = choice(wrong_descriptions)
        else:
            wrong_description = self.specified_text

        wrong_tokenized_text = self.tokenizer.tokenize(
            wrong_description,
            self.text_len,
            truncate_text=self.truncate_captions
        ).squeeze(0)
        try:
            image_tensor = self.image_transform(PIL.Image.open(image_file))
        except OSError as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        # Segmentation
        img_instances_mask = self.instances_masks[image_name]["instances"]
        seg_list = []
        for iim_id, iim in enumerate(img_instances_mask):
            seg = iim["segmentation"]
            if iim_id > 0 and iim["score"] < self.instance_confidence:
                continue

            seg = torch.from_numpy(mask_util.decode(seg))  # (1, H, W)
            seg_list.append(seg)
            if len(seg_list) >= self.max_instance_num:
                break
        seg_list = self.segment_transform(torch.stack(seg_list))  # .bool()  # (S, H, W)
        seg_list = self.segment_resize_func(seg_list).bool()
        seg_instance_num = seg_list.size(0)
        if seg_instance_num < self.max_instance_num:
            seg_list = torch.cat([
                seg_list,
                torch.ones(self.max_instance_num - seg_instance_num, seg_list.size(1), seg_list.size(2)).to(
                    seg_list)
            ], dim=0)
        return wrong_tokenized_text, image_tensor, wrong_description, image_name, "flower", seg_list, seg_instance_num


class Instance_COCO_WrongTextAndImageDataset(Dataset):
    def __init__(self,
                 root_path,
                 instance_folder,
                 max_instance_num,
                 instance_confidence,
                 segment_resize_func,
                 phase,
                 text_len=256,
                 image_size=256,
                 truncate_captions=False,
                 resize_ratio=0.75,
                 tokenizer=None,
                 shuffle=False,
                 subset='train',
                 specified_text=None
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        # assert subset in ['train', 'val', 'test']
        assert subset in ['train', 'test']
        self.shuffle = shuffle
        self.img_folder = os.path.join(root_path, "train2014" if subset == "train" else "val2014")
        text_json_path = os.path.join(root_path, "train2014_img2caption.json" if subset == "train" else "val2014_img2caption.json")
        assert os.path.exists(self.img_folder) and os.path.exists(text_json_path)
        self.img2caption_dict = json.load(open(text_json_path, "r"))
        self.img_list = sorted(list(self.img2caption_dict.keys()))

        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')
            if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])
        self.phase = phase
        assert phase in ["training", "inference"]

        instances_masks = torch.load(instance_folder, map_location="cpu")
        name_mask_dict = {}
        for im in instances_masks:
            name_mask_dict[im["file_name"]] = im
        del instances_masks
        self.instances_masks = name_mask_dict

        self.max_instance_num = max_instance_num
        self.instance_confidence = instance_confidence
        # if segment_resize_func is not None:
        #     print('Segment Preprocess Transformation is determined when the image size is decided')
        self.segment_resize_func = segment_resize_func if segment_resize_func is not None else lambda x: x
        if segment_resize_func is None:
            print('*' * 20, 'Segment has only been resized and center-cropped to {}'.format(image_size), '*' * 20)
        self.segment_transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size)
        ])
        if specified_text is not None:
            with open(specified_text, 'r') as f:
                self.specified_text = f.readlines()
                self.specified_text = [st.strip() for st in self.specified_text]
        else:
            self.specified_text = None

    def __len__(self):
        return len(self.img_list)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        image_name = self.img_list[ind]
        image_file = os.path.join(self.img_folder, image_name)

        if self.specified_text is None:
            wrong_img_idx = ind
            while wrong_img_idx == ind:
                wrong_img_idx = random.randint(0, len(self.img_list) - 1)
            wrong_image_name = self.img_list[wrong_img_idx]
            wrong_descriptions = self.img2caption_dict[wrong_image_name]
            wrong_description = choice(wrong_descriptions)
        else:
            wrong_description = self.specified_text

        wrong_tokenized_text = self.tokenizer.tokenize(
            wrong_description,
            self.text_len,
            truncate_text=self.truncate_captions
        ).squeeze(0)
        try:
            image_tensor = self.image_transform(PIL.Image.open(image_file))
        except OSError as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        # Segmentation
        img_instances_mask = self.instances_masks[image_name]["instances"]
        seg_list = []
        for iim_id, iim in enumerate(img_instances_mask):
            seg = iim["segmentation"]
            if iim_id > 0 and iim["score"] < self.instance_confidence:
                continue

            seg = torch.from_numpy(mask_util.decode(seg))  # (1, H, W)
            seg_list.append(seg)
            if len(seg_list) >= self.max_instance_num:
                break
        seg_list = self.segment_transform(torch.stack(seg_list))  # .bool()  # (S, H, W)
        seg_list = self.segment_resize_func(seg_list).bool()
        seg_instance_num = seg_list.size(0)
        if seg_instance_num < self.max_instance_num:
            seg_list = torch.cat([
                seg_list,
                torch.ones(self.max_instance_num - seg_instance_num, seg_list.size(1), seg_list.size(2)).to(
                    seg_list)
            ], dim=0)
        return wrong_tokenized_text, image_tensor, wrong_description, image_name, "coco", seg_list, seg_instance_num


class CUB_TextAndImageDataset(Dataset):
    def __init__(self,
                 text_folder,
                 img_folder,
                 phase,
                 text_len=256,
                 image_size=256,
                 truncate_captions=False,
                 resize_ratio=0.75,
                 tokenizer=None,
                 shuffle=False,
                 subset='train',
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        assert subset in ['train', 'val', 'test']
        self.shuffle = shuffle

        all_text_path = os.path.join(text_folder, 'text')
        all_img_path = img_folder
        with open(os.path.join(text_folder, subset if subset != 'val' else 'train', 'filenames.pickle'), 'rb') as f:
            self.obj_texts = pickle.load(f)  # '001.Black_footed_Albatross/Black_Footed_Albatross_0046_18'
            print('*'*20, 'The Split of dataset has been recalled back', '*'*20)
            # if subset == 'val':
            #     self.obj_texts = self.obj_texts[-1000:]
            # elif subset == 'train':
            #     self.obj_texts = self.obj_texts[:-1000]

        self.text_files = {}
        self.image_files = {}
        for ot in self.obj_texts:
            txt_file = os.path.join(all_text_path, ot+'.txt')
            assert Path(txt_file).exists()
            self.text_files[ot] = txt_file

            img_file = os.path.join(all_img_path, ot+'.jpg')  # os.path.join(all_img_path, ot.split('/')[-1] +'.jpg')  # os.path.join(all_img_path, ot+'.jpg')
            assert Path(img_file).exists()
            self.image_files[ot] = img_file

        assert len(self.text_files) == len(self.obj_texts)
        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')
            if img.mode != 'RGB' else img),
            T.RandomResizedCrop(image_size,
                                scale=(self.resize_ratio, 1.),  # before resizing
                                ratio=(1., 1.)),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])
        self.phase = phase
        assert phase in ["training", "inference"]

    def __len__(self):
        return len(self.obj_texts)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        key = self.obj_texts[ind]

        text_file = self.text_files[key]
        image_file = self.image_files[key]

        with open(text_file, 'r') as f:
            descriptions = [line.strip('\n') for line in f]
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))

        if self.phase == 'training':
            assert len(descriptions) == 10
            try:
                description = choice(descriptions)
            except IndexError as zero_captions_in_file_ex:
                print(f"An exception occurred trying to load file {text_file}.")
                print(f"Skipping index {ind}")
                return self.skip_sample(ind)
        else:
            description = descriptions[0]

        tokenized_text = self.tokenizer.tokenize(
            description,
            self.text_len,
            truncate_text=self.truncate_captions
        ).squeeze(0)
        try:
            image_tensor = self.image_transform(PIL.Image.open(image_file))
        except OSError as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        # Success
        return tokenized_text, image_tensor, description, os.path.basename(image_file)


class COCO_TextAndImageDataset(Dataset):
    def __init__(self,
                 root_path,
                 phase,
                 text_len=256,
                 image_size=256,
                 truncate_captions=False,
                 resize_ratio=0.75,
                 tokenizer=None,
                 shuffle=False,
                 subset='train',
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        # assert subset in ['train', 'val', 'test']
        assert subset in ['train', 'test']
        self.shuffle = shuffle
        self.img_folder = os.path.join(root_path, "train2014" if subset == "train" else "val2014")
        text_json_path = os.path.join(root_path, "train2014_img2caption.json" if subset == "train" else "val2014_img2caption.json")
        assert os.path.exists(self.img_folder) and os.path.exists(text_json_path)
        self.img2caption_dict = json.load(open(text_json_path, "r"))
        self.img_list = sorted(list(self.img2caption_dict.keys()))
        # if subset == "val":
        #     self.img_list = self.img_list[-1000:]
        # else:
        #     self.img_list = self.img_list[:-1000]

        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')
            if img.mode != 'RGB' else img),
            T.RandomResizedCrop(image_size,
                                scale=(self.resize_ratio, 1.),  # before resizing
                                ratio=(1., 1.)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor()
        ])
        self.phase = phase
        assert phase in ["training", "inference"]

    def __len__(self):
        return len(self.img_list)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        image_name = self.img_list[ind]
        image_file = os.path.join(self.img_folder, image_name)
        descriptions = self.img2caption_dict[image_name]

        if self.phase == 'training':
            # assert len(descriptions) == 5
            try:
                description = choice(descriptions)
            except IndexError as zero_captions_in_file_ex:
                # print(f"An exception occurred trying to load file {text_file}.")
                print(f"Skipping index {ind}")
                return self.skip_sample(ind)
        else:
            description = descriptions[0]

        tokenized_text = self.tokenizer.tokenize(
            description,
            self.text_len,
            truncate_text=self.truncate_captions
        ).squeeze(0)
        try:
            image_tensor = self.image_transform(PIL.Image.open(image_file))
        except OSError as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        # Success
        return tokenized_text, image_tensor, description, os.path.basename(image_file)


class Oxford102_TextAndImageDataset(Dataset):
    def __init__(self,
                 root_path,
                 phase,
                 text_len=256,
                 image_size=256,
                 truncate_captions=False,
                 resize_ratio=0.75,
                 tokenizer=None,
                 shuffle=False,
                 subset='train',
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        # assert subset in ['train', 'val', 'test']
        assert subset in ['train', 'test']
        self.shuffle = shuffle
        self.img_folder = os.path.join(root_path, "jpg")
        text_json_path = os.path.join(root_path, "train_img2caption.json" if subset == "train" else "test_img2caption.json")
        assert os.path.exists(self.img_folder) and os.path.exists(text_json_path), self.img_folder + " || " + text_json_path
        self.img2caption_dict = json.load(open(text_json_path, "r"))
        self.img_list = sorted(list(self.img2caption_dict.keys()))
        # if subset == "val":
        #     self.img_list = self.img_list[-1000:]
        # else:
        #     self.img_list = self.img_list[:-1000]

        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')
            if img.mode != 'RGB' else img),
            T.RandomResizedCrop(image_size,
                                scale=(self.resize_ratio, 1.),  # before resizing
                                ratio=(1., 1.)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor()
        ])
        self.phase = phase
        assert phase in ["training", "inference"]

    def __len__(self):
        return len(self.img_list)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        image_name = self.img_list[ind]
        image_file = os.path.join(self.img_folder, image_name)
        descriptions = self.img2caption_dict[image_name]

        if self.phase == 'training':
            # assert len(descriptions) == 5
            try:
                description = choice(descriptions)
            except IndexError as zero_captions_in_file_ex:
                # print(f"An exception occurred trying to load file {text_file}.")
                print(f"Skipping index {ind}")
                return self.skip_sample(ind)
        else:
            description = descriptions[0]

        tokenized_text = self.tokenizer.tokenize(
            description,
            self.text_len,
            truncate_text=self.truncate_captions
        ).squeeze(0)
        try:
            image_tensor = self.image_transform(PIL.Image.open(image_file))
        except OSError as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        # Success
        return tokenized_text, image_tensor, description, os.path.basename(image_file)


class CC12M_TextAndImageDataset(Dataset):
    def __init__(self,
                 text_folder,
                 img_folder,
                 restrict_data_len,
                 text_len=256,
                 image_size=128,
                 truncate_captions=False,
                 resize_ratio=0.75,
                 tokenizer=None,
                 shuffle=False,
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        OBJ_FIELDS = ["dir", 'image_name', 'text']
        self.shuffle = shuffle

        # files = [*Path(text_folder).glob('*.csv')]
        cnt = 0
        self.texts = []
        self.image_files = []
        for i in range(10):
            f = os.path.join(text_folder, 'cc12m_split_%d.csv' % i)
            df = pd.read_csv(f)[OBJ_FIELDS]

            assert df['image_name'].unique().shape[0] == df.shape[0]
            for i, r in df.iterrows():
                if restrict_data_len is not None:
                    if cnt >= restrict_data_len:
                        break
                img_path = os.path.join(img_folder, r['dir'])
                if os.path.exists(img_path) and len(r['text'].strip(' \n\t')) > 0:
                    cnt += 1
                    self.texts.append(r['text'])
                    self.image_files.append(img_path)

            if restrict_data_len is not None:
                if cnt >= restrict_data_len:
                    break

        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')
            if img.mode != 'RGB' else img),
            T.RandomResizedCrop(image_size,
                                scale=(self.resize_ratio, 1.),  # before resizing
                                ratio=(1., 1.)),
            T.ToTensor()
        ])

        if restrict_data_len is not None:
            assert len(self.texts) == restrict_data_len
        # print(len(self.texts))

    def __len__(self):
        return len(self.texts)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        text = self.texts[ind]
        image_file = self.image_files[ind]

        tokenized_text = self.tokenizer.tokenize(
            [text],
            self.text_len,
            truncate_text=self.truncate_captions
        ).squeeze(0)
        try:
            image_tensor = self.image_transform(PIL.Image.open(image_file))
        except OSError as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        # Success
        return tokenized_text, image_tensor


