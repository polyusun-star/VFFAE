from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import imp

import os
from io import BytesIO
import json
import logging
import base64
import threading
import random
from turtle import left, right
import numpy as np
from typing import Callable, List, Tuple, Union
from PIL import Image,ImageDraw
import torch.utils.data as data
import json
import time
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import copy
import math
from functools import partial
import albumentations as A
import clip
import bezier

from PIL import Image
from glob import glob

from torchvision import transforms

def bbox_process(bbox):
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = x_min + int(bbox[2])
    y_max = y_min + int(bbox[3])
    return list(map(int, [x_min, y_min, x_max, y_max]))


def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)


class COCOImageDataset(data.Dataset):
    def __init__(self,test_bench_dir):

        self.test_bench_dir=test_bench_dir
        self.id_list=np.load('test_bench/id_list.npy')
        self.id_list=self.id_list.tolist()
        print("length of test bench",len(self.id_list))
        self.length=len(self.id_list)

       

    
    def __getitem__(self, index):
        img_path=os.path.join(os.path.join(self.test_bench_dir,'GT_3500',str(self.id_list[index]).zfill(12)+'_GT.png'))
        img_p = Image.open(img_path).convert("RGB")

        ### Get reference
        ref_img_path=os.path.join(os.path.join(self.test_bench_dir,'Ref_3500',str(self.id_list[index]).zfill(12)+'_ref.png'))
        ref_img=Image.open(ref_img_path).resize((224,224)).convert("RGB")
        ref_img=get_tensor_clip()(ref_img)
        ref_image_tensor = ref_img.unsqueeze(0)


        ### Crop input image
        image_tensor = get_tensor()(img_p)
        W,H = img_p.size

   
        ### bbox mask
        mask_path=os.path.join(os.path.join(self.test_bench_dir,'Mask_bbox_3500',str(self.id_list[index]).zfill(12)+'_mask.png'))
        mask_img = Image.open(mask_path).convert('L')
        mask_tensor=1-get_tensor(normalize=False, toTensor=True)(mask_img)



      

        inpaint_tensor=image_tensor*mask_tensor
    
        return image_tensor, {"inpaint_image":inpaint_tensor,"inpaint_mask":mask_tensor,"ref_imgs":ref_image_tensor},str(self.id_list[index]).zfill(12)



    def __len__(self):
        return self.length

import torch
import torchvision.transforms as T
from torchvision.utils import save_image
import matplotlib.pyplot as plt

def save_and_visualize(outputs, save_dir="./visualize_output"):
    import os
    os.makedirs(save_dir, exist_ok=True)

    # 定义tensor转图像
    to_pil = T.ToPILImage()

    for key, tensor in outputs.items():
        if isinstance(tensor, torch.Tensor):
            if tensor.ndim == 4:  
                # ref_imgs 可能是 BxCxHxW，这里只保存第一个
                img = to_pil(tensor[0].cpu())
            else:  
                img = to_pil(tensor.cpu())

            img.save(os.path.join(save_dir, f"{key}.png"))

    # 可视化
    fig, axs = plt.subplots(1, 4, figsize=(16, 5))
    for idx, (key, tensor) in enumerate(outputs.items()):
        if tensor.ndim == 4:
            img = to_pil(tensor[0].cpu())
        else:
            img = to_pil(tensor.cpu())
        axs[idx].imshow(img)
        axs[idx].set_title(key)
        axs[idx].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "all_outputs.png"))
    plt.show()

class CustomImageMaskDataset(data.Dataset):
    def __init__(self, dataset_dir, state="train", image_size=512, **kwargs):
        super().__init__()
        self.state = state
        self.image_size = image_size

        self.image_dir = os.path.join(dataset_dir, "images")
        self.mask_dir = os.path.join(dataset_dir, "masks")
        self.ref_dir = os.path.join(dataset_dir, "ref_images")

        # 匹配同名文件
        self.image_paths = sorted(glob(os.path.join(self.image_dir, "*.jpg")))
        self.mask_paths = sorted(glob(os.path.join(self.mask_dir, "*.png")))
        self.ref_paths = sorted(glob(os.path.join(self.ref_dir, "*.jpg")))

        assert len(self.image_paths) == len(self.mask_paths), "图像和mask数量不一致！"

        # Albumentations 数据增强
        self.random_trans = A.Compose([
            A.Resize(height=224, width=224),
            A.HorizontalFlip(p=0.6),
            A.Rotate(limit=20, p=0.6),
            A.Blur(p=0.3),
            A.ElasticTransform(p=0.3)
        ])

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                 [0.26862954, 0.26130258, 0.27577711])
        ])

    def preprocess(self, img):
        """img: PIL.Image or tensor in [0, 1] range (C,H,W)"""
        if not isinstance(img, torch.Tensor):
            img = self.transform(img)
        if img.dim() == 3:
            img = img.unsqueeze(0)  # Add batch dimension
        return img#.to(self.device)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        ref_path = self.ref_paths[index]

        # === 读取图像和mask ===
        img = Image.open(img_path).convert("RGB")

        # # PIL -> tensor
        image_tensor = get_tensor()(img)

        # === 读取并二值化 mask ===
        mask = Image.open(mask_path).convert("L")
        mask_tensor = T.ToTensor()(mask)
        # mask_np = (mask_tensor.squeeze().numpy() > 0.5).astype(np.uint8)
        mask_tensor = (1- mask_tensor)

        # k_size = random.choice([15, 30])
        # kernel = np.ones((1, k_size), np.uint8)

        # mask_np = cv2.erode(mask_np, kernel, iterations=1)

        # mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()

        resize = T.Resize([self.image_size, self.image_size])
        image_tensor_resize = resize(image_tensor)
        mask_tensor_resize = resize(mask_tensor)

        # 生成 inpaint image
        inpaint_tensor_resize = image_tensor_resize * mask_tensor_resize

        ref_img = Image.open(ref_path).convert("RGB").resize((224, 224))
        # ref_image_tensor = get_tensor_clip()(ref_img)        
        ref_image_tensor = self.preprocess(ref_img)      

        return image_tensor_resize, {"inpaint_image":inpaint_tensor_resize,"inpaint_mask":mask_tensor_resize,"ref_imgs":ref_image_tensor}, str(index).zfill(12)

    def __len__(self):
        return len(self.image_paths)

