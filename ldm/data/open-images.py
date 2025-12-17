from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import imp

import os
from io import BytesIO
import json
import logging
import base64
from sys import prefix
import threading
import random
from turtle import left, right
import numpy as np
from typing import Callable, List, Tuple, Union
from PIL import Image, ImageDraw
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
import bezier

from PIL import Image
from glob import glob


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


class OpenImageDataset(data.Dataset):
    def __init__(self, state, arbitrary_mask_percent=0, **args
                 ):
        self.state = state
        self.args = args
        self.arbitrary_mask_percent = arbitrary_mask_percent
        self.kernel = np.ones((1, 1), np.uint8)
        self.random_trans = A.Compose([
            A.Resize(height=224, width=224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20),
            A.Blur(p=0.3),
            A.ElasticTransform(p=0.3)
        ])
        bad_list = [
            '1af17f3d912e9aac.txt',
            '1d5ef05c8da80e31.txt',
            '3095084b358d3f2d.txt',
            '3ad7415a11ac1f5e.txt',
            '42a30d8f8fba8b40.txt',
            '1366cde3b480a15c.txt',
            '03a53ed6ab408b9f.txt'
        ]
        self.bbox_path_list = []
        if state == "train":
            dir_name_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
            for dir_name in dir_name_list:
                bbox_dir = os.path.join(args['dataset_dir'], 'bbox', 'train_' + dir_name)
                per_dir_file_list = os.listdir(bbox_dir)
                for file_name in per_dir_file_list:
                    if file_name not in bad_list:
                        self.bbox_path_list.append(os.path.join(bbox_dir, file_name))
        elif state == "validation":
            bbox_dir = os.path.join(args['dataset_dir'], 'bbox', 'validation')
            per_dir_file_list = os.listdir(bbox_dir)
            for file_name in per_dir_file_list:
                if file_name not in bad_list:
                    self.bbox_path_list.append(os.path.join(bbox_dir, file_name))
        else:
            bbox_dir = os.path.join(args['dataset_dir'], 'bbox', 'test')
            per_dir_file_list = os.listdir(bbox_dir)
            for file_name in per_dir_file_list:
                if file_name not in bad_list:
                    self.bbox_path_list.append(os.path.join(bbox_dir, file_name))
        self.bbox_path_list.sort()
        self.length = len(self.bbox_path_list)

    def __getitem__(self, index):
        bbox_path = self.bbox_path_list[index]
        file_name = os.path.splitext(os.path.basename(bbox_path))[0] + '.jpg'
        dir_name = bbox_path.split('/')[-2]
        img_path = os.path.join('dataset/open-images/images', dir_name, file_name)

        bbox_list = []
        with open(bbox_path) as f:
            line = f.readline()
            while line:
                line_split = line.strip('\n').split(" ")
                bbox_temp = []
                for i in range(4):
                    bbox_temp.append(int(float(line_split[i])))
                bbox_list.append(bbox_temp)
                line = f.readline()
        bbox = random.choice(bbox_list)
        img_p = Image.open(img_path).convert("RGB")

        ### Get reference image
        bbox_pad = copy.copy(bbox)
        bbox_pad[0] = bbox[0] - min(10, bbox[0] - 0)
        bbox_pad[1] = bbox[1] - min(10, bbox[1] - 0)
        bbox_pad[2] = bbox[2] + min(10, img_p.size[0] - bbox[2])
        bbox_pad[3] = bbox[3] + min(10, img_p.size[1] - bbox[3])
        img_p_np = cv2.imread(img_path)
        img_p_np = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
        ref_image_tensor = img_p_np[bbox_pad[1]:bbox_pad[3], bbox_pad[0]:bbox_pad[2], :]
        ref_image_tensor = self.random_trans(image=ref_image_tensor)
        ref_image_tensor = Image.fromarray(ref_image_tensor["image"])
        ref_image_tensor = get_tensor_clip()(ref_image_tensor)

        ### Generate mask
        image_tensor = get_tensor()(img_p)
        W, H = img_p.size

        extended_bbox = copy.copy(bbox)
        left_freespace = bbox[0] - 0
        right_freespace = W - bbox[2]
        up_freespace = bbox[1] - 0
        down_freespace = H - bbox[3]
        extended_bbox[0] = bbox[0] - random.randint(0, int(0.4 * left_freespace))
        extended_bbox[1] = bbox[1] - random.randint(0, int(0.4 * up_freespace))
        extended_bbox[2] = bbox[2] + random.randint(0, int(0.4 * right_freespace))
        extended_bbox[3] = bbox[3] + random.randint(0, int(0.4 * down_freespace))

        prob = random.uniform(0, 1)
        if prob < self.arbitrary_mask_percent:
            mask_img = Image.new('RGB', (W, H), (255, 255, 255))
            bbox_mask = copy.copy(bbox)
            extended_bbox_mask = copy.copy(extended_bbox)
            top_nodes = np.asfortranarray([
                [bbox_mask[0], (bbox_mask[0] + bbox_mask[2]) / 2, bbox_mask[2]],
                [bbox_mask[1], extended_bbox_mask[1], bbox_mask[1]],
            ])
            down_nodes = np.asfortranarray([
                [bbox_mask[2], (bbox_mask[0] + bbox_mask[2]) / 2, bbox_mask[0]],
                [bbox_mask[3], extended_bbox_mask[3], bbox_mask[3]],
            ])
            left_nodes = np.asfortranarray([
                [bbox_mask[0], extended_bbox_mask[0], bbox_mask[0]],
                [bbox_mask[3], (bbox_mask[1] + bbox_mask[3]) / 2, bbox_mask[1]],
            ])
            right_nodes = np.asfortranarray([
                [bbox_mask[2], extended_bbox_mask[2], bbox_mask[2]],
                [bbox_mask[1], (bbox_mask[1] + bbox_mask[3]) / 2, bbox_mask[3]],
            ])
            top_curve = bezier.Curve(top_nodes, degree=2)
            right_curve = bezier.Curve(right_nodes, degree=2)
            down_curve = bezier.Curve(down_nodes, degree=2)
            left_curve = bezier.Curve(left_nodes, degree=2)
            curve_list = [top_curve, right_curve, down_curve, left_curve]
            pt_list = []
            random_width = 5
            for curve in curve_list:
                x_list = []
                y_list = []
                for i in range(1, 19):
                    if (curve.evaluate(i * 0.05)[0][0]) not in x_list and (
                            curve.evaluate(i * 0.05)[1][0] not in y_list):
                        pt_list.append((curve.evaluate(i * 0.05)[0][0] + random.randint(-random_width, random_width),
                                        curve.evaluate(i * 0.05)[1][0] + random.randint(-random_width, random_width)))
                        x_list.append(curve.evaluate(i * 0.05)[0][0])
                        y_list.append(curve.evaluate(i * 0.05)[1][0])
            mask_img_draw = ImageDraw.Draw(mask_img)
            mask_img_draw.polygon(pt_list, fill=(0, 0, 0))
            mask_tensor = get_tensor(normalize=False, toTensor=True)(mask_img)[0].unsqueeze(0)
        else:
            mask_img = np.zeros((H, W))
            mask_img[extended_bbox[1]:extended_bbox[3], extended_bbox[0]:extended_bbox[2]] = 1
            mask_img = Image.fromarray(mask_img)
            mask_tensor = 1 - get_tensor(normalize=False, toTensor=True)(mask_img)

        ### Crop square image
        if W > H:
            left_most = extended_bbox[2] - H
            if left_most < 0:
                left_most = 0
            right_most = extended_bbox[0] + H
            if right_most > W:
                right_most = W
            right_most = right_most - H
            if right_most <= left_most:
                image_tensor_cropped = image_tensor
                mask_tensor_cropped = mask_tensor
            else:
                left_pos = random.randint(left_most, right_most)
                free_space = min(extended_bbox[1] - 0, extended_bbox[0] - left_pos, left_pos + H - extended_bbox[2],
                                 H - extended_bbox[3])
                random_free_space = random.randint(0, int(0.6 * free_space))
                image_tensor_cropped = image_tensor[:, 0 + random_free_space:H - random_free_space,
                                       left_pos + random_free_space:left_pos + H - random_free_space]
                mask_tensor_cropped = mask_tensor[:, 0 + random_free_space:H - random_free_space,
                                      left_pos + random_free_space:left_pos + H - random_free_space]

        elif W < H:
            upper_most = extended_bbox[3] - W
            if upper_most < 0:
                upper_most = 0
            lower_most = extended_bbox[1] + W
            if lower_most > H:
                lower_most = H
            lower_most = lower_most - W
            if lower_most <= upper_most:
                image_tensor_cropped = image_tensor
                mask_tensor_cropped = mask_tensor
            else:
                upper_pos = random.randint(upper_most, lower_most)
                free_space = min(extended_bbox[1] - upper_pos, extended_bbox[0] - 0, W - extended_bbox[2],
                                 upper_pos + W - extended_bbox[3])
                random_free_space = random.randint(0, int(0.6 * free_space))
                image_tensor_cropped = image_tensor[:, upper_pos + random_free_space:upper_pos + W - random_free_space,
                                       random_free_space:W - random_free_space]
                mask_tensor_cropped = mask_tensor[:, upper_pos + random_free_space:upper_pos + W - random_free_space,
                                      random_free_space:W - random_free_space]
        else:
            image_tensor_cropped = image_tensor
            mask_tensor_cropped = mask_tensor

        image_tensor_resize = T.Resize([self.args['image_size'], self.args['image_size']])(image_tensor_cropped)
        mask_tensor_resize = T.Resize([self.args['image_size'], self.args['image_size']])(mask_tensor_cropped)
        inpaint_tensor_resize = image_tensor_resize * mask_tensor_resize

        return {"GT": image_tensor_resize, "inpaint_image": inpaint_tensor_resize, "inpaint_mask": mask_tensor_resize,
                "ref_imgs": ref_image_tensor}


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
        ref_image_tensor = get_tensor_clip()(ref_img)

        outputs = {
            "GT": image_tensor_resize,  # 原始图像
            "inpaint_image": inpaint_tensor_resize,  # 已经mask掉的图像
            "inpaint_mask": mask_tensor_resize,  # mask (1=保留,0=填充区域)
            "ref_imgs": ref_image_tensor  # 参考图像
        }

        # ========== 可视化 & 保存 ==========
        import os
        from torchvision.utils import save_image

        save_dir = "debug_vis"
        os.makedirs(save_dir, exist_ok=True)

        save_image(outputs["GT"], os.path.join(save_dir, f"{index}_gt.png"))
        save_image(outputs["inpaint_image"], os.path.join(save_dir, f"{index}_inpaint.png"))
        save_image(outputs["inpaint_mask"], os.path.join(save_dir, f"{index}_mask.png"))
        save_image(outputs["ref_imgs"], os.path.join(save_dir, f"{index}_ref.png"))

        return {
            "GT": image_tensor_resize,  # 原始图像
            "inpaint_image": inpaint_tensor_resize,  # 已经mask掉的图像
            "inpaint_mask": mask_tensor_resize,  # mask (1=保留,0=填充区域)
            "ref_imgs": ref_img  # 参考图像
        }

    def __len__(self):
        return len(self.image_paths)