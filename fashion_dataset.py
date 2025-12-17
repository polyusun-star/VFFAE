import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
import os

class FashionpediaDataset(Dataset):
    def __init__(self, image_dir, ann_file, clip_tokenizer, transform=None):
        self.coco = COCO(ann_file)
        self.image_dir = image_dir
        self.ids = list(self.coco.imgs.keys())
        # self.transform = transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((352, 352)),
        ])
        self.clip_tokenizer = clip_tokenizer
        self.resize = transforms.Resize((352, 352))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        # load image
        img_info = self.coco.loadImgs(image_id)[0]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")

        # use first annotation (for simplicity)
        ann = anns[0]
        mask = self.coco.annToMask(ann)

        # Get category name as text prompt
        cat_id = ann['category_id']
        cat_name = self.coco.loadCats(cat_id)[0]['name']
        text = f"a picture of {cat_name}"

        if self.transform:
            image = self.transform(image)            
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            mask = self.resize(mask)
            # print(image.shape, mask.shape)
            # mask = self.transform(mask)

        # CLIP tokenization
        text_tokens = self.clip_tokenizer(text, return_tensors="pt", padding='max_length', truncation=True)

        return image, text, img_info['file_name'], mask
        # return image, text, img_info['file_name'], text_tokens['attention_mask'].squeeze(0), mask

