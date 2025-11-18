import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import numpy as np
from tqdm import tqdm
from module.FFAS import FFAS
import time

image_dir = "put your input image dir here"
mask_dir = "put your output mask dir here"
os.makedirs(mask_dir, exist_ok=True)

mask_model = FFAS(version='ViT-B/16', reduce_dim=64)
mask_model.eval().to("cuda")
state_dict = torch.load("checkpoints/FFAS_best_model.pth", map_location="cpu")
mask_model.load_state_dict(state_dict, strict=False)

transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def get_cloth_type(fname: str) -> str:
    return fname.split("-")[1]

class FashionImageDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.fnames = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.image_dir, fname)
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        # cloth_type = get_cloth_type(fname)
        # prompt = f"a picture of {cloth_type}"
        prompt = "top garment"
        return img, fname, prompt

# === DataLoader ===
dataset = FashionImageDataset(image_dir, transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

with torch.no_grad():
    for imgs, fnames, prompts in tqdm(dataloader):
        imgs = imgs.cuda()
        torch.cuda.synchronize()
        start_time = time.time()

        for i in range(len(fnames)):            
            logits, _ = mask_model(imgs[i].unsqueeze(0), prompts[i])
            edit_mask = torch.sigmoid(logits)
            edit_mask = edit_mask > 0.5
            # edit_mask = ~edit_mask
            binary_mask = edit_mask.squeeze().cpu().numpy().astype(np.uint8) * 255
            mask_path = os.path.join(mask_dir, fnames[i])
            Image.fromarray(binary_mask).save(mask_path)
