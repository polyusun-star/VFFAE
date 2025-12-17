import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, utils
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").vision_model.to(device)
clip_model.eval()  # Freeze CLIP
for param in clip_model.parameters():
    param.requires_grad = False

class FeatureMapper(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.to_feat = nn.Sequential(
            nn.Linear(latent_dim, 64 * 16 * 16),
            nn.ReLU()
        )

        self.to_stat = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU()
        )

    def forward(self, z_c, z_s):
        feat_map = self.to_feat(z_c).view(-1, 64, 16, 16)
        stats = self.to_stat(z_s)
        mean = stats[:, :64]
        std = torch.abs(stats[:, 64:]) + 1e-5
        return feat_map, mean, std

class SubspaceProjector(nn.Module):
    def __init__(self, input_dim, content_rank, style_rank):
        super().__init__()
        self.input_dim = input_dim
        self.content_rank = content_rank
        self.style_rank = style_rank
        self.content_basis = nn.Parameter(torch.randn(input_dim, content_rank))
        self.style_basis = nn.Parameter(torch.randn(input_dim, style_rank))

        self.linner1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        self.linner2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
        )
    
    def forward(self, z):
        content_basis = F.normalize(self.content_basis, dim=0)
        style_basis = F.normalize(self.style_basis, dim=0)
        z_c = z @ content_basis @ content_basis.T
        z_s = z @ style_basis @ style_basis.T
        return z_c, z_s, content_basis, style_basis

class AdaIN(nn.Module):
    def forward(self, content, mean, std):
        print()
        size = content.size()
        content_mean = content.view(size[0], size[1], -1).mean(2).view(size[0], size[1], 1, 1)
        content_std = content.view(size[0], size[1], -1).std(2).view(size[0], size[1], 1, 1) + 1e-5
        normalized = (content - content_mean) / content_std
        return normalized * std.unsqueeze(2).unsqueeze(3) + mean.unsqueeze(2).unsqueeze(3)

class ConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 256, 4, stride=2, padding=1),  # 16 → 32
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 32 → 64
            nn.ReLU(),
            # nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 64 → 128
            # nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(x)

def orthogonality_loss(Bc, Bs):
    """
    Bc: [d, r_c] structure/content basis
    Bs: [d, r_s] style basis
    """
    # Bc^T Bs -> [r_c, r_s]
    cross = torch.matmul(Bc.T, Bs)
    loss = torch.norm(cross, p='fro') ** 2
    return loss


input_dim = 768
content_rank = 128
style_rank = 128
feature_map_dim = 256
style_dim = 256

def train():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758])  # CLIP normalization
    ])
    
    dataset = datasets.ImageFolder('./data/FashionVCdata/top', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    projector = SubspaceProjector(input_dim, content_rank, style_rank).to(device)
    mapper = FeatureMapper(768).to(device)
    adain = AdaIN().to(device)
    decoder = ConvDecoder().to(device)

    optimizer = torch.optim.Adam(list(projector.parameters()) + list(mapper.parameters()) + list(decoder.parameters()), lr=5e-5)

    for epoch in range(50):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            imgs, _ = batch
            imgs = imgs.to(device)

            with torch.no_grad():
                clip_feat = clip_model(imgs).pooler_output  # [B, 768]

            z_c, z_s, content_basis, style_basis = projector(clip_feat)

            # Non-linear mapping
            feat_map, mean, std = mapper(z_c, z_s)

            # ADAIN
            modulated = adain(feat_map, mean, std)

            # Decode
            recon = decoder(modulated)

            imgs_resized = F.interpolate(imgs, size=(64, 64))

            # Loss
            recon_loss = F.mse_loss(recon, imgs_resized)

            # Orthogonality loss
            orth_loss = orthogonality_loss(content_basis, style_basis)

            lambda_orth = 0.1
            loss = recon_loss + lambda_orth * orth_loss


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader):.4f}")
        utils.save_image(recon, f"recon_epoch_{epoch}.png", normalize=True)

    torch.save({
        'encoder': projector.state_dict()
    }, 'projector.pth')

if __name__ == "__main__":
    train()
