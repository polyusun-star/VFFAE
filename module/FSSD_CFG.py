import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import clip


class SubspaceProjector(nn.Module):
    def __init__(self, input_dim, content_rank, style_rank):
        super().__init__()
        self.input_dim = input_dim
        self.content_rank = content_rank
        self.style_rank = style_rank
        self.content_basis = nn.Parameter(torch.randn(input_dim, content_rank))
        self.style_basis = nn.Parameter(torch.randn(input_dim, style_rank))

    def forward(self, z):
        content_basis = F.normalize(self.content_basis, dim=0)
        style_basis = F.normalize(self.style_basis, dim=0)
        z_c = z @ content_basis @ content_basis.T
        z_s = z @ style_basis @ style_basis.T
        return z_c, z_s, content_basis, style_basis


class FeatureMapper(nn.Module):
    def __init__(self, input_dim, feature_map_dim, style_dim):
        super().__init__()
        self.content_mlp = nn.Sequential(
            nn.Linear(input_dim, feature_map_dim * 8 * 8),
            nn.ReLU(),
        )
        self.style_mlp = nn.Sequential(
            nn.Linear(input_dim, style_dim * 2),
        )
        self.feature_map_dim = feature_map_dim
        self.style_dim = style_dim

    def forward(self, content_vec, style_vec):
        feature_map = self.content_mlp(content_vec)
        feature_map = feature_map.view(-1, self.feature_map_dim, 8, 8)
        style_stats = self.style_mlp(style_vec)
        mean, std = style_stats.chunk(2, dim=1)
        return feature_map, mean.unsqueeze(-1).unsqueeze(-1), std.unsqueeze(-1).unsqueeze(-1)


def adain(content_feat, style_mean, style_std, eps=1e-5):
    c_mean = content_feat.mean(dim=[2, 3], keepdim=True)
    c_std = content_feat.std(dim=[2, 3], keepdim=True) + eps
    normalized = (content_feat - c_mean) / c_std
    return normalized * style_std + style_mean


class SimpleDecoder(nn.Module):
    def __init__(self, feature_map_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_map_dim, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=4, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(x)


# ----------------- 工具函数 -----------------
def load_clip(device="cuda"):
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    return clip_model.eval(), preprocess


def load_image(image_path, preprocess, device):
    img = Image.open(image_path).convert("RGB")
    img = preprocess(img).unsqueeze(0).to(device)
    return img


def decode(content_feat, style_mean, style_std, decoder):
    fused = adain(content_feat, style_mean, style_std)
    out = decoder(fused)
    out = F.interpolate(out, size=(224, 224), mode="bilinear", align_corners=False)
    return out


def decode_only(content_feat, decoder):
    out = decoder(content_feat)
    out = F.interpolate(out, size=(224, 224), mode="bilinear", align_corners=False)
    return out
