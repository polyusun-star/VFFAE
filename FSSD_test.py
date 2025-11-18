import torch
import os
from torchvision import utils as vutils

from module.FSSD_CFG import (
    SubspaceProjector,
    FeatureMapper,
    SimpleDecoder,
    load_clip,
    load_image
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = 512
content_rank = 128
style_rank = 128
feature_map_dim = 256
style_dim = 256

save_model_dir = "./saved_models2"
save_dir = "./swap_results"
os.makedirs(save_dir, exist_ok=True)



projector = SubspaceProjector(input_dim, content_rank, style_rank).to(device)
mapper = FeatureMapper(input_dim, feature_map_dim, style_dim).to(device)
decoder = SimpleDecoder(feature_map_dim).to(device)

projector.load_state_dict(torch.load(os.path.join(save_model_dir, "projector.pth")))
mapper.load_state_dict(torch.load(os.path.join(save_model_dir, "mapper.pth")))
decoder.load_state_dict(torch.load(os.path.join(save_model_dir, "decoder.pth")))

projector.eval()
mapper.eval()
decoder.eval()


def extract_content_style(img_tensor, clip_model, projector, mapper):
    with torch.no_grad():
        clip_feat = clip_model.encode_image(img_tensor)
        clip_feat = F.normalize(clip_feat, dim=-1).float()
        z_c, z_s, _, _ = projector(clip_feat)
        content_feat, style_mean, style_std = mapper(z_c, z_s)
    return content_feat, style_mean, style_std

clip_model, preprocess = load_clip(device)

img1 = load_image("put your image1 here", preprocess, device)
img2 = load_image("put your image2 here", preprocess, device)

content1, style_mean1, style_std1 = extract_content_style(img1, clip_model, projector, mapper)
content2, style_mean2, style_std2 = extract_content_style(img2, clip_model, projector, mapper)
