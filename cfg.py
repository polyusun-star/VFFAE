
import torch
import torch.nn.functional as F
from torchvision import transforms

from module.FSSD_CFG import (
    SubspaceProjector,
    FeatureMapper,
    SimpleDecoder,
    load_clip,
    extract_content_style,
    adain
)


class CLIPFeaturePredictor:
    def __init__(
        self,
        model_dir: str = './saved_models2',
        device: str = None
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_dir = model_dir

        self._load_models()
        self._build_transform()

    def _load_models(self):

        self.clip_model, self.preprocess_clip = load_clip(self.device)

        input_dim = 512
        content_rank = 128
        style_rank = 128
        feature_map_dim = 256
        style_dim = 256

        self.projector = SubspaceProjector(input_dim, content_rank, style_rank).to(self.device)
        self.mapper = FeatureMapper(input_dim, feature_map_dim, style_dim).to(self.device)
        self.decoder = SimpleDecoder(feature_map_dim).to(self.device)

        # Load weights
        projector_ckpt = torch.load(f"{self.model_dir}/projector.pth", map_location=self.device)

        self.projector.load_state_dict(projector_ckpt)

        self.projector.eval()

    # ------------------------------------------------
    #            CLIP IMAGE PREPROCESSING
    # ------------------------------------------------
    def _build_transform(self):
        # This transform is ONLY for images passed manually into predict()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                 [0.26862954, 0.26130258, 0.27577711])
        ])

    def preprocess(self, img):
        if not isinstance(img, torch.Tensor):
            img = self.transform(img)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        return img.to(self.device)

    # ------------------------------------------------
    #                MAIN PREDICTION
    # ------------------------------------------------
    def predict(self, imgA):
        imgA = self.preprocess(imgA)

        with torch.no_grad():

            clip_featA = self.clip_model.encode_image(imgA)

            clip_featA = F.normalize(clip_featA, dim=-1)

            z_cA, z_sA, _, _ = self.projector(clip_featA)

        return z_cA, z_sA# return decomposed vectors as CFG condition

