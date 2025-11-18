import math
from os.path import basename, dirname, join, isfile
import torch
from torch import nn
from torch.nn import functional as nnf
from torch.nn.modules.activation import ReLU
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


def get_prompt_list(prompt):
    if prompt == 'plain':
        return ['{}']
    elif prompt == 'fixed':
        return ['a photo of a {}.']
    elif prompt == 'shuffle':
        return ['a photo of a {}.', 'a photograph of a {}.', 'an image of a {}.', '{}.']
    elif prompt == 'shuffle+':
        return ['a photo of a {}.', 'a photograph of a {}.', 'an image of a {}.', '{}.',
                'a cropped photo of a {}.', 'a good photo of a {}.', 'a photo of one {}.',
                'a bad photo of a {}.', 'a photo of the {}.']
    else:
        raise ValueError('Invalid value for prompt')

def forward_multihead_attention(x, b, with_aff=False, attn_mask=None):
    """
    Simplified version of multihead attention (taken from torch source code but without tons of if clauses).
    The mlp and layer norm come from CLIP.
    x: input.
    b: multihead attention module.
    """

    x_ = b.ln_1(x)
    q, k, v = nnf.linear(x_, b.attn.in_proj_weight, b.attn.in_proj_bias).chunk(3, dim=-1)
    tgt_len, bsz, embed_dim = q.size()

    head_dim = embed_dim // b.attn.num_heads
    scaling = float(head_dim) ** -0.5

    q = q.contiguous().view(tgt_len, bsz * b.attn.num_heads, b.attn.head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * b.attn.num_heads, b.attn.head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * b.attn.num_heads, b.attn.head_dim).transpose(0, 1)

    q = q * scaling

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))  # n_heads * batch_size, tokens^2, tokens^2
    if attn_mask is not None:

        attn_mask_type, attn_mask = attn_mask
        n_heads = attn_output_weights.size(0) // attn_mask.size(0)
        attn_mask = attn_mask.repeat(n_heads, 1)

        if attn_mask_type == 'cls_token':
            # the mask only affects similarities compared to the readout-token.
            attn_output_weights[:, 0, 1:] = attn_output_weights[:, 0, 1:] * attn_mask[None, ...]
            # attn_output_weights[:, 0, 0] = 0*attn_output_weights[:, 0, 0]

        if attn_mask_type == 'all':
            # print(attn_output_weights.shape, attn_mask[:, None].shape)
            attn_output_weights[:, 1:, 1:] = attn_output_weights[:, 1:, 1:] * attn_mask[:, None]

    attn_output_weights = torch.softmax(attn_output_weights, dim=-1)

    attn_output = torch.bmm(attn_output_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = b.attn.out_proj(attn_output)

    x = x + attn_output
    x = x + b.mlp(b.ln_2(x))

    if with_aff:
        return x, attn_output_weights
    else:
        return x


class CLIPDenseBase(nn.Module):

    def __init__(self, version, reduce_cond, reduce_dim, prompt, n_tokens):
        super().__init__()

        import clip

        # prec = torch.FloatTensor
        self.clip_model, _ = clip.load(version, device='cpu', jit=False)
        self.model = self.clip_model.visual

        # if not None, scale conv weights such that we obtain n_tokens.
        self.n_tokens = n_tokens

        for p in self.clip_model.parameters():
            p.requires_grad_(False)

        # conditional
        if reduce_cond is not None:
            self.reduce_cond = nn.Linear(512, reduce_cond)
            for p in self.reduce_cond.parameters():
                p.requires_grad_(False)
        else:
            self.reduce_cond = None

        self.film_mul = nn.Linear(512 if reduce_cond is None else reduce_cond, reduce_dim)
        self.film_add = nn.Linear(512 if reduce_cond is None else reduce_cond, reduce_dim)

        self.film_mul2 = nn.Linear(512 if reduce_cond is None else reduce_cond, reduce_dim)
        self.film_add2 = nn.Linear(512 if reduce_cond is None else reduce_cond, reduce_dim)

        self.reduce = nn.Linear(768, reduce_dim)

        self.prompt_list = get_prompt_list(prompt)

        # precomputed prompts
        import pickle
        if isfile('precomputed_prompt_vectors.pickle'):
            precomp = pickle.load(open('precomputed_prompt_vectors.pickle', 'rb'))
            self.precomputed_prompts = {k: torch.from_numpy(v) for k, v in precomp.items()}
        else:
            self.precomputed_prompts = dict()

    def rescaled_pos_emb(self, new_size):
        assert len(new_size) == 2

        a = self.model.positional_embedding[1:].T.view(1, 768, *self.token_shape)
        b = nnf.interpolate(a, new_size, mode='bicubic', align_corners=False).squeeze(0).view(768,
                                                                                              new_size[0] * new_size[
                                                                                                  1]).T
        return torch.cat([self.model.positional_embedding[:1], b])

    def visual_forward(self, x_inp, extract_layers=(), skip=False, mask=None):

        with torch.no_grad():

            inp_size = x_inp.shape[2:]

            if self.n_tokens is not None:
                stride2 = x_inp.shape[2] // self.n_tokens
                conv_weight2 = nnf.interpolate(self.model.conv1.weight, (stride2, stride2), mode='bilinear',
                                               align_corners=True)
                x = nnf.conv2d(x_inp, conv_weight2, bias=self.model.conv1.bias, stride=stride2,
                               dilation=self.model.conv1.dilation)
            else:
                x = self.model.conv1(x_inp)  # shape = [*, width, grid, grid]

            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

            x = torch.cat([self.model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                                dtype=x.dtype, device=x.device), x],
                          dim=1)  # shape = [*, grid ** 2 + 1, width]

            standard_n_tokens = 50 if self.model.conv1.kernel_size[0] == 32 else 197

            if x.shape[1] != standard_n_tokens:
                new_shape = int(math.sqrt(x.shape[1] - 1))
                x = x + self.rescaled_pos_emb((new_shape, new_shape)).to(x.dtype)[None, :, :]
            else:
                x = x + self.model.positional_embedding.to(x.dtype)

            x = self.model.ln_pre(x)

            x = x.permute(1, 0, 2)  # NLD -> LND

            activations, affinities = [], []
            for i, res_block in enumerate(self.model.transformer.resblocks):

                if mask is not None:
                    mask_layer, mask_type, mask_tensor = mask
                    if mask_layer == i or mask_layer == 'all':
                        # import ipdb; ipdb.set_trace()
                        size = int(math.sqrt(x.shape[0] - 1))

                        attn_mask = (mask_type, nnf.interpolate(mask_tensor.unsqueeze(1).float(), (size, size)).view(
                            mask_tensor.shape[0], size * size))

                    else:
                        attn_mask = None
                else:
                    attn_mask = None

                x, aff_per_head = forward_multihead_attention(x, res_block, with_aff=True, attn_mask=attn_mask)

                if i in extract_layers:
                    affinities += [aff_per_head]

                    # if self.n_tokens is not None:
                    #    activations += [nnf.interpolate(x, inp_size, mode='bilinear', align_corners=True)]
                    # else:
                    activations += [x]

                if len(extract_layers) > 0 and i == max(extract_layers) and skip:
                    print('early skip')
                    break

            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.model.ln_post(x[:, 0, :])

            if self.model.proj is not None:
                x = x @ self.model.proj

            return x, activations, affinities

    def sample_prompts(self, words, prompt_list=None):

        prompt_list = prompt_list if prompt_list is not None else self.prompt_list

        prompt_indices = torch.multinomial(torch.ones(len(prompt_list)), len(words), replacement=True)
        prompts = [prompt_list[i] for i in prompt_indices]
        return [promt.format(w) for promt, w in zip(prompts, words)]

    def get_cond_vec(self, conditional, batch_size):
        # compute conditional from a single string
        if conditional is not None and type(conditional) == str:
            cond = self.compute_conditional(conditional)
            cond = cond.repeat(batch_size, 1)

        # compute conditional from string list/tuple
        elif conditional is not None and type(conditional) in {list, tuple} and type(conditional[0]) == str:
            assert len(conditional) == batch_size
            cond = self.compute_conditional(conditional)

        # use conditional directly
        elif conditional is not None and type(conditional) == torch.Tensor and conditional.ndim == 2:
            cond = conditional

        # compute conditional from image
        elif conditional is not None and type(conditional) == torch.Tensor:
            with torch.no_grad():
                cond, _, _ = self.visual_forward(conditional)
        else:
            raise ValueError('invalid conditional')
        return cond

    def compute_conditional(self, conditional):
        import clip

        dev = next(self.parameters()).device

        if type(conditional) in {list, tuple}:
            text_tokens = clip.tokenize(conditional).to(dev)
            cond = self.clip_model.encode_text(text_tokens)
        else:
            if conditional in self.precomputed_prompts:
                cond = self.precomputed_prompts[conditional].float().to(dev)
            else:
                text_tokens = clip.tokenize([conditional]).to(dev)
                cond = self.clip_model.encode_text(text_tokens)[0]

        if self.shift_vector is not None:
            return cond + self.shift_vector
        else:
            return cond


def clip_load_untrained(version):
    assert version == 'ViT-B/16'
    from clip.model import CLIP
    from clip.clip import _MODELS, _download
    model = torch.jit.load(_download(_MODELS['ViT-B/16'])).eval()
    state_dict = model.state_dict()

    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len(
        [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    return CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
                context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)


class DeformableAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=2, num_points=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads


        # 注意力权重预测
        self.attn_conv = nn.Conv2d(
            embed_dim,
            num_heads * num_points,
            kernel_size=3,
            padding=1
        )

        # 输出投影
        self.proj = nn.Conv2d(embed_dim, embed_dim, 1)

    def forward(self, query, value, reference_points):
        """
        query/value: [B, C, H, W]
        reference_points: [B, H, W, 2] (归一化坐标)
        """
        B, C, H, W = query.shape

        # 预测注意力权重
        attn_weights = self.attn_conv(query)  # [B, num_heads*num_points, H, W]
        attn_weights = attn_weights.view(B, self.num_heads, self.num_points, H, W)
        attn_weights = F.softmax(attn_weights, dim=2)  # 对num_points维度归一化

        # 生成采样网格
        grid = self._generate_grid(reference_points)  # [B, num_heads*num_points, H, W, 2]

        # 执行可变形采样
        output = self._deform_attn_core(value, grid, attn_weights)

        return self.proj(output)

    def _generate_grid(self, offset_added_ref_points):
        # 输入维度: [B, H, W, num_heads, num_points, 2]
        B, H, W, num_heads, num_points, _ = offset_added_ref_points.shape
        offset_added_ref_points = offset_added_ref_points * 2 - 1
        grid = offset_added_ref_points.permute(0, 3, 4, 1, 2, 5).contiguous()  # -> [B, num_heads, num_points, H, W, 2]
        return grid.view(B, num_heads * num_points, H, W, 2)

    def _deform_attn_core(self, value, grid, attn_weights):
        B, C, H, W = value.shape
        num_heads = self.num_heads
        num_points = self.num_points
        head_dim = C // num_heads

        # 拆分 head: [B, num_heads, head_dim, H, W]
        value = value.view(B, num_heads, head_dim, H, W)

        grid = grid.view(B * num_heads * num_points, H, W, 2)  # [B*Np*Nh, H, W, 2]
        value = value.unsqueeze(2).expand(-1, -1, num_points, -1, -1, -1)  # [B, Nh, Np, C, H, W]
        value = value.contiguous().view(B * num_heads * num_points, head_dim, H, W)  # [B*Np*Nh, C, H, W]

        # grid_sample: [B*Np*Nh, C, H, W] and [B*Np*Nh, H, W, 2]
        sampled = F.grid_sample(
            value, grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )  # [B*Np*Nh, C, H, W]

        #[B, Nh, Np, C, H, W]
        sampled = sampled.view(B, num_heads, num_points, head_dim, H, W)

        # [B, Nh, C, H, W, Np]
        sampled = sampled.permute(0, 1, 3, 4, 5, 2)  # [B, Nh, C, H, W, Np]

        # attn_weights: [B, Nh, 1, H, W, Np]
        attn_weights = attn_weights.view(B, num_heads, 1, H, W, num_points)

        weighted_value = (sampled * attn_weights).sum(-1)  # [B, Nh, C, H, W]

        output = weighted_value.view(B, C, H, W)

        return output


import os
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


def visualize_position_map(inp_image, position_map, save_path=None):
    """
    Visualize predicted position_map keypoints on original input image.

    Args:
        inp_image: Tensor [B, 3, H, W] — original images (unnormalized or normalized with known mean/std)
        position_map: Tensor [B, h, p, 2] — normalized coords in [0,1]
        save_path: str or None — if given, saves the visualization
    """
    inp_image = inp_image.detach().cpu()
    position_map = position_map.detach().cpu()

    B, C, H, W = inp_image.shape
    _, h, p, _ = position_map.shape

    # If image is normalized, unnormalize it
    # Adjust the mean and std according to your preprocessing
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image = inp_image[0]
    if image.max() <= 1.0:  # assume it's normalized to [0,1] or standard normalize
        try:
            image = image * std + mean  # unnormalize if needed
        except:
            pass
        image = image.clamp(0, 1)  # clamp to avoid overflow

    image = TF.to_pil_image(image)

    plt.figure(figsize=(6, 6))
    plt.imshow(image)

    for head in range(h):
        for pt in range(p):
            x_norm, y_norm = position_map[0, head, pt]
            x = x_norm.item() * W
            y = y_norm.item() * H
            plt.scatter(x, y, c='red', s=15)

    plt.axis('off')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f"{save_path}_sample0.png", bbox_inches='tight')
    else:
        plt.show()
    plt.close()


import torch
from PIL import Image
import torchvision
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights


class FFAS(CLIPDenseBase):

    def __init__(self, version='ViT-B/32', extract_layers=(3, 6, 9), cond_layer=0, reduce_dim=128, n_heads=4,
                 prompt='fixed',
                 extra_blocks=0, reduce_cond=None, fix_shift=False,
                 learn_trans_conv_only=False, limit_to_clip_only=False, upsample=False,
                 add_calibration=False, rev_activations=False, trans_conv=None, n_tokens=None, complex_trans_conv=False,
                 deform_heads=2, deform_points=8):

        super().__init__(version, reduce_cond, reduce_dim, prompt, n_tokens)
        # device = 'cpu'

        self.extract_layers = extract_layers
        self.cond_layer = cond_layer
        self.limit_to_clip_only = limit_to_clip_only
        self.process_cond = None
        self.rev_activations = rev_activations

        self.deform_attn = DeformableAttention(
            embed_dim=reduce_dim,
            num_heads=deform_heads,
            num_points=deform_points
        )

        self.weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        self.pose_model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=self.weights).to("cuda")
        self.pose_model.eval()
        self.pose_transform = self.weights.transforms()

        depth = len(extract_layers)

        if add_calibration:
            self.calibration_conds = 1

        self.upsample_proj = nn.Conv2d(reduce_dim, 1, kernel_size=1) if upsample else None

        self.add_activation1 = True

        self.version = version

        self.token_shape = {'ViT-B/32': (7, 7), 'ViT-B/16': (14, 14)}[version]

        if fix_shift:
            # self.shift_vector = nn.Parameter(torch.load(join(dirname(basename(__file__)), 'clip_text_shift_vector.pth')), requires_grad=False)
            self.shift_vector = nn.Parameter(torch.load(join(dirname(basename(__file__)), 'shift_text_to_vis.pth')),
                                             requires_grad=False)
            # self.shift_vector = nn.Parameter(-1*torch.load(join(dirname(basename(__file__)), 'shift2.pth')), requires_grad=False)
        else:
            self.shift_vector = None

        if trans_conv is None:
            trans_conv_ks = {'ViT-B/32': (32, 32), 'ViT-B/16': (16, 16)}[version]
        else:
            # explicitly define transposed conv kernel size
            trans_conv_ks = (trans_conv, trans_conv)

        if not complex_trans_conv:
            self.trans_conv = nn.ConvTranspose2d(reduce_dim, 1, trans_conv_ks, stride=trans_conv_ks)
        else:
            assert trans_conv_ks[0] == trans_conv_ks[1]

            tp_kernels = (trans_conv_ks[0] // 4, trans_conv_ks[0] // 4)

            self.trans_conv = nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(reduce_dim, reduce_dim // 2, kernel_size=tp_kernels[0], stride=tp_kernels[0]),
                nn.ReLU(),
                nn.ConvTranspose2d(reduce_dim // 2, 1, kernel_size=tp_kernels[1], stride=tp_kernels[1]),
            )

        assert len(self.extract_layers) == depth

        self.reduces = nn.ModuleList([nn.Linear(768, reduce_dim) for _ in range(depth)])
        self.blocks = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=reduce_dim, nhead=n_heads) for _ in range(len(self.extract_layers))])
        self.extra_blocks = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=reduce_dim, nhead=n_heads) for _ in range(extra_blocks)])

        self.position_predictor = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(reduce_dim, deform_heads * deform_points * 2, kernel_size=1),
        )

        # refinement and trans conv

        if learn_trans_conv_only:
            for p in self.parameters():
                p.requires_grad_(False)

            for p in self.trans_conv.parameters():
                p.requires_grad_(True)

        self.prompt_list = get_prompt_list(prompt)

    def _get_ref_points(self, B, H, W, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5 / H, 1 - 0.5 / H, H, device=device),
            torch.linspace(0.5 / W, 1 - 0.5 / W, W, device=device)
        )
        ref_points = torch.stack((ref_x, ref_y), dim=-1)
        return ref_points.unsqueeze(0).repeat(B, 1, 1, 1)

    def forward(self, inp_image, conditional=None, return_features=False, mask=None):

        assert type(return_features) == bool

        inp_image = inp_image.to(self.model.positional_embedding.device)

        if mask is not None:
            raise ValueError('mask not supported')
        
        x_inp = inp_image

        bs, dev = inp_image.shape[0], x_inp.device

        cond = self.get_cond_vec(conditional, bs)

        visual_q, activations, _ = self.visual_forward(x_inp, extract_layers=[0] + list(self.extract_layers))

        prediction = self.pose_model(inp_image)
        
        pose_scores_batch = [item["scores"][0].to("cuda") if len(item["scores"]) > 0 else torch.tensor(0.0).to("cuda") for item in prediction]
        pose_score = torch.stack(pose_scores_batch)

        B, _, H2, W2 = inp_image.shape
        keypoints_list = []
        for i in range(B):
            if len(prediction[i]['keypoints']) == 0:
                dummy_points = torch.linspace(0, 1, steps=self.deform_attn.num_heads * self.deform_attn.num_points)
                dummy_grid = torch.stack(torch.meshgrid(dummy_points, dummy_points, indexing='ij'), dim=-1)
                dummy_grid = dummy_grid.view(-1, 2)[:self.deform_attn.num_heads * self.deform_attn.num_points]
                keypoints_list.append(dummy_grid.to(inp_image.device))
                continue

            keypoints = prediction[i]['keypoints'][0, :16, :2]
            keypoints_list.append(keypoints)

        keypoints_batch = torch.stack(keypoints_list, dim=0)
        keypoints_norm = keypoints_batch.clone()
        keypoints_norm[..., 0] /= W2
        keypoints_norm[..., 1] /= H2

        position_map = keypoints_norm.view(B, self.deform_attn.num_heads, self.deform_attn.num_points, 2)

        activation1 = activations[0]
        activations = activations[1:]

        _activations = activations[::-1] if not self.rev_activations else activations

        a = None
        for i, (activation, block, reduce) in enumerate(zip(_activations, self.blocks, self.reduces)):

            if a is not None:
                a = reduce(activation) + a
            else:
                a = reduce(activation)

            a_tem = a.permute(1, 0, 2)
            B, L, C = a_tem.shape
            H = W = int(math.sqrt(L - 1))

            spatial_feat = a_tem[:, 1:].permute(0, 2, 1).view(B, C, H, W)

            deform_feat = self.deform_attn(
                query=spatial_feat,
                value=spatial_feat,
                reference_points=position_map.unsqueeze(2).unsqueeze(2).expand(-1, -1, H, W, -1, -1)
            )
            deform_feat = deform_feat.flatten(2).permute(0, 2, 1)
            a_tem = torch.cat([a_tem[:, :1], deform_feat], dim=1)
            a = a_tem.permute(1, 0, 2) + a

            if i == self.cond_layer:
                if self.reduce_cond is not None:
                    cond = self.reduce_cond(cond)
                a = self.film_mul(cond) * a + self.film_add(cond)

            a = block(a)
            # print("the size of the lowest layer: ", a.shape)

        for block in self.extra_blocks:
            a = a + block(a)

        a = a[1:].permute(1, 2, 0)  # rm cls token and -> BS, Feats, Tokens

        size = int(math.sqrt(a.shape[2]))

        a = a.view(bs, a.shape[1], size, size)

        a = self.trans_conv(a)

        if self.n_tokens is not None:
            a = nnf.interpolate(a, x_inp.shape[2:], mode='bilinear', align_corners=True)

        if self.upsample_proj is not None:
            a = self.upsample_proj(a)
            a = nnf.interpolate(a, x_inp.shape[2:], mode='bilinear')

        if return_features:
            return a, visual_q, cond, [activation1] + activations
        else:
            return a, pose_score
