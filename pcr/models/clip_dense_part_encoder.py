"""Turns one of CLIP's own frozen visual towers into a dense, per-location feature extractor --
ClipViTDenseBackbone for ViT-family archs, ClipRN50DenseBackbone for RN-family archs (both of
CLIP-ReID's own two officially-pretrained backbone choices) -- then classifies each location into
part maps two different ways:

  ClipDensePartEncoder -- zero-training: cosine similarity against a handful of fixed body-part
    text prompts. No BPAM checkpoint needed, but the raw similarity spread between classes turned
    out too small relative to per-patch noise for fine body-part discrimination on ReID crops
    (measured directly, not assumed) -- kept here for reference/comparison, not the current path.
    ViT-only (never ported to RN50).

  ClipBPAMEncoder -- supervised, backbone-agnostic: a real, trained PixelToPartClassifier
    (bpbreid's own class, reused verbatim) sits on top of whichever frozen dense backbone it's
    given and is trained with real masks (examples/train_bpa_segmentation_vit.py /
    train_bpa_segmentation_rn50.py), exactly like today's Stage 0 does for HRNet/ResNet -- just
    with a frozen CLIP backbone instead of a trainable CNN one, so only the small classifier head
    ever needs training. ClipViTBPAMEncoder/ClipRN50BPAMEncoder are thin constructors around it.

Both dense backbones solve the same underlying problem -- CLIP was pretrained to collapse every
image down to ONE embedding, but BPAM needs each spatial location to keep its own -- with a
surgery specific to where that collapse actually happens in each architecture: ViT's last
transformer block (Q-K attention swapped for V-V/"value-based" attention, CLIP Surgery/SCLIP's
trick) vs. RN50's AttentionPool2d (re-invoked with every location as query instead of only the
mean-pooled token, using its own pretrained weights unchanged). All three encoder classes honor
BPBReIDEncoder's own interface (forward -> (f_out, vis), forward_full adding pixels_cls_scores,
num_features, num_parts) so any of them drops into every existing consumer (Stage 1/2 training
loops, the evaluator, Stage 3) unchanged. See the "CLIP Backbone Pivot" artifact and this
session's own discussion for the full design rationale.
"""
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchreid.models.bpbreid import PixelToPartClassifier

# CLIP's own image normalization -- NOT BPBreID's ImageNet stats. Pretrained CLIP weights are
# calibrated for this specific normalization; using the wrong one silently degrades every
# downstream similarity score.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _interpolate_pos_embed(pos_embed, orig_grid, new_grid):
    """pos_embed: [1 + orig_grid*orig_grid, width]. Splits off the CLS position (index 0, kept
    as-is) and bicubic-interpolates the square grid part to new_grid x new_grid -- CLIP's own ViT
    positional embedding is trained for a fixed square input resolution (224px -> 14x14 for
    patch16), and ReID crops are neither square nor that resolution."""
    cls_pos, grid_pos = pos_embed[:1], pos_embed[1:]
    width = grid_pos.size(-1)
    grid_pos = grid_pos.reshape(1, orig_grid, orig_grid, width).permute(0, 3, 1, 2)
    grid_pos = F.interpolate(grid_pos, size=new_grid, mode='bicubic', align_corners=False)
    grid_pos = grid_pos.permute(0, 2, 3, 1).reshape(new_grid[0] * new_grid[1], width)
    return torch.cat([cls_pos, grid_pos], dim=0)


class ClipViTDenseBackbone(nn.Module):
    """Frozen CLIP ViT, modified only at the last transformer block (Q/K attention -> V-V
    attention) so per-patch identity survives to the output instead of being mixed away. Returns
    raw, pre-projection dense tokens at the ViT's own internal width (768 for ViT-B/16) -- the
    same convention BPBreID's own pixel classifier uses (classifies on the backbone's raw width,
    before any dimension reduction). `project()` below applies CLIP's own ln_post+proj afterward,
    only when something needs the text-aligned 512-dim joint space (pooling into f_out, or the
    zero-shot cosine-similarity path)."""

    def __init__(self, clip_arch='ViT-B/16', height=256, width=128, device='cuda'):
        super(ClipViTDenseBackbone, self).__init__()
        clip_model, _ = clip.load(clip_arch, device=device, jit=False)
        visual = clip_model.visual
        assert hasattr(visual, 'transformer'), (
            "needs a ViT-based CLIP arch (has patch tokens to keep) -- RN50-style CLIP visual "
            "towers have no per-patch token sequence to preserve.")

        self.conv1 = visual.conv1
        self.class_embedding = visual.class_embedding
        self.ln_pre = visual.ln_pre
        self.resblocks = visual.transformer.resblocks
        self.ln_post = visual.ln_post
        self.proj = visual.proj
        self.dtype = clip_model.dtype
        self.num_heads = self.resblocks[-1].attn.num_heads
        self.vision_width = self.conv1.out_channels
        self.embed_dim = self.proj.shape[1]

        patch_size = self.conv1.kernel_size[0]
        orig_grid = visual.input_resolution // patch_size
        new_grid = (height // patch_size, width // patch_size)
        pos_embed = _interpolate_pos_embed(visual.positional_embedding.float(), orig_grid, new_grid)
        self.register_buffer('positional_embedding', pos_embed.type(self.dtype))
        self.grid_h, self.grid_w = new_grid

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, images):
        """images: [B, 3, H, W], CLIP-normalized (see CLIP_MEAN/CLIP_STD). Returns
        patch_feats [B, grid_h*grid_w, vision_width] (raw, not L2-normalized, CLS dropped) --
        every patch individually carried to the output, not blurred into one summary vector."""
        x = images.type(self.dtype)
        x = self.conv1(x)  # [B, vision_width, grid_h, grid_w]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B, grid_h*grid_w, vision_width]
        cls = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)  # [B, 1+grid_h*grid_w, vision_width]
        x = x + self.positional_embedding
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND, matches CLIP's own internal layout
        for block in self.resblocks[:-1]:
            x = block(x)  # normal MHSA -- unchanged, this is where per-patch content is built

        # Last block only: replace Q/K-based attention with V-V attention (CLIP Surgery / SCLIP)
        # so this final step stops mixing patches into each other -- every patch keeps its own
        # identity through to the output instead of collapsing toward one CLS summary.
        last = self.resblocks[-1]
        normed = last.ln_1(x)  # [L, B, vision_width]
        L, B, D = normed.shape
        Wv = last.attn.in_proj_weight[2 * D:3 * D]
        bv = last.attn.in_proj_bias[2 * D:3 * D]
        v = F.linear(normed, Wv, bv)  # [L, B, D]
        head_dim = D // self.num_heads
        v_heads = v.reshape(L, B * self.num_heads, head_dim).permute(1, 0, 2)  # [B*heads, L, head_dim]
        attn = torch.softmax((v_heads @ v_heads.transpose(-1, -2)) / (head_dim ** 0.5), dim=-1)
        out = attn @ v_heads  # [B*heads, L, head_dim]
        out = out.permute(1, 0, 2).reshape(L, B, D)
        out = last.attn.out_proj(out)
        x = x + out
        x = x + last.mlp(last.ln_2(x))

        x = x.permute(1, 0, 2)  # LND -> NLD
        return x[:, 1:, :].float()  # drop CLS -- raw, pre-projection, vision_width-dim

    def project(self, patch_feats):
        """patch_feats: [..., vision_width] (raw, from forward()). Applies CLIP's own ln_post +
        proj -- the exact op CLIP normally reserves for the CLS token alone -- landing every
        token in the 512-dim text-aligned joint space. Cast back to float32 (CLIP's own
        projection runs in fp16); downstream matmuls (zero-shot cosine similarity) need it to
        match whatever else they're multiplied against, and float32 is this repo's convention
        for every embedding a loss touches."""
        return (self.ln_post(patch_feats.type(self.dtype)) @ self.proj).float()


def _gwap_pool(mask, feats):
    """mask: [B, N] (per-branch pooling weight per patch). feats: [B, N, D]. Same weighted-average
    formula as bpbreid.py's own GlobalWeightedAveragePoolingHead: sum(mask*feat) / sum(mask)."""
    weighted = mask.unsqueeze(-1) * feats
    return weighted.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1e-6)


class ClipRN50DenseBackbone(nn.Module):
    """Frozen CLIP RN50 (ModifiedResNet), modified at two points so per-location identity survives
    to the output instead of being discarded the way stock CLIP discards it:

    1. layer4's stride is dropped from 2 to 1 (the standard ReID "last-stride" trick -- the same
       one BPBreID's own `last_stride` config applies to its resnet50/hrnet32 backbones), keeping
       the final grid 2x finer for small body parts. CLIP's own Bottleneck (third_party/clip/
       model.py) encodes stride entirely via parameter-free nn.AvgPool2d(stride) layers -- every
       conv in it is stride=1 always -- so this is done by swapping those two AvgPool2d(2)
       instances (the main branch's post-conv2 one and the shortcut's pre-conv one) for
       nn.Identity(); no pretrained weight is touched or needs retraining.
    2. AttentionPool2d's own forward only ever lets ONE token (the spatial mean) serve as query,
       discarding every individual location's own post-attention representation (`query=x[:1]` in
       third_party/clip/model.py -- verified directly in the installed package, not assumed).
       `project()` below re-invokes that same layer's pretrained q_proj/k_proj/v_proj/c_proj
       weights, completely unchanged, but with every location serving as both query and key/value
       -- so every patch keeps its own joint-space embedding instead of collapsing into one vector.

    Same interface as ClipViTDenseBackbone (forward -> raw [B,N,vision_width], project -> joint
    [B,N,embed_dim], grid_h/grid_w) so ClipBPAMEncoder below is backbone-agnostic between the two.
    """

    def __init__(self, clip_arch='RN50', height=256, width=128, device='cuda'):
        super(ClipRN50DenseBackbone, self).__init__()
        clip_model, _ = clip.load(clip_arch, device=device, jit=False)
        visual = clip_model.visual
        assert hasattr(visual, 'attnpool'), (
            "needs an RN-family CLIP arch (conv backbone + AttentionPool2d) -- ViT-based CLIP "
            "visual towers have no such pooling head (see ClipViTDenseBackbone instead).")

        self.dtype = clip_model.dtype
        self.conv1, self.bn1, self.relu1 = visual.conv1, visual.bn1, visual.relu1
        self.conv2, self.bn2, self.relu2 = visual.conv2, visual.bn2, visual.relu2
        self.conv3, self.bn3, self.relu3 = visual.conv3, visual.bn3, visual.relu3
        self.avgpool = visual.avgpool
        self.layer1 = visual.layer1
        self.layer2 = visual.layer2
        self.layer3 = visual.layer3
        self.layer4 = visual.layer4
        self._drop_last_stride(self.layer4)

        self.attnpool = visual.attnpool
        self.vision_width = self.attnpool.k_proj.in_features   # 2048 for RN50: channels into attnpool
        self.embed_dim = self.attnpool.c_proj.out_features     # 1024 for RN50: final joint-space dim
        self.num_heads = self.attnpool.num_heads

        # Stock total downsampling is 32x (stem 4x, layer2 2x, layer3 2x, layer4 2x); dropping
        # layer4's stride above makes it 16x -- the grid attnpool's own pretrained
        # positional_embedding was fit to (visual.input_resolution // 32) is smaller than the one
        # actually reached now, same bicubic-interpolation fix ClipViTDenseBackbone already needed
        # for its own (non-square, non-224px) grid mismatch.
        orig_grid = visual.input_resolution // 32
        new_grid = (height // 16, width // 16)
        pos_embed = _interpolate_pos_embed(self.attnpool.positional_embedding.float(), orig_grid, new_grid)
        self.register_buffer('positional_embedding', pos_embed.type(self.dtype))
        self.grid_h, self.grid_w = new_grid

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @staticmethod
    def _drop_last_stride(layer4):
        first_block = layer4[0]
        first_block.avgpool = nn.Identity()
        if first_block.downsample is not None:
            first_block.downsample[0] = nn.Identity()  # the "-1" AvgPool2d(stride) entry

    def forward(self, images):
        """images: [B, 3, H, W], CLIP-normalized. Returns patch_feats [B, grid_h*grid_w,
        vision_width] (raw conv features, pre attention-pool) -- same convention as
        ClipViTDenseBackbone.forward: every location individually carried to the output."""
        x = images.type(self.dtype)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [B, vision_width, grid_h, grid_w]
        B, C, H, W = x.shape
        return x.reshape(B, C, H * W).permute(0, 2, 1).float()  # [B, N, vision_width]

    def project(self, patch_feats):
        """patch_feats: [B, N, vision_width] (raw, from forward()). Re-runs AttentionPool2d's own
        pretrained weights with every location as both query and key/value (see class docstring
        point 2) instead of stock CLIP's mean-token-only query. Cast to float32 to match this
        repo's convention for every embedding a loss touches."""
        B, N, C = patch_feats.shape
        x = patch_feats.permute(1, 0, 2).type(self.dtype)  # NLC -> LNC, matches attnpool's own layout
        mean_tok = x.mean(dim=0, keepdim=True)
        x = torch.cat([mean_tok, x], dim=0)  # [1+N, B, C]
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        ap = self.attnpool
        out, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1], num_heads=self.num_heads,
            q_proj_weight=ap.q_proj.weight, k_proj_weight=ap.k_proj.weight, v_proj_weight=ap.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([ap.q_proj.bias, ap.k_proj.bias, ap.v_proj.bias]),
            bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0.0,
            out_proj_weight=ap.c_proj.weight, out_proj_bias=ap.c_proj.bias,
            use_separate_proj_weight=True, training=False, need_weights=False)
        return out[1:].permute(1, 0, 2).float()  # drop the mean token, LNC -> NLC: [B, N, embed_dim]


class ClipBPAMEncoder(nn.Module):
    """Backbone-agnostic supervised BPAM encoder: any frozen dense CLIP backbone exposing
    forward()->[B,N,vision_width], project()->[B,N,embed_dim] and grid_h/grid_w (ViT or RN50,
    ClipViTDenseBackbone/ClipRN50DenseBackbone above), plus a real, trained PixelToPartClassifier
    (bpbreid's own class, reused verbatim) on top -- trained against real masks by
    examples/train_bpa_segmentation_vit.py/train_bpa_segmentation_rn50.py, exactly like today's
    HRNet/ResNet Stage 0, just with a frozen backbone so only the classifier's own small
    BN+1x1-conv head ever needs a gradient. Load the trained classifier's state_dict before using
    this for Stage 1/2 (see checkpoint_path). ClipViTBPAMEncoder/ClipRN50BPAMEncoder below are
    thin constructors that build the right backbone; this class's own logic doesn't care which one
    it got."""

    def __init__(self, backbone, num_parts=5, checkpoint_path=None, device='cuda'):
        super(ClipBPAMEncoder, self).__init__()
        self.backbone = backbone
        self.num_parts = num_parts
        self.num_features = self.backbone.embed_dim
        self.pixel_classifier = PixelToPartClassifier(self.backbone.vision_width, num_parts).to(device)
        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location=device)
            self.pixel_classifier.load_state_dict(state['pixel_classifier'] if 'pixel_classifier' in state else state)

    def _forward_common(self, images):
        patch_feats = self.backbone(images)  # [B, N, vision_width], raw
        B, N, D = patch_feats.shape
        grid = patch_feats.permute(0, 2, 1).reshape(B, D, self.backbone.grid_h, self.backbone.grid_w)
        pixels_cls_scores = self.pixel_classifier(grid)  # [B, 1+K, grid_h, grid_w]
        probs = F.softmax(pixels_cls_scores, dim=1).reshape(B, 1 + self.num_parts, N).permute(0, 2, 1)  # [B,N,1+K]

        joint_feats = F.normalize(self.backbone.project(patch_feats), p=2, dim=-1)  # [B, N, embed_dim]

        background_masks = probs[:, :, 0]
        parts_masks = probs[:, :, 1:]
        foreground_masks = parts_masks.amax(dim=-1)

        foreground_emb = _gwap_pool(foreground_masks, joint_feats)
        part_embs = torch.stack([_gwap_pool(parts_masks[:, :, k], joint_feats) for k in range(self.num_parts)], dim=1)
        f_out = torch.cat([foreground_emb.unsqueeze(1), part_embs], dim=1)
        f_out = F.normalize(f_out, p=2, dim=-1)

        parts_visibility = parts_masks.amax(dim=1)
        foreground_visibility = parts_visibility.amax(dim=1)
        vis = torch.cat([foreground_visibility.unsqueeze(1), parts_visibility], dim=1)

        return f_out, vis, pixels_cls_scores

    def forward(self, images):
        f_out, vis, _ = self._forward_common(images)
        return f_out, vis

    def forward_full(self, images):
        return self._forward_common(images)


def ClipViTBPAMEncoder(clip_arch='ViT-B/16', height=256, width=128, num_parts=5,
                        checkpoint_path=None, device='cuda'):
    backbone = ClipViTDenseBackbone(clip_arch, height, width, device)
    return ClipBPAMEncoder(backbone, num_parts, checkpoint_path, device)


def ClipRN50BPAMEncoder(clip_arch='RN50', height=256, width=128, num_parts=5,
                         checkpoint_path=None, device='cuda'):
    backbone = ClipRN50DenseBackbone(clip_arch, height, width, device)
    return ClipBPAMEncoder(backbone, num_parts, checkpoint_path, device)


class ClipDensePartEncoder(nn.Module):
    """Zero-training path -- see module docstring. Kept for reference/comparison; not the current
    path (see ClipViTBPAMEncoder)."""

    def __init__(self, clip_arch='ViT-B/16', height=256, width=128, parts=None,
                 similarity_scale=None, device='cuda'):
        super(ClipDensePartEncoder, self).__init__()
        self.backbone = ClipViTDenseBackbone(clip_arch, height, width, device)
        clip_model, _ = clip.load(clip_arch, device=device, jit=False)
        parts = parts or ['head', 'chest', 'hips', 'thighs', 'feet']
        self.num_parts = len(parts)
        self.num_features = self.backbone.embed_dim
        prompts = ["a photo of background"] + ["a photo of a person's {}".format(p) for p in parts]
        tokenized = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            anchors = F.normalize(clip_model.encode_text(tokenized).float(), p=2, dim=-1)
        self.register_buffer('part_name_anchors', anchors)  # [1+K, embed_dim], index 0 = background
        self.register_buffer('logit_scale', torch.tensor(
            similarity_scale if similarity_scale is not None else clip_model.logit_scale.exp().item()))
        self.eval()

    def _forward_common(self, images):
        patch_feats = F.normalize(self.backbone.project(self.backbone(images)), p=2, dim=-1)
        B, N, D = patch_feats.shape
        sims = patch_feats @ self.part_name_anchors.t() * self.logit_scale
        probs = F.softmax(sims, dim=-1)

        background_masks = probs[:, :, 0]
        parts_masks = probs[:, :, 1:]
        foreground_masks = parts_masks.amax(dim=-1)

        foreground_emb = _gwap_pool(foreground_masks, patch_feats)
        part_embs = torch.stack([_gwap_pool(parts_masks[:, :, k], patch_feats) for k in range(self.num_parts)], dim=1)
        f_out = torch.cat([foreground_emb.unsqueeze(1), part_embs], dim=1)
        f_out = F.normalize(f_out, p=2, dim=-1)

        parts_visibility = parts_masks.amax(dim=1)
        foreground_visibility = parts_visibility.amax(dim=1)
        vis = torch.cat([foreground_visibility.unsqueeze(1), parts_visibility], dim=1)
        return f_out, vis, probs

    def forward(self, images):
        f_out, vis, _ = self._forward_common(images)
        return f_out, vis

    def forward_full(self, images):
        patch_feats_shape_probs = self._forward_common(images)
        f_out, vis, probs = patch_feats_shape_probs
        pixels_cls_scores = probs.permute(0, 2, 1).reshape(-1, 1 + self.num_parts, self.backbone.grid_h, self.backbone.grid_w)
        return f_out, vis, pixels_cls_scores
