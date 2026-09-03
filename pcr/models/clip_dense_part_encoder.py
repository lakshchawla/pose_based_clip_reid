"""Turns CLIP's own frozen ViT into a dense, per-patch feature extractor (ClipViTDenseBackbone),
then classifies each patch into part maps two different ways:

  ClipDensePartEncoder -- zero-training: cosine similarity against a handful of fixed body-part
    text prompts. No BPAM checkpoint needed, but the raw similarity spread between classes turned
    out too small relative to per-patch noise for fine body-part discrimination on ReID crops
    (measured directly, not assumed) -- kept here for reference/comparison, not the current path.

  ClipViTBPAMEncoder -- supervised: a real, trained PixelToPartClassifier (bpbreid's own class,
    reused verbatim) sits on top of the frozen backbone's dense features and is trained with real
    masks (examples/train_bpa_segmentation_vit.py), exactly like today's Stage 0 does for HRNet/
    ResNet -- just with a frozen ViT backbone instead of a trainable CNN one, so only the small
    classifier head ever needs training.

Both share ClipViTDenseBackbone and honor BPBReIDEncoder's own interface (forward -> (f_out, vis),
forward_full adding pixels_cls_scores, num_features, num_parts) so either drops into every
existing consumer (Stage 1/2 training loops, the evaluator, Stage 3) unchanged. See the
"CLIP Backbone Pivot" artifact and this session's own discussion for the full design rationale
(CLIP Surgery / SCLIP's "V-V attention" trick for keeping per-patch identity through the last
transformer block instead of letting it collapse into one CLS summary).
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


class ClipViTBPAMEncoder(nn.Module):
    """Supervised path: ClipViTDenseBackbone (frozen) + a real, trained PixelToPartClassifier
    (bpbreid's own class, reused verbatim) -- trained by examples/train_bpa_segmentation_vit.py
    against real masks, exactly like today's HRNet/ResNet Stage 0, just with a frozen backbone so
    only the classifier's own small BN+1x1-conv head ever needs a gradient. Load the trained
    classifier's state_dict before using this for Stage 1/2 (see checkpoint_path)."""

    def __init__(self, clip_arch='ViT-B/16', height=256, width=128, num_parts=5,
                 checkpoint_path=None, device='cuda'):
        super(ClipViTBPAMEncoder, self).__init__()
        self.backbone = ClipViTDenseBackbone(clip_arch, height, width, device)
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
