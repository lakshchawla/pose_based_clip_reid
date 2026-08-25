"""Faithful port of the original CLIP paper's own training loss (Radford et al. 2021, section 2.3
/ the paper's own Numpy-pseudocode Algorithm 1) -- NOT CLIP-ReID's SupConLoss/I2TLoss variants
(pcr/loss/clip_supcon_loss.py, clip_i2t_loss.py). Row i's only positive is column i (the i-th
image pairs with the i-th text), matching the paper's own `labels = np.arange(n)` -- unlike
SupConLoss's identity-equality mask, this loss has no notion of "other same-identity samples in
this batch are also positives."

Used by examples/train_relational_clip.py, which explicitly asks for the CLIP paper's own loss in
place of CLIP-ReID's two-stage SupCon+I2T scheme. Known, accepted limitation: examples/
train_relational_clip.py's batches are PK-sampled (multiple images per identity, needed for the
triplet loss it also uses), so other same-identity images within a batch are treated as negatives
here -- exactly the behavior this loss's diagonal-only design implies, not a bug. This is the
tradeoff of following the CLIP paper's own loss literally rather than adapting it (as CLIP-ReID's
SupConLoss already does) to a labeled setting with repeated classes per batch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipContrastiveLoss(nn.Module):
    """Owns the learnable temperature (logit_scale) exactly as CLIP's own model does: initialized
    to log(1/0.07) (the paper's own starting point) and clamped to log(100) every forward pass
    (CLIP's own training-stability trick -- never let the temperature sharpen past 100)."""

    def __init__(self):
        super(ClipContrastiveLoss, self).__init__()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

    def forward(self, image_features, text_features):
        """image_features, text_features: [B, D]. L2-normalized here (not assumed pre-normalized
        by the caller) -- CLIP's own encode_image/encode_text always normalize immediately before
        computing logits, and the learned logit_scale is only calibrated correctly against
        unit-norm inputs (a residual-gated block like VisualAttentionBlock can otherwise leave
        this input's norm adrift from 1)."""
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        scale = self.logit_scale.exp().clamp(max=100)
        logits_per_image = scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        labels = torch.arange(image_features.size(0), device=image_features.device)
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        return (loss_i2t + loss_t2i) / 2
