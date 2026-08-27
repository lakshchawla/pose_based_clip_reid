"""Faithful port of "Algorithm 2 -- Stage 2: Backbone Fine-Tuning" step 14's own alignment term:
`1 - cosine_sim(relation_feats[:,k], frozen_text_anchors[label,k])` -- NOT CLIP-ReID's I2TLoss
(pcr/loss/clip_i2t_loss.py, removed: this was Stage 2's only caller), which is a full
label-smoothed cross-entropy classification against the *entire* frozen prototype table (every
other identity's anchor acts as an implicit negative via the softmax). This loss has no such
notion -- it is a direct, per-sample regression toward the sample's own identity's anchor only;
no other identity's prototype participates at all.

Unlike I2TLoss (which takes the whole [num_identities, D] prototype table and an index tensor,
since it needs every row for its softmax), this loss takes pre-gathered per-sample anchors --
the caller indexes text_prototypes[targets, k, :] before calling this, since there is nothing
else in the table this loss needs to see.

Per-sample `weights` (required, not optional -- this class has exactly one caller,
examples/train_relational_finetune.py, which always has a per-branch visibility score available
once Stage 2's own encoder construction switches to continuous visibility scores): replaces the
upstream image-level visibility filter this repo used to run before Stage 2 ever saw an image.
Every image now enters training; a branch with low visibility for a given sample contributes
proportionally less to that branch's own alignment term instead of the whole image being rejected
outright. Weights are floored (`weight_floor`, default 1e-3) so no sample is ever exactly
zero-weighted, matching pcr/loss/clip_infonce_loss.py::InfoNCELoss's identical convention.
"""
import torch.nn as nn
import torch.nn.functional as F


class CosineAlignLoss(nn.Module):
    def __init__(self, weight_floor=1e-3):
        super(CosineAlignLoss, self).__init__()
        self.weight_floor = weight_floor

    def forward(self, visual_features, text_anchors, weights):
        """visual_features: [B, D]. text_anchors: [B, D], already gathered for each sample's own
        identity/branch. weights: [B] float, per-image visibility score for this branch. Returns
        the weighted mean of 1 - cosine_similarity."""
        per_sample = 1 - F.cosine_similarity(visual_features, text_anchors, dim=-1)  # [B]
        w = weights.clamp(min=self.weight_floor)
        return (w * per_sample).sum() / w.sum().clamp(min=1e-8)
