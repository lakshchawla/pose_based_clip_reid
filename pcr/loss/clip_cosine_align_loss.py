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
"""
import torch.nn as nn
import torch.nn.functional as F


class CosineAlignLoss(nn.Module):
    def forward(self, visual_features, text_anchors):
        """visual_features: [B, D]. text_anchors: [B, D], already gathered for each sample's own
        identity/branch. Returns the batch-mean of 1 - cosine_similarity."""
        return (1 - F.cosine_similarity(visual_features, text_anchors, dim=-1)).mean()
