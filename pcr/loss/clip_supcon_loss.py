"""Ported faithfully from CLIP-ReID's ../CLIP-ReID/loss/supcontrast.py::SupConLoss -- this is
CLIP-ReID's actual Stage-1 loss, and it is NOT plain diagonal InfoNCE. The positive mask is built
from identity equality across the two feature sets (every same-identity pair is a positive, not
only the same-index pair), so an anchor can have multiple positives; `mask.sum(1)` normalizes the
log-prob sum over however many positives that anchor has. Applied symmetrically (call once per
direction with the two feature sets swapped) to get CLIP-ReID's loss_i2t + loss_t2i.

Requires every anchor to have at least one positive (mask.sum(1) > 0), same precondition the
original relies on without a clamp/epsilon -- guaranteed by construction as long as both label
arguments come from the same batch of identities (CLIP-ReID's own stage-1 loop always calls this
with `t_label=target, i_targets=target`, the identical tensor for both sides of one batch, so
every anchor's own same-index counterpart is always a positive). Callers here must preserve that
invariant rather than passing disjoint label sets.
"""
import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, anchor_features, other_features, anchor_labels, other_labels):
        """anchor_features: [Ba, D]. other_features: [Bo, D]. anchor_labels: [Ba]. other_labels:
        [Bo]."""
        mask = torch.eq(anchor_labels.unsqueeze(1), other_labels.unsqueeze(0)).float()
        logits = (anchor_features @ other_features.t()) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        return -mean_log_prob_pos.mean()
