"""Ported from CLIP-ReID's ../CLIP-ReID/loss/supcontrast.py::SupConLoss -- CLIP-ReID's actual
Stage-1 loss, and not plain diagonal InfoNCE. Restored here (2026-08-28) after a round-trip through
`InfoNCELoss` (removed): this repo's own PK-sampled batches (multiple images per identity in the
same batch, on purpose -- see examples/train_relational_prompts.py's own docstring) are exactly the
situation SupCon was designed for, and exactly the situation InfoNCE was not. InfoNCELoss's own
docstring already documented the mismatch it had to patch around (deduplicating a PK batch down to
one representative image per identity, to avoid treating a person's own other photos as false
negatives) -- that patch worked, but it meant only one of a person's several photos in a batch ever
shaped the text-side gradient, and the multiple-choice comparison shrank to only the handful of
unique identities in one batch (~8, with this repo's batch_size=32/num_instances=4), not the full
training set. SupCon needs no such patch: the positive mask below is built from real identity
labels, so an anchor's own siblings in the batch are correctly recognized as positives (not
negatives) and all of them contribute to that anchor's gradient in one calculation, and every image
in the batch (not just deduplicated representatives) stays a comparison point.

The positive mask is built from identity equality across the two feature sets (every same-identity
pair is a positive, not only the same-index pair), so an anchor can have multiple positives;
`mask.sum(1)` normalizes the log-prob sum over however many positives that anchor has. Applied
symmetrically (call once per direction with the two feature sets swapped) to get CLIP-ReID's
loss_i2t + loss_t2i -- unlike InfoNCELoss, which folded both directions into one call, this keeps
the original file's own two-call convention, since neither side needs deduplication here.

Requires every anchor to have at least one positive (mask.sum(1) > 0), same precondition the
original relies on without a clamp/epsilon -- guaranteed by construction as long as both label
arguments come from the same batch of identities (this repo's own stage-1 loop always calls this
with `anchor_labels=other_labels=b_labels`, the identical tensor for both sides of one batch, so
every anchor's own same-index counterpart is always a positive).

Per-sample `weights` (required, not optional -- matches this repo's other Stage 1/2 losses'
convention): each row's own per-part visibility score, weighting how much that row's own log-
probability counts toward the batch mean -- a poorly-visible part still participates fully as a
comparison point for every other row (SupCon's multi-positive mechanism is left untouched), it just
counts less as an anchor in its own right. Detached before use (`weights.detach()`) for the same
reason documented in pcr/loss/clip_cosine_align_loss.py's own docstring: Stage 1's `build_encoder`
uses continuous (differentiable, in general) visibility scores, so leaving a live gradient path
through the weight would let the model reduce this loss by declaring a hard-to-align row "less
visible" instead of actually aligning it better. (Not currently exploitable in Stage 1 specifically,
since `cached_visibility` is built entirely under `torch.no_grad()` before training starts -- but
the detach makes that a guarantee of this class, not an accident of the caller, same rationale as
CosineAlignLoss.) Weights are additionally floored (`weight_floor`, default 1e-3) so no anchor is
ever exactly zero-weighted.

Temperature default restored to `1.0` (CLIP-ReID's own original value for this exact loss) --
`InfoNCELoss` used `0.07` instead specifically because CLIP's `0.07` was tuned for a plain
single-positive contrastive loss, not this multi-positive one (see InfoNCELoss's own now-removed
docstring, or progress.md's entry on that swap); restoring SupCon without also restoring its
matching temperature would silently run a multi-positive loss at a single-positive loss's much
sharper (smaller) temperature, likely destabilizing training rather than helping it.
"""
import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    def __init__(self, temperature=1.0, weight_floor=1e-3):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.weight_floor = weight_floor

    def forward(self, anchor_features, other_features, anchor_labels, other_labels, weights):
        """anchor_features: [Ba, D]. other_features: [Bo, D]. anchor_labels: [Ba]. other_labels:
        [Bo]. weights: [Ba] float, each anchor row's own visibility score for this call's part/
        direction."""
        mask = torch.eq(anchor_labels.unsqueeze(1), other_labels.unsqueeze(0)).float()
        logits = (anchor_features @ other_features.t()) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        w = weights.detach().clamp(min=self.weight_floor)
        return -(w * mean_log_prob_pos).sum() / w.sum().clamp(min=1e-8)
