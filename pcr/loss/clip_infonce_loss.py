"""Literal InfoNCE, matching "Algorithm 1 -- Stage 1: Prompt + Relation Learning" step 15's own
wording (`InfoNCE(relation_feats[:,k], t_i^k)`) -- replaces `clip_supcon_loss.py::SupConLoss`
(CLIP-ReID's real, multi-positive contrastive loss; removed, this was its only caller) as Stage
1's loss, per direct user request.

The catch this file exists to avoid: plain diagonal InfoNCE over a raw batch treats every other
row as a negative regardless of identity. Stage 1's batches are PK-sampled (multiple images per
identity, for SupConLoss's own multi-positive benefit before this change) -- naive diagonal
InfoNCE would therefore treat two images of the *same* identity as false negatives of each other,
which is exactly the failure mode already observed once in this repo: `train_relational_clip.py`
used this exact naive form and was removed for underperforming (see progress.md's entries on that
file). Fixed here by deduplicating to unique identities before building the negative set on
*both* directions, so a PK batch's duplicate rows never contaminate each other:

  - i2t: every image in the batch is its own row/anchor (no dedup needed there -- more images is
    more signal, not a problem), classified against the C *unique* identities' text anchors in
    this batch. Two images of the same identity simply share the same correct target class --
    normal, correct behavior for any classification-style loss, not a collision.
  - t2i: to keep this direction single-positive and symmetric with i2t rather than introducing a
    second, different kind of multi-positive problem (one text anchor, several correct images),
    both sides are deduplicated to one representative row per identity here, giving a clean C x C
    match with no ambiguity.

Temperature is a fixed constant (not learned, unlike e.g. the now-removed
`clip_contrastive_loss.py::ClipContrastiveLoss`'s logit_scale), defaulting to 0.07 -- CLIP's own
well-established optimal starting temperature (`logit_scale = log(1/0.07)` in the original paper
and in `third_party/clip/model.py`), reused directly here per "keep an optimal temperature const"
rather than SupConLoss's own temperature=1.0 default (a value tuned for the multi-positive
formulation this loss replaces, not for a single-positive one).

Per-sample `weights` (required, not optional -- this class has exactly one caller,
examples/train_relational_prompts.py, which always has a per-part visibility score available
once Stage 1's own encoder construction switches to continuous visibility scores; see
progress.md's entry on the visibility-filter-to-weighting refactor for why): replaces the upstream
image-level visibility filter this repo used to run before Stage 1 ever saw an image. Every image
now enters training; a part with low visibility for a given sample contributes proportionally less
to that part's own InfoNCE terms instead of the whole image being rejected outright. Weights are
floored (`weight_floor`, default 1e-3) so no sample is ever exactly zero-weighted -- a pathological
score still gets a negligible-but-nonzero gradient, consistent with "no image is excluded, only
down-weighted."

`first_idx` (the representative row picked per unique identity, for both the negative-dedup role
described above and the anchor role below) is chosen as the *highest-weighted* occurrence of that
identity in the batch, not simply the first one encountered. Rationale: `text_anchors`/
`visual_anchors` for the t2i direction are built entirely from this one row, so t2i's quality
rides on that row being trustworthy -- picking the best-visibility instance among the batch's
`num_instances` copies of each identity (already available for free via PK sampling) removes a
real source of previously-arbitrary t2i noise, where a poorly-visible image could win "first
occurrence" purely by sampling order.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07, weight_floor=1e-3):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.weight_floor = weight_floor

    def forward(self, visual_features, text_features, labels, weights):
        """visual_features, text_features: [B, D], one row per sample -- duplicate rows expected
        whenever labels repeats (PK-sampled batches). labels: [B]. weights: [B] float, per-image
        visibility score for the specific branch this call is scoring (continuous in [0,1] once
        the caller's encoder uses continuous visibility scores). Returns loss_i2t + loss_t2i."""
        unique_labels, inverse = torch.unique(labels, sorted=True, return_inverse=True)
        num_unique = unique_labels.size(0)
        w = weights.clamp(min=self.weight_floor)

        # Representative row per unique identity: the highest-weighted (most visible) occurrence
        # in this batch, not merely the first one -- see module docstring.
        best_weight = torch.full((num_unique,), -1.0, device=labels.device, dtype=w.dtype)
        first_idx = torch.zeros(num_unique, dtype=torch.long, device=labels.device)
        for i in range(labels.size(0)):
            c = inverse[i]
            if w[i] > best_weight[c]:
                best_weight[c] = w[i]
                first_idx[c] = i

        text_anchors = text_features[first_idx]      # [C, D]
        visual_anchors = visual_features[first_idx]  # [C, D]

        # i2t: every image classified against the C unique text anchors -- duplicate rows (same
        # identity) simply share the same correct target, not a collision. Weighted by each row's
        # own visibility for this branch.
        logits_i2t = visual_features @ text_anchors.t() / self.temperature  # [B, C]
        per_sample_i2t = F.cross_entropy(logits_i2t, inverse, reduction='none')  # [B]
        loss_i2t = (w * per_sample_i2t).sum() / w.sum().clamp(min=1e-8)

        # t2i: one representative image per identity, so this direction is single-positive too.
        # Weighted by the representative row's own visibility.
        logits_t2i = text_anchors @ visual_anchors.t() / self.temperature  # [C, C]
        targets_t2i = torch.arange(num_unique, device=labels.device)
        per_identity_t2i = F.cross_entropy(logits_t2i, targets_t2i, reduction='none')  # [C]
        w_t2i = w[first_idx]
        loss_t2i = (w_t2i * per_identity_t2i).sum() / w_t2i.sum().clamp(min=1e-8)

        return loss_i2t + loss_t2i
