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
    this batch (one representative text embedding per identity, its first occurrence in the
    batch). Two images of the same identity simply share the same correct target class -- normal,
    correct behavior for any classification-style loss, not a collision.
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
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, visual_features, text_features, labels):
        """visual_features, text_features: [B, D], one row per sample -- duplicate rows expected
        whenever labels repeats (PK-sampled batches). labels: [B]. Returns loss_i2t + loss_t2i."""
        unique_labels, inverse = torch.unique(labels, sorted=True, return_inverse=True)
        num_unique = unique_labels.size(0)

        # First occurrence of each unique identity in this batch -- the representative row used
        # to dedupe negatives on both directions (see module docstring).
        first_idx = torch.zeros(num_unique, dtype=torch.long, device=labels.device)
        seen = torch.zeros(num_unique, dtype=torch.bool, device=labels.device)
        for i in range(labels.size(0)):
            c = inverse[i]
            if not seen[c]:
                seen[c] = True
                first_idx[c] = i

        text_anchors = text_features[first_idx]      # [C, D]
        visual_anchors = visual_features[first_idx]  # [C, D]

        # i2t: every image classified against the C unique text anchors -- duplicate rows (same
        # identity) simply share the same correct target, not a collision.
        logits_i2t = visual_features @ text_anchors.t() / self.temperature  # [B, C]
        loss_i2t = F.cross_entropy(logits_i2t, inverse)

        # t2i: one representative image per identity, so this direction is single-positive too.
        logits_t2i = text_anchors @ visual_anchors.t() / self.temperature  # [C, C]
        targets_t2i = torch.arange(num_unique, device=labels.device)
        loss_t2i = F.cross_entropy(logits_t2i, targets_t2i)

        return loss_i2t + loss_t2i
