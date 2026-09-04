
import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """temperature is fixed, not learned. A learnable version (CLIP's own log_logit_scale
    convention) was tried and caused real instability on a full 120-epoch Stage 1 run: it shrinks
    every epoch with nothing bounding how fast, sharpening the softmax faster than ctx/TAB/VAB
    can keep up with it -- for pairs already well-separated this trivially lowers their loss (a
    free win the optimizer always takes regardless of real progress), while the many pairs not
    yet well-separated get punished increasingly harshly, and average loss climbs even though
    nothing is actually getting worse in a normal sense. Verified directly from that run's own
    log (avg loss climbing from epoch ~22 onward while temperature kept dropping, no accuracy
    signal available to show whether real separation was improving underneath). Matches
    CosineAlignLoss's own fixed-temperature convention (pcr/loss/clip_cosine_align_loss.py) and
    CLIP-ReID's original Stage 1, which also never learned this."""

    def __init__(self, temperature=0.1, weight_floor=1e-3):
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
