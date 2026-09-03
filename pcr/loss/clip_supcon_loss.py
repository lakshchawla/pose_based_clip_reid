
import math

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    MAX_LOGIT_SCALE = 100.0  # temperature >= 0.01 -- same clamp value as CLIP's own training code

    def __init__(self, temperature=0.1, weight_floor=1e-3):
        super(SupConLoss, self).__init__()
        self.log_logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))
        self.weight_floor = weight_floor

    @property
    def temperature(self):
        return 1.0 / self.log_logit_scale.exp().clamp(max=self.MAX_LOGIT_SCALE)

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
