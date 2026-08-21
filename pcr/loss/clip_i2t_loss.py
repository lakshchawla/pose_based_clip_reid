"""Ported from CLIP-ReID's Stage-2 alignment term (../CLIP-ReID/loss/make_loss.py, the
`I2TLOSS = xent(i2tscore, target)` branch). Despite the name "alignment loss," this is NOT a
cosine-similarity regression term -- it's label-smoothed cross-entropy where the "logits" are the
image embedding dotted against a frozen, precomputed-once, per-identity text-prototype table
(image_features @ text_prototypes.T), and the target is the real identity label. The frozen
prototype table acts as a fixed linear classifier built from Stage 1's learned prompts.

Reuses pcr/loss/crossentropy.py::CrossEntropyLabelSmooth exactly as CLIP-ReID's own make_loss.py
does (`xent = CrossEntropyLabelSmooth(num_classes=num_classes)`), rather than reimplementing
label smoothing here.
"""
import torch.nn as nn

from .crossentropy import CrossEntropyLabelSmooth


class I2TLoss(nn.Module):
    def __init__(self, num_identities, epsilon=0.1):
        super(I2TLoss, self).__init__()
        self.xent = CrossEntropyLabelSmooth(num_identities, epsilon=epsilon)

    def forward(self, image_features, text_prototypes, targets):
        """image_features: [B, D]. text_prototypes: [num_identities, D], frozen (no grad
        expected to flow into it -- callers should pass a detached/no_grad table). targets: [B]
        real identity labels indexing into text_prototypes' first dimension."""
        logits = image_features @ text_prototypes.t()
        return self.xent(logits, targets)
