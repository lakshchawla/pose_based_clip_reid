from __future__ import absolute_import

from .contrastive import PartViewContrastiveLoss
from .crossentropy import CrossEntropyLabelSmooth
from .part_triplet_loss import PartTripletLoss
from .gilt_loss import PartGiLtLoss
from .body_part_attention_loss import BodyPartAttentionLoss

__all__ = [
    'PartViewContrastiveLoss',
    'CrossEntropyLabelSmooth',
    'PartTripletLoss',
    'PartGiLtLoss',
    'BodyPartAttentionLoss',
]
