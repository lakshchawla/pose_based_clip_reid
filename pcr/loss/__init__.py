from __future__ import absolute_import

from .contrastive import PartViewContrastiveLoss
from .crossentropy import CrossEntropyLabelSmooth
from .part_triplet_loss import PartTripletLoss
from .gilt_loss import PartGiLtLoss
from .body_part_attention_loss import BodyPartAttentionLoss
from .clip_supcon_loss import SupConLoss
from .clip_i2t_loss import I2TLoss
from .clip_contrastive_loss import ClipContrastiveLoss

__all__ = [
    'PartViewContrastiveLoss',
    'CrossEntropyLabelSmooth',
    'PartTripletLoss',
    'PartGiLtLoss',
    'BodyPartAttentionLoss',
    'SupConLoss',
    'I2TLoss',
    'ClipContrastiveLoss',
]
