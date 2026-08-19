"""Adapter wrapping bpbreid's BPBreID model as a drop-in PCR encoder.

Requires torchreid (bpbreid's own package, not deep-person-reid's) importable, i.e.
`pip install -e /path/to/bpbreid` into this repo's environment -- see README.md.
"""
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchreid.models.bpbreid import BPBreID
from torchreid.utils import load_pretrained_weights
from torchreid.utils.constants import BN_FOREGROUND, BN_PARTS, FOREGROUND, PARTS


@dataclass
class BPBReIDModelCfg:
    """Plain-Python stand-in for bpbreid's yacs model_cfg -- only the ~13 attributes
    BPBreID.__init__/forward actually read. Not a training-hyperparameter config; it just
    describes the frozen architecture matching whatever checkpoint gets loaded.

    parts_num=5 is correct for the market1501/dukemtmc bpbreid configs: masks.preprocess=
    'five_v' in those yamls auto-resolves parts_num to 5 via
    torchreid/data/masks_transforms/__init__.py::compute_parts_num_and_names, even though the
    yaml's own explicit masks.parts_num default is 1.
    """

    class Masks:
        parts_num: int = 5

    masks: Masks = field(default_factory=Masks)
    shared_parts_id_classifier: bool = False
    test_use_target_segmentation: str = 'none'
    training_binary_visibility_score: bool = True
    testing_binary_visibility_score: bool = True
    backbone: str = 'hrnet32'
    last_stride: int = 1
    dim_reduce: str = 'after_pooling'
    dim_reduce_output: int = 512
    hrnet_pretrained_path: str = ''
    learnable_attention_enabled: bool = True
    normalization: str = 'identity'
    pooling: str = 'gwap'


class BPBReIDEncoder(nn.Module):
    """learnable_attention_enabled=True means the pixel-to-part classifier is learned
    end-to-end from the ID/triplet/contrastive gradient -- no external masks needed for
    forward, so source and target domain images are on identical footing here.

    num_classes=1 is a placeholder: PCR never uses BPBreID's own ID classifier heads, only
    the returned embeddings + visibility scores.
    """

    def __init__(self, model_cfg, checkpoint_path=None):
        super(BPBReIDEncoder, self).__init__()
        self.model_cfg = model_cfg
        self.model = BPBreID(num_classes=1, pretrained=False, loss='part_based', model_cfg=model_cfg)
        if checkpoint_path:
            load_pretrained_weights(self.model, checkpoint_path)
        self.num_parts = 1 + model_cfg.masks.parts_num
        self.num_features = model_cfg.dim_reduce_output

    def forward(self, images):
        embeddings, visibility_scores, _, _, _, _ = self.model(images)

        foreground_emb = embeddings[BN_FOREGROUND]  # [B, D]
        parts_emb = embeddings[BN_PARTS]  # [B, K, D]
        foreground_vis = visibility_scores[FOREGROUND]  # [B]
        parts_vis = visibility_scores[PARTS]  # [B, K]

        f_out = torch.cat([foreground_emb.unsqueeze(1), parts_emb], dim=1)  # [B, 1+K, D]
        f_out = F.normalize(f_out, p=2, dim=-1)  # each of the M branches normalized independently

        vis = torch.cat([foreground_vis.unsqueeze(1), parts_vis], dim=1)  # [B, 1+K]

        return f_out, vis

    def forward_full(self, images):
        """Like forward(), but also returns pixels_cls_scores [B, K, Hf, Wf] (needed by
        BodyPartAttentionLoss). A separate method rather than an always-3-tuple forward() so
        every existing caller (trainers, evaluators, memory init) keeps working unchanged --
        the tradeoff is that this method isn't nn.DataParallel-aware: call it on the unwrapped
        module (`model.module.forward_full(...)` if `model` is DataParallel-wrapped), which
        runs on a single device. Fine for this repo's single-GPU dev setup; would need
        revisiting (e.g. always returning the 3-tuple from forward() instead) to actually
        scale BPA loss across multiple GPUs.
        """
        embeddings, visibility_scores, _, pixels_cls_scores, _, _ = self.model(images)

        foreground_emb = embeddings[BN_FOREGROUND]
        parts_emb = embeddings[BN_PARTS]
        foreground_vis = visibility_scores[FOREGROUND]
        parts_vis = visibility_scores[PARTS]

        f_out = torch.cat([foreground_emb.unsqueeze(1), parts_emb], dim=1)
        f_out = F.normalize(f_out, p=2, dim=-1)
        vis = torch.cat([foreground_vis.unsqueeze(1), parts_vis], dim=1)

        return f_out, vis, pixels_cls_scores


