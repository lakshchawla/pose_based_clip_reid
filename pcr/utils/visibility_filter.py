"""Upstream candidate-acceptance gate (reid_pipeline_plan.md's part-relational-attention addendum,
section 0.1): rejects images whose body parts aren't reliably visible enough for
VisualAttentionBlock/TextualAttentionBlock's no-masking assumption to hold, *before* they ever reach
Stage 1 or Stage 2's training loop -- an explicit, logged, configurable filtering step (the
threshold is a real argument here, not an unstated assumption baked into a dataset file somewhere
no one remembers).

"Visibility index" for one image is defined here as the mean of BPBreID's own per-branch
visibility scores across the K part branches only (branches 1..M-1 in BPBreIDEncoder's
convention -- foreground/branch 0 is excluded, matching relation_blocks.py's own K-parts-only
scope). This is one reasonable reading of the plan's own under-specified "visibility-index
threshold check" -- documented here as a deliberate choice, not asserted as the only valid one;
change `visibility_index` below if a different aggregation is wanted later.
"""
import torch
from torch.utils.data import DataLoader

from .data import transforms as T
from .data.preprocessor import Preprocessor


def visibility_index(vis):
    """vis: [N, M] per-branch visibility (bool or float in [0,1]), branch 0 = foreground.
    Returns [N]: mean visibility across the K part branches (1..M-1), the scalar this file
    filters on."""
    parts_vis = vis[:, 1:].float()
    return parts_vis.mean(dim=1)


def filter_by_visibility(dataset_list, encoder, height, width, threshold, root=None,
                          batch_size=64, workers=4):
    """dataset_list: list of (fname, pid, camid). encoder: frozen BPBreIDEncoder (or DataParallel-
    wrapped), assumed already .eval() and on the right device. root: passed straight through to
    Preprocessor (None when dataset_list's fnames are already absolute paths, matching how every
    other script in this repo calls Preprocessor). threshold: lambda_v_min, an image is kept iff
    its visibility_index() >= threshold. Returns (kept_list, num_rejected) -- always prints the
    kept/rejected counts, so a run's logs always show whether/how much filtering happened rather
    than leaving it implicit."""
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer,
    ])
    loader = DataLoader(
        Preprocessor(dataset_list, root=root, transform=transformer),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)

    kept = []
    with torch.no_grad():
        offset = 0
        for imgs, _, _, _, _ in loader:
            _, vis = encoder(imgs.cuda())
            scores = visibility_index(vis).cpu()
            for i, score in enumerate(scores):
                if score.item() >= threshold:
                    kept.append(dataset_list[offset + i])
            offset += imgs.size(0)

    num_rejected = len(dataset_list) - len(kept)
    print('==> Visibility filter (threshold={:.2f}): kept {}/{} images, rejected {}'.format(
        threshold, len(kept), len(dataset_list), num_rejected))
    return kept, num_rejected
