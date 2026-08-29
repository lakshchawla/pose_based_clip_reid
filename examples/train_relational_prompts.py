"""Stage 1: per-part CLIP prompt learning, with relational mixing across a person's K part
tokens on both sides -- TextualAttentionBlock (owned by PromptLearner) on the text side,
VisualAttentionBlock on the image side. Implements "Algorithm 1 -- Stage 1: Prompt + Relation
Learning" exactly (see progress.md's entry on this file for the full step-by-step mapping);
correspondence to that algorithm's own names:

  Algorithm 1 name          This file / pcr/models
  -----------------          -----------------------
  backbone + BPAM            BPBReIDEncoder (frozen, loaded from Stage 0's checkpoint)
  CLIP image encoder         ClipImageEncoder (frozen, loaded -- NOT consumed by this script's
                              forward pass; BPBreID's backbone remains the sole visual encoder
                              actually producing embeddings used in any loss here. Loaded/frozen
                              only because Algorithm 1's own initialization step says to.)
  CLIP text encoder          ClipTextEncoder (frozen)
  ctx_params                 PromptLearner.part_ctx (K parts only -- PromptLearner also owns a
                              separate fg_ctx for a foreground branch that Algorithm 1 does not
                              use at all; this script trains part_ctx only, see below)
  VRB                        VisualAttentionBlock (pcr/models/relation_blocks.py)
  TRB                        TextualAttentionBlock (owned by PromptLearner, same file)
  InfoNCE                    SupConLoss (pcr/loss/clip_supcon_loss.py) -- CLIP-ReID's actual,
                              multi-positive contrastive loss, restored 2026-08-28 after a
                              round-trip through a literal single-positive `InfoNCELoss` (removed)
                              that matched Algorithm 1 step 15's own wording more closely, but
                              fit this file's PK-sampled batches worse: InfoNCE needed a
                              deduplication patch to avoid treating a person's own other photos in
                              the batch as false negatives, and that patch meant only one
                              representative photo per identity ever shaped the text-side
                              gradient, and the comparison shrank to only the unique identities in
                              one batch (~8). SupCon needs no such patch -- its positive mask
                              already recognizes same-identity rows as positives directly, so
                              every photo in the batch stays a comparison point and contributes to
                              its identity's gradient, not just one representative. See that
                              file's own docstring for the full mechanism and why this is the
                              better fit specifically because this file uses PK sampling on
                              purpose. Temperature restored to `1.0` (SupCon's own original value)
                              alongside the loss swap -- `0.07` was tuned for InfoNCE's
                              single-positive shape, not this one.

                              Widened further, 2026-08-28: the comparison pool on both sides of
                              SupCon is no longer restricted to the current PK batch. i2t compares
                              each image against *every* training identity's text anchor (not just
                              the ~8 in the current batch) via build_text_snapshot's once-per-epoch
                              full-identity table, spliced with this batch's own fresh/
                              differentiable rows; t2i compares each text prompt against *every*
                              cached image in the training set (cached_features never goes stale --
                              the backbone/BPAM are frozen for all of Stage 1). This is what
                              matches CLIP-ReID's own original design (full-identity-table
                              classification), which a single PK batch alone cannot reach no
                              matter how it's sampled -- see IMPROVEMENT_PLAN.md section 4 and
                              progress.md's entry on this change.

Deliberately NOT in this script, matching Algorithm 1 exactly (steps 8, 15-16 extract BPAM's
global/foreground feature f_g but never use it again, and the loss sum is over K parts only, no
foreground term): no foreground contrastive loss, and PromptLearner.fg_ctx is excluded from this
script's optimizer entirely -- it stays at its random initialization after Stage 1 runs. This
leaves it meaningless for Stage 2's own foreground alignment term downstream; that inconsistency
is a known, flagged consequence of scoping this fix to Stage 1 only, not something this file
silently works around.

Batches are genuinely PK-sampled from the cached feature set (build_pk_batches, below) --
Algorithm 1 step 6, and now the very thing SupConLoss's own multi-positive mechanism is built to
use: every identity's several images in a batch become several positives contributing to the same
gradient, not merely "extra useful context" the way it was framed under InfoNCE.

The whole training set's part-embeddings are cached once under no_grad before the training loop
starts (BPBreID/BPAM are frozen throughout, so nothing about them changes step to step) -- PK
batches are then drawn from indices into that cache, not by re-running the encoder. Unlike an
earlier version of this file, there is no upstream visibility filter before this cache is built:
every image in the dataset is cached and trained on, no exceptions (see below for why).

No hard per-branch visibility gating anywhere -- every branch contributes for every cached
sample, unconditionally. Reliability is instead handled by *weighting*, not exclusion:
build_encoder (below) switches this stage's own encoder to continuous (not binary) visibility
scores, and each part's own SupConLoss call is weighted by that part's own per-image visibility
(cached alongside the features, in cached_visibility) -- a poorly-visible part contributes
proportionally less to its own loss term instead of the whole image being rejected outright. This
replaces the upstream image-level filter this file used to run (pcr/utils/visibility_filter.py,
deleted -- see progress.md's entry on this change): that filter discarded 61% of Market1501's
training images in practice, was the wrong granularity (an image with 4 good parts and 1 occluded
one lost all 4), and was found to be driven by an undertrained BPAM signal rather than genuine
occlusion. VisualAttentionBlock is also now visibility-aware at the attention level itself, not
just in the loss that consumes its output (see pcr/models/relation_blocks.py's own docstring) --
each image's own per-part visibility (b_vis, already cached) is passed into vab() as a soft
attention-score bias, so a poorly-visible part's near-garbage feature contributes less as a KEY to
every other part's post-attention representation, closing the contamination gap loss-level
weighting alone could never reach.

TextualAttentionBlock has no per-image signal to use the same way -- PromptLearner.part_ctx is
indexed only by identity label (build_part_prompts(labels, part_visibility)); it has no per-image
input at all. compute_identity_visibility (below) substitutes each identity's *mean* per-part
visibility across every cached image of that identity, computed once here and passed into both
this script's training loop and (via the saved identity_visibility.pth) examples/
cache_text_anchors.py's own final prompt-building pass, so TAB's output stays a consistent,
deterministic function of identity alone in both places. The gradient that reaches part_ctx/TAB
through SupConLoss's own per-image weighting was already an implicitly visibility-weighted
aggregate over whichever instances of that identity are in the current PK batch; this adds a
second, complementary mechanism at the attention level itself, mirroring VAB's.

Produces three files consumed by examples/cache_text_anchors.py (not by Stage 2 directly -- see
that script's own docstring for why the final text-prototype computation is a separate step):
prompt_learner.pth (PromptLearner's state, including TextualAttentionBlock and the untrained
fg_ctx), vab.pth (VisualAttentionBlock's state -- unlike the text-side modules, VAB is *not*
discarded after Stage 1; Stage 2 loads vab.pth to continue training the same VisualAttentionBlock
instance), and identity_visibility.pth (the per-identity mean visibility table described above).

Config-driven (YAML), not argparse -- see configs/stage1_relational_prompts.yaml. This is a
deliberate deviation from the rest of pcr2 (train_uda.py/train_usl.py stay argparse-only, decided
directly with the user -- see progress.md).
"""
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pcr import datasets
from pcr.models.bpbreid_encoder import BPBReIDEncoder, BPBReIDModelCfg
from pcr.models.clip_image_encoder import ClipImageEncoder
from pcr.models.clip_text_encoder import ClipTextEncoder
from pcr.models.prompt_learner import PromptLearner
from pcr.models.relation_blocks import VisualAttentionBlock
from pcr.loss.clip_supcon_loss import SupConLoss
from pcr.utils.config import load_yaml_config
from pcr.utils.data import transforms as T
from pcr.utils.data.preprocessor import Preprocessor
from pcr.utils.logging import Logger
from pcr.utils.lr_scheduler import WarmupCosineLR
from pcr.utils.osutils import mkdir_if_missing


def get_data(name, data_dir):
    return datasets.create(name, osp.join(data_dir, name))


def get_cache_loader(dataset_list, root, height, width, batch_size, workers):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer,
    ])
    return DataLoader(
        Preprocessor(dataset_list, root=root, transform=transformer),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)


def cache_part_features(encoder, data_loader):
    """Single full-dataset forward pass under no_grad, caching every image's part embeddings,
    visibility, and real identity label -- mirrors CLIP-ReID's own stage-1 full-dataset feature
    cache, generalized to BPBreID's [M, D] per-branch embeddings."""
    encoder.eval()
    features, visibilities, labels = [], [], []
    with torch.no_grad():
        for imgs, _, pids, _, _ in data_loader:
            f_out, vis = encoder(imgs.cuda())
            features.append(f_out.cpu())
            visibilities.append(vis.cpu())
            labels.append(pids)
    return torch.cat(features, 0), torch.cat(visibilities, 0), torch.cat(labels, 0)


def compute_identity_visibility(cached_visibility, cached_labels, num_identities):
    """cached_visibility: [N, 1+K] (per-image, from cache_part_features). cached_labels: [N].
    Returns [num_identities, 1+K]: each identity's mean visibility across every cached image of
    that identity. TextualAttentionBlock has no per-image signal available (PromptLearner.part_ctx
    is indexed by identity alone -- see relation_blocks.py's own docstring), so this is the
    per-identity substitute passed into it as its attention bias; saved to disk
    (identity_visibility.pth) so examples/cache_text_anchors.py can reuse the exact same values
    when it rebuilds the final frozen prompts, keeping TAB's output a true, consistent function of
    identity alone rather than depending on which script last computed it."""
    num_branches = cached_visibility.size(1)
    sums = torch.zeros(num_identities, num_branches, device=cached_visibility.device)
    sums.index_add_(0, cached_labels, cached_visibility)
    counts = torch.zeros(num_identities, device=cached_visibility.device)
    counts.index_add_(0, cached_labels, torch.ones_like(cached_labels, dtype=cached_visibility.dtype))
    return sums / counts.unsqueeze(1).clamp(min=1)


def build_text_snapshot(prompt_learner, text_encoder, num_identities, num_parts,
                         identity_visibility, id_batch):
    """Full-dataset text-anchor snapshot, [num_identities, num_parts, D], rebuilt once at the
    start of every epoch (not every iteration -- see main_worker's own call site) under no_grad,
    using the model's CURRENT part_ctx/TextualAttentionBlock weights. This is what widens Stage
    1's negative pool from "the ~8 identities in one PK batch" to "every identity in the training
    set", matching CLIP-ReID's own original Stage 1 design (full-identity-table classification,
    not a batch-restricted one) -- see IMPROVEMENT_PLAN.md section 4 and progress.md's entry on
    this change for the full reasoning.

    Only used as a *negative* pool for identities NOT present in the current PK batch (see
    main_worker's own per-iteration splicing) -- an identity that IS in the current batch gets a
    fresh, differentiable re-encoding instead, since gradient must still reach part_ctx/TAB for it.
    A once-per-epoch refresh (rather than once per iteration) keeps this affordable: rebuilding it
    costs one extra CLIP-text forward pass over the whole identity set, not per training step, and
    a whole epoch's worth of iterations (hundreds) is far more than enough for the small
    per-iteration drift in part_ctx/TAB to matter for what is, after all, only a negative-comparison
    pool, not something being directly optimized against.

    TextualAttentionBlock's own attention only mixes tokens *within* one identity's own K*n_ctx-
    token sequence (standard transformer batching never attends across the batch dimension), so
    building this in chunks of `id_batch` identities at a time is exactly equivalent to building
    every identity one at a time -- no cross-identity leakage or batching-order sensitivity."""
    prompt_learner.eval()
    D = text_encoder.embed_dim
    snapshot = torch.zeros(num_identities, num_parts, D, device='cuda')
    with torch.no_grad():
        for start in range(0, num_identities, id_batch):
            ids = torch.arange(start, min(start + id_batch, num_identities), device='cuda')
            part_vis = identity_visibility[ids, 1:]
            prompts = prompt_learner.build_part_prompts(ids, part_vis)
            for k in range(num_parts):
                text_feat = text_encoder(prompts[1 + k], prompt_learner.tokenized_prompts).float()
                # L2-normalized before storing -- see this file's own module docstring (the
                # "SupCon" mapping-table entry) for why: SupConLoss's dot product only behaves as
                # a real cosine similarity, matching its temperature's calibration, if both sides
                # are unit-norm -- part_visual (the image side) already is; this was the one place
                # text wasn't.
                snapshot[ids, k] = F.normalize(text_feat, p=2, dim=-1)
    return snapshot


def build_pk_batches(cached_labels, num_instances, batch_size):
    """Groups the cached feature set's indices by identity, then partitions all identities into
    PK batches for one epoch: batch_size // num_instances identities per batch, num_instances
    cached images per identity (sampled with replacement if that identity has fewer than
    num_instances cached images). Algorithm 1 step 6 ("Sample a PK batch of pre-filtered
    images") -- see this file's own module docstring for why SupConLoss's multi-positive mechanism
    depends on this. A final partial group of identities (fewer than batch_size // num_instances
    left over) is dropped, matching this repo's other PK samplers' drop_last convention
    (pcr/utils/data/sampler.py::RandomIdentitySampler)."""
    labels_np = cached_labels.cpu().numpy()
    id_to_indices = {}
    for idx, pid in enumerate(labels_np):
        id_to_indices.setdefault(int(pid), []).append(idx)
    pids = list(id_to_indices.keys())
    random.shuffle(pids)

    num_pids_per_batch = max(1, batch_size // num_instances)
    batches = []
    for start in range(0, len(pids), num_pids_per_batch):
        batch_pids = pids[start:start + num_pids_per_batch]
        if len(batch_pids) < num_pids_per_batch:
            break
        batch_idx = []
        for pid in batch_pids:
            pool = id_to_indices[pid]
            replace = len(pool) < num_instances
            chosen = np.random.choice(pool, size=num_instances, replace=replace)
            batch_idx.extend(int(i) for i in chosen)
        batches.append(torch.tensor(batch_idx, dtype=torch.long, device=cached_labels.device))
    return batches


def build_encoder(cfg):
    model_cfg = BPBReIDModelCfg(backbone=cfg.model.backbone)
    model_cfg.masks.parts_num = cfg.model.parts_num
    model_cfg.dim_reduce_output = cfg.model.dim_reduce_output
    # Continuous visibility scores, not the dataclass's own binary default -- needed for
    # SupConLoss's per-part weighting to be meaningfully graduated rather than near-binary.
    # Overridden only here (this stage's own encoder construction), not in BPBReIDModelCfg's
    # shared default -- Stage 3 stays binary, untouched. Stage 1's encoder calls .eval()
    # immediately below and never leaves eval mode, so testing_binary_visibility_score is the
    # one that's actually reachable here; training_binary_visibility_score is set for symmetry.
    model_cfg.training_binary_visibility_score = False
    model_cfg.testing_binary_visibility_score = False
    encoder = BPBReIDEncoder(model_cfg, checkpoint_path=cfg.model.checkpoint_path or None).cuda()
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def main():
    parser = argparse.ArgumentParser(description="PCR Stage 1: per-part CLIP prompt learning")
    parser.add_argument('--config', type=str, metavar='PATH', default="configs/stage1_relational_prompts.yaml")
    parser.add_argument('--setup-only', action='store_true',
                         help="build dataset/encoder/prompt-learner/cache, print shapes, exit "
                              "before the training loop")
    args = parser.parse_args()
    cfg = load_yaml_config(args.config)
    main_worker(cfg, setup_only=args.setup_only)


def main_worker(cfg, setup_only=False):
    seed = getattr(cfg.logging, 'seed', None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    mkdir_if_missing(cfg.logging.logs_dir)
    sys.stdout = Logger(osp.join(cfg.logging.logs_dir, 'log.txt'))
    print("==========\nConfig:{}\n==========".format(vars(cfg)))
    start_time = time.monotonic()

    dataset = get_data(cfg.data.dataset, cfg.data.data_dir)
    num_identities = dataset.num_train_pids
    num_parts = cfg.model.parts_num
    num_branches = 1 + num_parts

    encoder = build_encoder(cfg)
    text_encoder = ClipTextEncoder(clip_arch=cfg.clip.arch, device='cuda').cuda()
    # Loaded and frozen per Algorithm 1's own initialization step -- not consumed anywhere below;
    # see this file's module docstring for why.
    image_encoder = ClipImageEncoder(clip_arch=cfg.clip.arch, device='cuda').cuda()
    prompt_learner = PromptLearner(num_identities, num_parts, text_encoder, n_ctx=cfg.clip.n_ctx,
                                    tab_num_heads=cfg.tab.num_heads, tab_num_layers=cfg.tab.num_layers,
                                    device='cuda').cuda()
    vab = VisualAttentionBlock(dim=cfg.model.dim_reduce_output, num_heads=cfg.vab.num_heads,
                               num_layers=cfg.vab.num_layers).cuda()

    train_set = sorted(dataset.train)
    print("==> Caching part-embeddings for the full training set (frozen encoder, single pass, "
          "no upstream filtering -- every image enters training, weighted per-part inside the "
          "loss instead)")
    cache_loader = get_cache_loader(train_set, dataset.images_dir, cfg.data.height, cfg.data.width,
                                     cfg.data.cache_batch_size, cfg.data.workers)
    cached_features, cached_visibility, cached_labels = cache_part_features(encoder, cache_loader)
    cached_features = cached_features.cuda()
    cached_visibility = cached_visibility.cuda()
    cached_labels = cached_labels.cuda()
    num_images = cached_labels.size(0)
    print("==> Cached {} images across {} identities, {} branches".format(
        num_images, num_identities, num_branches))

    if setup_only:
        print('==> Setup complete: {} branches, {} cached images, part_ctx shape {} (trainable), '
              'fg_ctx shape {} (untrained -- see module docstring). Exiting before the training '
              'loop (--setup-only).'.format(
                  num_branches, num_images, tuple(prompt_learner.part_ctx.shape),
                  tuple(prompt_learner.fg_ctx.shape)))
        return

    # Per-identity mean visibility -- TextualAttentionBlock's attention bias (see
    # compute_identity_visibility's own docstring and relation_blocks.py for why this differs from
    # VisualAttentionBlock's per-image one).
    identity_visibility = compute_identity_visibility(cached_visibility, cached_labels, num_identities)

    supcon = SupConLoss(temperature=cfg.loss.temperature).cuda()
    # fg_ctx deliberately excluded -- Algorithm 1 has no foreground term (see module docstring);
    # only part_ctx, TextualAttentionBlock (prompt_learner.tab), and VisualAttentionBlock train.
    trainable_params = ([prompt_learner.part_ctx] + list(prompt_learner.tab.parameters())
                         + list(vab.parameters()))
    optimizer = torch.optim.Adam(trainable_params, lr=cfg.optim.lr,
                                  weight_decay=cfg.optim.weight_decay)
    scheduler = WarmupCosineLR(optimizer, max_epochs=cfg.optim.epochs,
                                warmup_epochs=cfg.optim.warmup_epochs,
                                warmup_lr_init=cfg.optim.warmup_lr_init,
                                lr_min=cfg.optim.lr_min)
    # GradScaler, not raw fp16 backward -- the CLIP text tower runs in fp16 (matches CLIP-ReID's
    # own dtype exactly, see pcr/models/clip_text_encoder.py's docstring), and CLIP-ReID's own
    # stage-1 loop always wraps its backward in a GradScaler to guard against fp16 gradient
    # underflow through the text transformer -- ported faithfully rather than assuming raw fp16
    # backward is fine. VisualAttentionBlock and PromptLearner's own parameters run in fp32
    # (GradScaler is harmless for fp32 leaves), so one scaler covers everything trainable.
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(cfg.optim.epochs):
        # Refresh the full-identity text-anchor snapshot once per epoch, against this epoch's
        # part_ctx/TAB weights -- see build_text_snapshot's own docstring. Puts prompt_learner in
        # eval() mode briefly; explicitly switched back to train() below before any real training
        # step runs.
        text_snapshot = build_text_snapshot(prompt_learner, text_encoder, num_identities, num_parts,
                                             identity_visibility, cfg.data.cache_batch_size)
        prompt_learner.train()
        vab.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        # Algorithm 1 step 6: a fresh PK partition of the cached feature set every epoch, not a
        # plain random sub-batch -- see build_pk_batches' and this file's own module docstring.
        batches = build_pk_batches(cached_labels, cfg.data.num_instances, cfg.data.batch_size)
        iters_per_epoch = len(batches)

        for it, b_idx in enumerate(batches):
            b_labels = cached_labels[b_idx]
            b_features = cached_features[b_idx]  # [b, 1+K, D], already L2-normalized per branch
            b_vis = cached_visibility[b_idx]     # [b, 1+K]

            optimizer.zero_grad()

            # Algorithm 1 steps 10-14: only the K part prompts are built and pushed through the
            # frozen CLIP text encoder -- no foreground prompt/loss (module docstring).
            id_vis = identity_visibility[b_labels]  # [b, 1+K], TAB's per-identity bias
            prompts = prompt_learner.build_part_prompts(b_labels, id_vis[:, 1:])  # list of 1+K tensors
            part_visual = vab(b_features[:, 1:, :], b_vis[:, 1:])  # [b, K, D], relationally mixed

            # Identities NOT in this batch -- their text row comes from this epoch's (detached)
            # snapshot instead of a fresh re-encoding, widening i2t's negative pool to the full
            # training set. See build_text_snapshot's own docstring / IMPROVEMENT_PLAN.md section 4.
            in_batch = torch.zeros(num_identities, dtype=torch.bool, device=b_labels.device)
            in_batch[b_labels] = True
            other_ids = in_batch.logical_not().nonzero(as_tuple=True)[0]

            # Algorithm 1 step 15 (loss_i2t + loss_t2i), via SupConLoss's own two-call convention
            # (see that file's docstring).
            loss = b_features.new_zeros(())
            for k in range(num_parts):
                # L2-normalized -- see build_text_snapshot's own comment on why: visual_k (below)
                # is already unit-norm, and SupConLoss's dot product only behaves as a real cosine
                # similarity, matching its own temperature, if both sides are.
                part_text = F.normalize(
                    text_encoder(prompts[1 + k], prompt_learner.tokenized_prompts).float(), p=2, dim=-1)
                visual_k = part_visual[:, k, :]
                w_k = b_vis[:, 1 + k]  # same 1+k branch offset as prompts[1+k]/part_visual[:,k,:]

                # i2t: full num_identities-way classification, not just this batch's ~8 -- this
                # batch's own identities keep their fresh, differentiable text row (gradient must
                # reach part_ctx/TAB for them); every other identity is a detached negative from
                # this epoch's text_snapshot.
                other_text = torch.cat([part_text, text_snapshot[other_ids, k, :]], dim=0)
                other_text_labels = torch.cat([b_labels, other_ids], dim=0)
                loss = loss + supcon(visual_k, other_text, b_labels, other_text_labels, w_k)

                # t2i: full-dataset classification -- every cached image, not just this batch's,
                # is a comparison point. No snapshot/splicing needed here at all: cached_features
                # never goes stale, since the backbone/BPAM are frozen for the whole of Stage 1.
                loss = loss + supcon(part_text, cached_features[:, 1 + k, :], b_labels, cached_labels, w_k)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

            if (it + 1) % cfg.logging.print_freq == 0:
                print('Epoch: [{}][{}/{}]\tLoss {:.3f}\tLR {:.2e}\tVAB gate {:.3f}'.format(
                    epoch, it + 1, iters_per_epoch, loss.item(), optimizer.param_groups[0]['lr'],
                    torch.tanh(vab.gate).item()))

        scheduler.step()
        print('Epoch {} done in {:.1f}s, avg loss {:.4f}'.format(
            epoch, time.time() - epoch_start, epoch_loss / iters_per_epoch))

    torch.save(prompt_learner.state_dict(), osp.join(cfg.logging.logs_dir, 'prompt_learner.pth'))
    torch.save(vab.state_dict(), osp.join(cfg.logging.logs_dir, 'vab.pth'))
    torch.save(identity_visibility.cpu(), osp.join(cfg.logging.logs_dir, 'identity_visibility.pth'))
    print('==> Saved prompt_learner.pth, vab.pth and identity_visibility.pth to {}. Run '
          'examples/cache_text_anchors.py next to build text_prototypes.pth for Stage 2.'.format(
              cfg.logging.logs_dir))

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    main()
