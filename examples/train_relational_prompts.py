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
  InfoNCE                    InfoNCELoss (pcr/loss/clip_infonce_loss.py) -- literal single-
                              positive InfoNCE, matching the algorithm's own step 15 wording,
                              replacing the earlier `SupConLoss` (CLIP-ReID's multi-positive
                              contrastive loss, removed -- this was its only caller). Per direct
                              user request. Naive diagonal InfoNCE over a raw PK-sampled batch
                              would reproduce the exact collision that made the earlier
                              train_relational_clip.py underperform (same-identity images treated
                              as false negatives) -- InfoNCELoss avoids this by deduplicating to
                              unique identities before building the negative set on both
                              directions; see that file's own docstring for the full mechanism.
                              Temperature is a fixed constant (0.07 by default, CLIP's own
                              established optimal starting value), not learned.

Deliberately NOT in this script, matching Algorithm 1 exactly (steps 8, 15-16 extract BPAM's
global/foreground feature f_g but never use it again, and the loss sum is over K parts only, no
foreground term): no foreground contrastive loss, and PromptLearner.fg_ctx is excluded from this
script's optimizer entirely -- it stays at its random initialization after Stage 1 runs. This
leaves it meaningless for Stage 2's own foreground alignment term downstream; that inconsistency
is a known, flagged consequence of scoping this fix to Stage 1 only, not something this file
silently works around.

Batches are genuinely PK-sampled from the cached feature set (build_pk_batches, below) --
Algorithm 1 step 6, and not just a label change from the previous plain-random-sub-batch version:
even with InfoNCELoss's per-identity deduplication making it safe against PK-batch collisions,
PK sampling still means every anchor is far more likely to share its batch with useful additional
signal (other views of related identities feeding the same TAB/VAB mixing step) than a plain
random batch would give -- and matches Algorithm 1's own step 6 regardless of loss-type specifics.

The whole training set's part-embeddings are cached once under no_grad before the training loop
starts (BPBreID/BPAM are frozen throughout, so nothing about them changes step to step) -- PK
batches are then drawn from indices into that cache, not by re-running the encoder. Unlike an
earlier version of this file, there is no upstream visibility filter before this cache is built:
every image in the dataset is cached and trained on, no exceptions (see below for why).

No hard per-branch visibility gating anywhere -- every branch contributes for every cached
sample, unconditionally. Reliability is instead handled by *weighting*, not exclusion:
build_encoder (below) switches this stage's own encoder to continuous (not binary) visibility
scores, and each part's own InfoNCELoss call is weighted by that part's own per-image visibility
(cached alongside the features, in cached_visibility) -- a poorly-visible part contributes
proportionally less to its own loss term instead of the whole image being rejected outright. This
replaces the upstream image-level filter this file used to run (pcr/utils/visibility_filter.py,
deleted -- see progress.md's entry on this change): that filter discarded 61% of Market1501's
training images in practice, was the wrong granularity (an image with 4 good parts and 1 occluded
one lost all 4), and was found to be driven by an undertrained BPAM signal rather than genuine
occlusion. VisualAttentionBlock/TextualAttentionBlock themselves still do no masking of any kind
(see pcr/models/relation_blocks.py's own docstring and changes.md's entry on this -- a deliberate,
separately-tracked scope limit, not fixed by this change).

No parallel visibility-weighting mechanism exists for the text side (PromptLearner.part_ctx/
TextualAttentionBlock) -- none is needed. part_ctx is indexed only by identity label
(build_part_prompts(labels)); it has no per-image input at all, so there's no independent
"text visibility" to weight. The gradient that reaches part_ctx/TAB through InfoNCELoss's own
per-image weighting is already an implicitly visibility-weighted aggregate over whichever
instances of that identity are in the current PK batch -- a heavily-occluded instance
contributes proportionally less gradient automatically, as a direct consequence of contributing
less to the (now-weighted) loss value itself. It adapts via the InfoNCE loss alone.

Produces two files consumed by examples/cache_text_anchors.py (not by Stage 2 directly -- see
that script's own docstring for why the final text-prototype computation is a separate step):
prompt_learner.pth (PromptLearner's state, including TextualAttentionBlock and the untrained
fg_ctx) and vab.pth (VisualAttentionBlock's state -- unlike the text-side modules, VAB is *not*
discarded after Stage 1; Stage 2 loads vab.pth to continue training the same
VisualAttentionBlock instance).

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
from torch.utils.data import DataLoader

from pcr import datasets
from pcr.models.bpbreid_encoder import BPBReIDEncoder, BPBReIDModelCfg
from pcr.models.clip_image_encoder import ClipImageEncoder
from pcr.models.clip_text_encoder import ClipTextEncoder
from pcr.models.prompt_learner import PromptLearner
from pcr.models.relation_blocks import VisualAttentionBlock
from pcr.loss.clip_infonce_loss import InfoNCELoss
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


def build_pk_batches(cached_labels, num_instances, batch_size):
    """Groups the cached feature set's indices by identity, then partitions all identities into
    PK batches for one epoch: batch_size // num_instances identities per batch, num_instances
    cached images per identity (sampled with replacement if that identity has fewer than
    num_instances cached images). Algorithm 1 step 6 ("Sample a PK batch of pre-filtered
    images") -- see this file's own module docstring for why this matters even with InfoNCELoss's
    per-identity deduplication making it safe against PK-batch collisions. A final partial group
    of identities (fewer than batch_size // num_instances left over) is dropped, matching this
    repo's other PK
    samplers' drop_last convention (pcr/utils/data/sampler.py::RandomIdentitySampler)."""
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
    # InfoNCELoss's per-part weighting to be meaningfully graduated rather than near-binary.
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

    infonce = InfoNCELoss(temperature=cfg.loss.temperature).cuda()
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
            prompts = prompt_learner.build_part_prompts(b_labels)  # list of 1+K tensors
            part_visual = vab(b_features[:, 1:, :])  # [b, K, D], relationally mixed
            loss = b_features.new_zeros(())
            for k in range(num_parts):
                part_text = text_encoder(prompts[1 + k], prompt_learner.tokenized_prompts).float()
                visual_k = part_visual[:, k, :]
                w_k = b_vis[:, 1 + k]  # same 1+k branch offset as prompts[1+k]/part_visual[:,k,:]
                loss = loss + infonce(visual_k, part_text, b_labels, w_k)

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
    print('==> Saved prompt_learner.pth and vab.pth to {}. Run examples/cache_text_anchors.py '
          'next to build text_prototypes.pth for Stage 2.'.format(cfg.logging.logs_dir))

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    main()
