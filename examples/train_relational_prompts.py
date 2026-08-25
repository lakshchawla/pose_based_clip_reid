"""Stage 1: per-part CLIP prompt learning, with relational mixing across a person's K part
tokens on both sides -- TextualAttentionBlock (owned by PromptLearner) on the text side,
VisualAttentionBlock on the image side. Frozen BPBreID encoder + frozen CLIP text encoder; the
only trainable things are PromptLearner (which includes TextualAttentionBlock as a submodule) and
VisualAttentionBlock. Mirrors CLIP-ReID's own do_train_stage1
(../CLIP-ReID/processor/processor_clipreid_stage1.py), generalized per-branch: the whole
(visibility-filtered) training set's part-embeddings are cached once under no_grad, then every
step draws a random image sub-batch (not PK-sampled -- matches the reference's plain
shuffled-index sampling) and applies SupConLoss per branch, symmetrically (i2t + t2i).

Two scope notes, both explicit decisions logged in progress.md, not defaults slipped in quietly:
- No per-branch visibility gating inside the loss loop -- every branch contributes for every
  cached sample, unconditionally. Reliability is handled once, upstream, by the visibility filter
  below (pcr/utils/visibility_filter.py) rejecting whole images before caching, not by skipping
  individual (sample, branch) pairs during training the way the pre-relational-attention version
  of this file did.
- VisualAttentionBlock and TextualAttentionBlock both operate on the K=5 part branches only. The
  foreground/global branch (branch 0) keeps its own independent context and pooled feature,
  untouched by either relation block -- see pcr/models/relation_blocks.py's module docstring.

Renamed from train_prompts.py -- this is the file that changed for the part-relational-attention
plan (see progress.md); train_finetune.py became train_relational_finetune.py for the same
reason (Stage 2 also changed, to carry VisualAttentionBlock forward and drop visibility gating).

Produces two files consumed by examples/cache_text_anchors.py (not by Stage 2 directly -- see
that script's own docstring for why the final text-prototype computation is a separate step):
prompt_learner.pth (PromptLearner's state, including TextualAttentionBlock) and vab.pth
(VisualAttentionBlock's state -- unlike the text-side modules, VAB is *not* discarded after Stage
1; Stage 2 loads vab.pth to continue training the same VisualAttentionBlock instance).

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
from pcr.utils.visibility_filter import filter_by_visibility


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


def build_encoder(cfg):
    model_cfg = BPBReIDModelCfg(backbone=cfg.model.backbone)
    model_cfg.masks.parts_num = cfg.model.parts_num
    model_cfg.dim_reduce_output = cfg.model.dim_reduce_output
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
    prompt_learner = PromptLearner(num_identities, num_parts, text_encoder, n_ctx=cfg.clip.n_ctx,
                                    tab_num_heads=cfg.tab.num_heads, tab_num_layers=cfg.tab.num_layers,
                                    device='cuda').cuda()
    vab = VisualAttentionBlock(dim=cfg.model.dim_reduce_output, num_heads=cfg.vab.num_heads,
                               num_layers=cfg.vab.num_layers).cuda()

    print("==> Filtering training set by visibility index (threshold={})".format(
        cfg.visibility.lambda_v_min))
    train_set, _ = filter_by_visibility(
        sorted(dataset.train), encoder, cfg.data.height, cfg.data.width,
        cfg.visibility.lambda_v_min, root=dataset.images_dir, batch_size=cfg.data.cache_batch_size,
        workers=cfg.data.workers)

    print("==> Caching part-embeddings for the filtered training set (frozen encoder, single pass)")
    cache_loader = get_cache_loader(train_set, dataset.images_dir, cfg.data.height, cfg.data.width,
                                     cfg.data.cache_batch_size, cfg.data.workers)
    cached_features, cached_visibility, cached_labels = cache_part_features(encoder, cache_loader)
    cached_features = cached_features.cuda()
    cached_labels = cached_labels.cuda()
    num_images = cached_labels.size(0)
    print("==> Cached {} images across {} identities, {} branches".format(
        num_images, num_identities, num_branches))

    if setup_only:
        print('==> Setup complete: {} branches, {} cached images, fg_ctx shape {}, part_ctx '
              'shape {}. Exiting before the training loop (--setup-only).'.format(
                  num_branches, num_images, tuple(prompt_learner.fg_ctx.shape),
                  tuple(prompt_learner.part_ctx.shape)))
        return

    supcon = SupConLoss(temperature=cfg.loss.temperature).cuda()
    trainable_params = list(prompt_learner.parameters()) + list(vab.parameters())
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

    batch = cfg.data.batch_size
    iters_per_epoch = max(1, num_images // batch)

    for epoch in range(cfg.optim.epochs):
        prompt_learner.train()
        vab.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        iter_list = torch.randperm(num_images, device='cuda')

        for it in range(iters_per_epoch):
            b_idx = iter_list[it * batch: (it + 1) * batch]
            b_labels = cached_labels[b_idx]
            b_features = cached_features[b_idx]  # [b, 1+K, D], already L2-normalized per branch

            optimizer.zero_grad()

            prompts = prompt_learner.build_part_prompts(b_labels)  # list of 1+K tensors
            fg_text = text_encoder(prompts[0], prompt_learner.tokenized_prompts).float()
            fg_visual = b_features[:, 0, :]
            loss = supcon(fg_visual, fg_text, b_labels, b_labels) \
                + supcon(fg_text, fg_visual, b_labels, b_labels)

            part_visual = vab(b_features[:, 1:, :])  # [b, K, D], relationally mixed
            for k in range(num_parts):
                part_text = text_encoder(prompts[1 + k], prompt_learner.tokenized_prompts).float()
                visual_k = part_visual[:, k, :]
                loss = loss + supcon(visual_k, part_text, b_labels, b_labels) \
                    + supcon(part_text, visual_k, b_labels, b_labels)

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
