"""Stage 1: per-part CLIP prompt learning. Frozen BPBreID encoder + frozen CLIP text encoder;
only PartPromptLearner.cls_ctx trains. Mirrors CLIP-ReID's own do_train_stage1
(../CLIP-ReID/processor/processor_clipreid_stage1.py), generalized per-branch: the whole training
set's part-embeddings are cached once under no_grad, then every step draws a random image
sub-batch (not PK-sampled -- matches the reference's plain shuffled-index sampling) and applies
SupConLoss per visible branch, symmetrically (i2t + t2i).

Produces two files consumed by Stage 2 (examples/train_finetune.py): prompt_learner.pth (the
learned cls_ctx) and text_prototypes.pth (a frozen, precomputed-once [num_identities,
num_branches, embed_dim] text-embedding lookup table).

Config-driven (YAML), not argparse -- see configs/stage1_prompt_learning.yaml. This is a
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
from pcr.models.prompt_learner import PartPromptLearner
from pcr.loss.clip_supcon_loss import SupConLoss
from pcr.utils.config import load_yaml_config
from pcr.utils.data import transforms as T
from pcr.utils.data.preprocessor import Preprocessor
from pcr.utils.logging import Logger
from pcr.utils.lr_scheduler import WarmupCosineLR
from pcr.utils.osutils import mkdir_if_missing
from pcr.utils.part_distance import branch_visible_mask


def get_data(name, data_dir):
    return datasets.create(name, osp.join(data_dir, name))


def get_cache_loader(dataset, height, width, batch_size, workers):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer,
    ])
    return DataLoader(
        Preprocessor(sorted(dataset.train), root=dataset.images_dir, transform=transformer),
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


def compute_text_prototypes(prompt_learner, text_encoder, num_identities, num_branches, id_batch):
    """Frozen, precomputed-once per-(identity,branch) text-embedding lookup table -- built the
    same way CLIP-ReID's stage 2 precomputes its per-identity table right before the epoch loop
    (../CLIP-ReID/processor/processor_clipreid_stage2.py, lines 56-71), generalized per branch."""
    prompt_learner.eval()
    text_prototypes = torch.zeros(num_identities, num_branches, text_encoder.embed_dim,
                                   dtype=torch.float32, device='cuda')
    with torch.no_grad():
        for branch in range(num_branches):
            for start in range(0, num_identities, id_batch):
                ids = torch.arange(start, min(start + id_batch, num_identities), device='cuda')
                prompts = prompt_learner(ids, branch)
                text_feat = text_encoder(prompts, prompt_learner.tokenized_prompts)
                text_prototypes[ids, branch] = text_feat.float()
    return text_prototypes


def main():
    parser = argparse.ArgumentParser(description="PCR Stage 1: per-part CLIP prompt learning")
    parser.add_argument('--config', type=str, required=True, metavar='PATH')
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
    num_branches = 1 + cfg.model.parts_num

    encoder = build_encoder(cfg)
    text_encoder = ClipTextEncoder(clip_arch=cfg.clip.arch, device='cuda').cuda()
    prompt_learner = PartPromptLearner(num_identities, num_branches, text_encoder,
                                        n_ctx=cfg.clip.n_ctx, device='cuda').cuda()

    print("==> Caching part-embeddings for the full training set (frozen encoder, single pass)")
    cache_loader = get_cache_loader(dataset, cfg.data.height, cfg.data.width,
                                     cfg.data.cache_batch_size, cfg.data.workers)
    cached_features, cached_visibility, cached_labels = cache_part_features(encoder, cache_loader)
    cached_features = cached_features.cuda()
    cached_visibility = cached_visibility.cuda()
    cached_labels = cached_labels.cuda()
    num_images = cached_labels.size(0)
    print("==> Cached {} images across {} identities, {} branches".format(
        num_images, num_identities, num_branches))

    if setup_only:
        print('==> Setup complete: {} branches, {} cached images, cls_ctx shape {}. Exiting '
              'before the training loop (--setup-only).'.format(
                  num_branches, num_images, tuple(prompt_learner.cls_ctx.shape)))
        return

    supcon = SupConLoss(temperature=cfg.loss.temperature).cuda()
    optimizer = torch.optim.Adam(prompt_learner.parameters(), lr=cfg.optim.lr,
                                  weight_decay=cfg.optim.weight_decay)
    scheduler = WarmupCosineLR(optimizer, max_epochs=cfg.optim.epochs,
                                warmup_epochs=cfg.optim.warmup_epochs,
                                warmup_lr_init=cfg.optim.warmup_lr_init,
                                lr_min=cfg.optim.lr_min)
    # GradScaler, not raw fp16 backward -- cls_ctx and the CLIP text tower both run in fp16
    # (matches CLIP-ReID's own dtype exactly, see pcr/models/clip_text_encoder.py's docstring),
    # and CLIP-ReID's own stage-1 loop always wraps its backward in a GradScaler to guard
    # against fp16 gradient underflow through the text transformer -- ported faithfully rather
    # than assuming raw fp16 backward is fine.
    scaler = torch.amp.GradScaler('cuda')

    batch = cfg.data.batch_size
    iters_per_epoch = max(1, num_images // batch)

    for epoch in range(cfg.optim.epochs):
        prompt_learner.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        iter_list = torch.randperm(num_images, device='cuda')

        for it in range(iters_per_epoch):
            b_idx = iter_list[it * batch: (it + 1) * batch]
            b_labels = cached_labels[b_idx]
            b_features = cached_features[b_idx]      # [b, M, D], already L2-normalized per branch
            b_visibility = cached_visibility[b_idx]   # [b, M]

            optimizer.zero_grad()
            loss = b_features.new_zeros(())
            any_branch_used = False
            for branch in range(num_branches):
                visible = branch_visible_mask(b_visibility[:, branch], cfg.loss.visibility_threshold)
                if visible.sum().item() < 1:
                    continue
                branch_labels = b_labels[visible]
                image_feat = b_features[visible, branch, :]
                text_feat = prompt_learner(branch_labels, branch)
                text_feat = text_encoder(text_feat, prompt_learner.tokenized_prompts).float()
                loss_i2t = supcon(image_feat, text_feat, branch_labels, branch_labels)
                loss_t2i = supcon(text_feat, image_feat, branch_labels, branch_labels)
                loss = loss + loss_i2t + loss_t2i
                any_branch_used = True

            if not any_branch_used:
                # every branch had zero visible samples in this random sub-batch -- `loss` is
                # still the plain leaf zero tensor from above, with no grad_fn; backward() would
                # crash ("does not require grad"). Only plausible with an untrained/ImageNet-init
                # encoder (the default -- checkpoint_path is optional here), whose visibility head
                # hasn't learned anything reliable yet. Skip this step rather than crash the run.
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

            if (it + 1) % cfg.logging.print_freq == 0:
                print('Epoch: [{}][{}/{}]\tLoss {:.3f}\tLR {:.2e}'.format(
                    epoch, it + 1, iters_per_epoch, loss.item(), optimizer.param_groups[0]['lr']))

        scheduler.step()
        print('Epoch {} done in {:.1f}s, avg loss {:.4f}'.format(
            epoch, time.time() - epoch_start, epoch_loss / iters_per_epoch))

    print("==> Precomputing final per-identity, per-branch text prototype table")
    text_prototypes = compute_text_prototypes(prompt_learner, text_encoder, num_identities,
                                               num_branches, cfg.data.cache_batch_size)

    torch.save(prompt_learner.state_dict(), osp.join(cfg.logging.logs_dir, 'prompt_learner.pth'))
    torch.save({'text_prototypes': text_prototypes.cpu(), 'num_identities': num_identities,
                'num_branches': num_branches},
               osp.join(cfg.logging.logs_dir, 'text_prototypes.pth'))
    print('==> Saved prompt_learner.pth and text_prototypes.pth to {}'.format(cfg.logging.logs_dir))

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    main()
