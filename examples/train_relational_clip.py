"""Single-stage alternative to the Stage 1 + cache_text_anchors.py + Stage 2 pipeline
(examples/train_relational_prompts.py, cache_text_anchors.py, train_relational_finetune.py):
trains every learnable module jointly, in one forward/backward pass per iteration, using the
original CLIP paper's own training loss (pcr/loss/clip_contrastive_loss.py::ClipContrastiveLoss)
instead of CLIP-ReID's two-stage SupCon-then-frozen-I2T scheme.

What "jointly" means here, concretely: BPBreID's backbone + BPA pixel-classifier (fully
trainable, unlike Stage 1's frozen backbone), VisualAttentionBlock, and PromptLearner (including
TextualAttentionBlock) all update from the same backward pass, every iteration -- no cross-stage
checkpoint hand-off, no frozen prompts, no precomputed text-prototype table. Every module is
freshly initialized here (aside from the optional model.checkpoint_path, e.g. Stage 0's BPA
pretraining or an external bpbreid checkpoint) -- there is no stage1.prompt_dir to load from.
The frozen CLIP text tower itself is unchanged from every other stage in this repo: still always
requires_grad_(False) (see pcr/models/clip_text_encoder.py) -- "generic CLIP training" here means
following the CLIP paper's own *loss*, not unfreezing CLIP's own pretrained text transformer.

Two loss terms, matching exactly what was asked for -- no id loss, no BPA loss (unlike Stage 2's
four-loss combination):
  - ClipContrastiveLoss, per branch (foreground + K VisualAttentionBlock-mixed parts), summed --
    the paper's own diagonal symmetric cross-entropy over a learned temperature, computed between
    each branch's visual embedding and that identity's text embedding from PromptLearner +
    TextualAttentionBlock + the frozen CLIP text encoder, all built freshly every iteration (no
    caching -- unlike Stage 1, the visual encoder here is not frozen, so nothing is cacheable).
  - PartTripletLoss, across all branches jointly (pcr/loss/part_triplet_loss.py), same as Stage 2.

Batches are PK-sampled (RandomIdentitySampler, same as Stage 2) because the triplet loss needs
multiple instances per identity per batch -- see pcr/loss/clip_contrastive_loss.py's own docstring
for the resulting, accepted tradeoff against the CLIP paper's own diagonal-only assumption.

Produces a checkpoint in exactly Stage 2's own save format ({'state_dict':
encoder.model.state_dict(), ...}), directly loadable by train_uda.py/train_usl.py
--checkpoint-path unchanged, plus vab.pth and prompt_learner.pth for completeness (no downstream
script in this design actually needs to reload them, unlike Stage 2's vab.pth -> nothing, or
Stage 1's prompt_learner.pth -> cache_text_anchors.py).

Config-driven (YAML), matching the rest of this repo's CLIP-adjacent scripts -- see
configs/relational_clip.yaml.
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
from pcr.loss import PartTripletLoss, ClipContrastiveLoss
from pcr.evaluators import Evaluator
from pcr.utils.config import load_yaml_config
from pcr.utils.data import IterLoader
from pcr.utils.data import transforms as T
from pcr.utils.data.sampler import RandomIdentitySampler
from pcr.utils.data.preprocessor import Preprocessor
from pcr.utils.logging import Logger
from pcr.utils.lr_scheduler import WarmupCosineLR
from pcr.utils.osutils import mkdir_if_missing
from pcr.utils.serialization import save_checkpoint, load_checkpoint
from pcr.utils.visibility_filter import filter_by_visibility


def get_data(name, data_dir):
    return datasets.create(name, osp.join(data_dir, name))


def get_train_transform(height, width):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]),
    ])


def get_train_loader(dataset, cfg, train_set):
    sampler = RandomIdentitySampler(train_set, cfg.data.num_instances)
    dataset_wrapper = Preprocessor(train_set, root=dataset.images_dir,
                                    transform=get_train_transform(cfg.data.height, cfg.data.width))
    return IterLoader(
        DataLoader(dataset_wrapper, batch_size=cfg.data.batch_size, num_workers=cfg.data.workers,
                   sampler=sampler, pin_memory=True, drop_last=True))


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transformer = T.Compose([T.Resize((height, width), interpolation=3), T.ToTensor(), normalizer])
    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))
    return DataLoader(Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
                       batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)


def build_encoder(cfg):
    model_cfg = BPBReIDModelCfg(backbone=cfg.model.backbone)
    model_cfg.masks.parts_num = cfg.model.parts_num
    model_cfg.dim_reduce_output = cfg.model.dim_reduce_output
    encoder = BPBReIDEncoder(model_cfg, checkpoint_path=cfg.model.checkpoint_path or None).cuda()
    encoder.train()
    for p in encoder.parameters():
        p.requires_grad_(True)
    return encoder


def compute_losses(encoder, vab, prompt_learner, text_encoder, triplet_loss, clip_loss,
                    imgs, targets, cfg):
    f_out, vis = encoder(imgs)  # [B, 1+K, D], each branch already L2-normalized

    # Same VisualAttentionBlock convention as every other stage: mixes the K part branches only,
    # foreground passes through untouched, recombined into one [B, 1+K, D] tensor.
    relation_parts = vab(f_out[:, 1:, :])
    combined = torch.cat([f_out[:, 0:1, :], relation_parts], dim=1)

    prompts = prompt_learner.build_part_prompts(targets)  # list of 1+K tensors, one g_theta() call each

    total = f_out.new_zeros(())
    log = {}

    contrastive_total = f_out.new_zeros(())
    for branch in range(combined.size(1)):
        text_feat = text_encoder(prompts[branch], prompt_learner.tokenized_prompts).float()
        contrastive_total = contrastive_total + clip_loss(combined[:, branch, :], text_feat)
    total = total + cfg.loss.contrastive_weight * contrastive_total
    log['contrastive'] = contrastive_total.item()

    # No parts_visibility argument -- every branch contributes for every sample unconditionally,
    # same upstream-filter convention as Stage 1/2 (reid_pipeline_plan.md's part-relational
    # -attention addendum, section 0.1).
    result = triplet_loss(combined, targets)
    if result is not None:
        l_tri = result[0]
        total = total + cfg.loss.triplet_weight * l_tri
        log['triplet'] = l_tri.item()

    return total, log


def main():
    parser = argparse.ArgumentParser(
        description="PCR single-stage CLIP training: every module trained jointly, using the "
                    "CLIP paper's own contrastive loss instead of CLIP-ReID's two-stage scheme")
    parser.add_argument('--config', type=str, metavar='PATH', default="configs/relational_clip.yaml")
    parser.add_argument('--setup-only', action='store_true',
                         help="build dataset/encoder/prompt-learner/loader, print shapes, exit "
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

    encoder = build_encoder(cfg)
    text_encoder = ClipTextEncoder(clip_arch=cfg.clip.arch, device='cuda').cuda()
    prompt_learner = PromptLearner(num_identities, num_parts, text_encoder, n_ctx=cfg.clip.n_ctx,
                                    tab_num_heads=cfg.tab.num_heads, tab_num_layers=cfg.tab.num_layers,
                                    device='cuda').cuda()
    vab = VisualAttentionBlock(dim=cfg.model.dim_reduce_output, num_heads=cfg.vab.num_heads,
                               num_layers=cfg.vab.num_layers).cuda()

    triplet_loss = PartTripletLoss(margin=cfg.loss.triplet_margin).cuda()
    clip_loss = ClipContrastiveLoss().cuda()

    print("==> Filtering training set by visibility index (threshold={})".format(
        cfg.visibility.lambda_v_min))
    encoder.eval()
    filtered_train_set, _ = filter_by_visibility(
        sorted(dataset.train), encoder, cfg.data.height, cfg.data.width,
        cfg.visibility.lambda_v_min, root=dataset.images_dir, batch_size=cfg.data.batch_size,
        workers=cfg.data.workers)
    encoder.train()

    train_loader = get_train_loader(dataset, cfg, filtered_train_set)
    test_loader = get_test_loader(dataset, cfg.data.height, cfg.data.width, cfg.data.batch_size,
                                   cfg.data.workers)

    if setup_only:
        print('==> Setup complete: {} identities, {} images after visibility filtering, fg_ctx '
              'shape {}, part_ctx shape {}. Exiting before the training loop (--setup-only).'.format(
                  num_identities, len(filtered_train_set), tuple(prompt_learner.fg_ctx.shape),
                  tuple(prompt_learner.part_ctx.shape)))
        return

    params = (list(encoder.parameters()) + list(vab.parameters())
              + list(prompt_learner.parameters()) + list(clip_loss.parameters()))
    optimizer = torch.optim.Adam(params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    scheduler = WarmupCosineLR(optimizer, max_epochs=cfg.optim.epochs,
                                warmup_epochs=cfg.optim.warmup_epochs,
                                warmup_lr_init=cfg.optim.warmup_lr_init,
                                lr_min=cfg.optim.lr_min)
    # GradScaler, same rationale as Stage 1 (examples/train_relational_prompts.py): the CLIP text
    # tower runs in fp16 every iteration here too (unlike Stage 2, where prompts/text encoder were
    # frozen and never touched), so the same fp16-gradient-underflow guard applies. Every trainable
    # parameter here (encoder, VAB, PromptLearner, ClipContrastiveLoss's logit_scale) is fp32, so
    # the scaler remains harmless for all of them, exactly as Stage 1 already establishes.
    scaler = torch.amp.GradScaler('cuda')
    evaluator = Evaluator(encoder)

    best_mAP = 0
    for epoch in range(cfg.optim.epochs):
        encoder.train()
        vab.train()
        prompt_learner.train()
        train_loader.new_epoch()
        train_iters = len(train_loader)

        epoch_start = time.time()
        for it in range(train_iters):
            imgs, _, targets, _, _ = train_loader.next()
            imgs = imgs.cuda()
            targets = targets.cuda()

            optimizer.zero_grad()
            loss, log = compute_losses(encoder, vab, prompt_learner, text_encoder, triplet_loss,
                                        clip_loss, imgs, targets, cfg)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (it + 1) % cfg.logging.print_freq == 0:
                print('Epoch: [{}][{}/{}]\tLoss {:.3f}\tVAB gate {:.3f}\tlogit_scale {:.3f}\t{}'.format(
                    epoch, it + 1, train_iters, loss.item(), torch.tanh(vab.gate).item(),
                    clip_loss.logit_scale.exp().item(),
                    '\t'.join('{} {:.3f}'.format(k, v) for k, v in log.items())))

        scheduler.step()
        print('Epoch {} done in {:.1f}s'.format(epoch, time.time() - epoch_start))

        if (epoch + 1) % cfg.logging.eval_step == 0 or epoch == cfg.optim.epochs - 1:
            # float(): see examples/train_relational_finetune.py's identical cast for why
            # (numpy.float64 in a checkpoint dict breaks PyTorch 2.6+'s stricter weights_only=True
            # default in bpbreid's own loader).
            mAP = float(evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False))
            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': encoder.model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
                'optimizer': optimizer.state_dict(),
            }, is_best, fpath=osp.join(cfg.logging.logs_dir, 'checkpoint.pth.tar'))
            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.format(
                epoch, mAP, best_mAP, ' *' if is_best else ''))

    torch.save(vab.state_dict(), osp.join(cfg.logging.logs_dir, 'vab.pth'))
    torch.save(prompt_learner.state_dict(), osp.join(cfg.logging.logs_dir, 'prompt_learner.pth'))
    print('==> Saved vab.pth and prompt_learner.pth to {} (kept for completeness/inspection -- no '
          'downstream script in this single-stage design needs to reload either)'.format(
              cfg.logging.logs_dir))

    print('==> Test with the best model:')
    best_fpath = osp.join(cfg.logging.logs_dir, 'model_best.pth.tar')
    if osp.isfile(best_fpath):
        checkpoint = load_checkpoint(best_fpath)
        encoder.model.load_state_dict(checkpoint['state_dict'])
    else:
        print('No model_best.pth.tar in {}, testing with the final model'.format(cfg.logging.logs_dir))
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    main()
