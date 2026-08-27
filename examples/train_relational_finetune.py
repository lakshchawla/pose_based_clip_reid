"""Stage 2: supervised backbone finetune, implementing "Algorithm 2 -- Stage 2: Backbone
Fine-Tuning" exactly (see progress.md's entry on this file for the full step-by-step mapping);
correspondence to that algorithm's own names:

  Algorithm 2 name           This file / pcr/models
  -----------------          -----------------------
  backbone + BPAM             BPBReIDEncoder, now trainable, initialized from Stage 0's converged
                               checkpoint (the SAME one Stage 1 used going in, since Stage 1 never
                               updates it -- see configs/stage2_relational_finetune.yaml)
  VRB                         VisualAttentionBlock, now trainable, initialized from vab.pth
                               (Stage 1's trained starting point, read from stage1.prompt_dir)
  frozen_text_anchors          text_prototypes.pth (built by examples/cache_text_anchors.py),
                               loaded once; ctx_params/TRB/CLIP text encoder are never loaded
                               here at all -- nothing to "discard", they simply aren't imported
  global ID classifier         PartIdClassifiers (foreground/branch-0 only)
  L_id_global                  id_loss (CrossEntropyLabelSmooth) on the global classifier's logits
  L_tri_global + L_tri_parts    two separate PartTripletLoss calls, NOT one fused call across all
                               branches -- see compute_losses' own comments for why this matters
  L_align                      CosineAlignLoss (pcr/loss/clip_cosine_align_loss.py) -- a softmax
                               classification of each part's feature against that branch's FULL
                               frozen prototype table (every identity acts as an implicit negative
                               via the softmax), restoring CLIP-ReID's own original I2TLoss
                               mechanism rather than the literal per-sample regression Algorithm 2's
                               own wording describes (see changes.md's "Red flag 4" /
                               IMPROVEMENT_PLAN.md section 3 for why the pure-regression version was
                               replaced -- it had no term pushing different identities' features
                               apart at all). Parts only (branches 1..K) -- no foreground/global
                               alignment term at all, matching Stage 1's own scope (fg_ctx is never
                               trained there, so a foreground anchor would be meaningless noise; see
                               changes.md's now-resolved entry on this).
  L_attn                       BodyPartAttentionLoss, mandatory by default now (data.masks_dir
                               defaults to Market1501's real masks, matching Algorithm 2's own
                               unconditional framing) but still optional in code for datasets with
                               no masks on disk (dukemtmc-reid) -- see the config's own comment.

Loss combination in compute_losses() matches Algorithm 2 step 16 exactly: L_attn + L_id_global +
L_tri_global + L_tri_parts + lambda_clip * L_align, where lambda_clip is cfg.loss.align_weight --
the one term the algorithm gives its own explicit coefficient; every other term uses an implicit
weight of 1 in the algorithm's own formula, which this file's id_weight/triplet_weight/bpa_weight
config knobs default to (kept configurable rather than hardcoded to 1, since every other stage in
this repo already exposes its loss weights the same way).

No hard per-branch visibility gating anywhere in these loss computations -- every branch
contributes for every sample, unconditionally. Reliability is handled by *weighting*, not
exclusion, same design as Stage 1: build_encoder switches this stage's own encoder to continuous
(not binary) visibility scores, L_align is weighted per-part by that part's own visibility
(CosineAlignLoss's weights argument, detached before use -- see that file's own docstring for why),
and L_tri_global/L_tri_parts keep a loose hard exclusion via PartTripletLoss's own parts_visibility
argument (batch-hard mining's max/min operations don't compose with soft weights the way
InfoNCE/align's weighted means do -- see configs/stage2_relational_finetune.yaml's
loss.triplet_visibility_min comment). This replaces the
upstream image-level filter Stage 1/2 both used to run (pcr/utils/visibility_filter.py, deleted --
see progress.md's entry on this change) before either stage's training set was ever built: that
filter discarded 61% of Market1501's training images in practice, was the wrong granularity (an
image with 4 good parts and 1 occluded one lost all 4), and was found to be driven by an
undertrained BPAM signal rather than genuine occlusion. VisualAttentionBlock itself still does no
masking of any kind (see pcr/models/relation_blocks.py's own docstring and changes.md's entry on
this -- a deliberate, separately-tracked scope limit, not fixed by this change).

End-of-training checkpoint (Algorithm 2 step 20) bundles VisualAttentionBlock's state into the
SAME saved dict as the encoder's own state ('vab_state_dict' alongside 'state_dict'), rather than
a separate vab.pth -- one checkpoint containing {backbone, BPAM, VRB}, directly loadable by the
existing examples/train_uda.py --checkpoint-path / examples/train_usl.py --checkpoint-path
unchanged (both only ever read the 'state_dict' key, ignoring the rest -- confirmed against
bpbreid's own load_pretrained_weights). Stage 3 stays completely out of this file's scope
otherwise; nothing downstream reads 'vab_state_dict' yet.

Renamed from train_finetune.py -- paired with train_relational_prompts.py's rename.

Config-driven (YAML) -- see configs/stage2_relational_finetune.yaml. Same deliberate deviation
from the rest of pcr2 as examples/train_relational_prompts.py (train_uda.py/train_usl.py stay
argparse-only).
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
from pcr.models.id_classifier import PartIdClassifiers
from pcr.models.relation_blocks import VisualAttentionBlock
from pcr.loss import PartTripletLoss, CrossEntropyLabelSmooth, CosineAlignLoss, BodyPartAttentionLoss
from pcr.evaluators import Evaluator
from pcr.utils.config import load_yaml_config
from pcr.utils.data import IterLoader
from pcr.utils.data import transforms as T
from pcr.utils.data.sampler import RandomIdentitySampler
from pcr.utils.data.preprocessor import Preprocessor, PreprocessorMaskedSingleView
from pcr.utils.logging import Logger
from pcr.utils.osutils import mkdir_if_missing
from pcr.utils.serialization import save_checkpoint, load_checkpoint


def get_data(name, data_dir):
    return datasets.create(name, osp.join(data_dir, name))


def get_photometric_transform():
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return T.Compose([
        T.RandomApply([T.GaussianBlur((.1, 2.))], p=0.5),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]),
    ])


def get_unmasked_transform(height, width):
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
    if cfg.data.masks_dir:
        dataset_wrapper = PreprocessorMaskedSingleView(
            train_set, masks_root=dataset.dataset_dir, masks_dir=cfg.data.masks_dir,
            height=cfg.data.height, width=cfg.data.width,
            photometric_transform=get_photometric_transform(),
            root=dataset.images_dir, mask_suffix=cfg.data.masks_suffix)
    else:
        dataset_wrapper = Preprocessor(train_set, root=dataset.images_dir,
                                        transform=get_unmasked_transform(cfg.data.height, cfg.data.width))
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
    # Continuous visibility scores, not the dataclass's own binary default -- needed for
    # CosineAlignLoss's per-part weighting (and PartTripletLoss's parts_visibility mask) to work
    # as intended. Overridden only here (this stage's own encoder construction), not in
    # BPBReIDModelCfg's shared default -- Stage 3 stays binary, untouched. Unlike Stage 1, this
    # encoder genuinely toggles between .train() (main loop) and .eval() (inside Evaluator.evaluate,
    # called periodically), so both flags are live here.
    model_cfg.training_binary_visibility_score = False
    model_cfg.testing_binary_visibility_score = False
    encoder = BPBReIDEncoder(model_cfg, checkpoint_path=cfg.model.checkpoint_path or None).cuda()
    encoder.train()
    for p in encoder.parameters():
        p.requires_grad_(True)
    return encoder


def mask_to_pixel_targets(mask, pixels_cls_scores):
    """mask: [B, 1+parts_num, H, W] (soft, sums to 1 per pixel). Resized to pixels_cls_scores'
    spatial size and argmax'd into an integer target per pixel -- matches bpbreid's own
    part_based_engine.py::combine_losses and pcr/trainers_usl.py::ICEUSLTrainer._mask_targets."""
    mask = F.interpolate(mask, size=pixels_cls_scores.shape[2:], mode='bilinear', align_corners=True)
    return mask.argmax(dim=1)


def compute_losses(encoder, vab, id_classifiers, triplet_loss, id_loss, align_loss, bpa_loss,
                    text_prototypes, imgs, mask, targets, cfg):
    use_masks = bpa_loss is not None
    if use_masks:
        f_out, vis, pixels_cls_scores = encoder.forward_full(imgs)
    else:
        f_out, vis = encoder(imgs)
        pixels_cls_scores = None

    # VisualAttentionBlock mixes the K part branches only; foreground (branch 0) passes through
    # untouched, then the two are recombined into one [B, 1+K, D] tensor so the rest of this
    # function (triplet/align losses) can keep treating "all branches" uniformly, same as before
    # VAB existed.
    relation_parts = vab(f_out[:, 1:, :])  # [B, K, D]
    combined = torch.cat([f_out[:, 0:1, :], relation_parts], dim=1)  # [B, 1+K, D]
    num_branches = combined.size(1)

    # vis shares combined's exact branch axis (0=foreground, 1..K=parts) -- both are built from
    # the same encoder call. Loose hard exclusion for triplet's batch-hard mining only (soft
    # weights don't compose with max/min mining); continuous weighting for align, below.
    vis_mask = vis >= cfg.loss.triplet_visibility_min  # [B, 1+K] bool

    total = f_out.new_zeros(())
    log = {}

    # Algorithm 2 step 10: L_id_global, the global classifier's cross-entropy on f_g alone.
    id_logits = id_classifiers(combined, 0)
    l_id = id_loss(id_logits, targets)
    total = total + cfg.loss.id_weight * l_id
    log['id'] = l_id.item()

    # Algorithm 2 steps 11-13: L_tri_global (batch-hard triplet on f_g alone) and L_tri_parts (K
    # separate per-part batch-hard triplet losses, summed) -- two independent computations, not
    # one triplet loss fused across all M branches' distances the way this loop used to call
    # PartTripletLoss once on `combined` directly. Calling PartTripletLoss with a single-branch
    # slice ([B, 1, D]) gives that branch's own, unfused batch-hard mining. parts_visibility is a
    # loose boolean exclusion (vis_mask, threshold cfg.loss.triplet_visibility_min) -- the one
    # place hard exclusion still applies in this design; see module docstring.
    global_result = triplet_loss(combined[:, 0:1, :], targets, parts_visibility=vis_mask[:, 0:1])
    if global_result is not None:
        l_tri_global = global_result[0]
        total = total + cfg.loss.triplet_weight * l_tri_global
        log['tri_global'] = l_tri_global.item()

    l_tri_parts = f_out.new_zeros(())
    for branch in range(1, num_branches):
        part_result = triplet_loss(combined[:, branch:branch + 1, :], targets,
                                    parts_visibility=vis_mask[:, branch:branch + 1])
        if part_result is not None:
            l_tri_parts = l_tri_parts + part_result[0]
    total = total + cfg.loss.triplet_weight * l_tri_parts
    log['tri_parts'] = l_tri_parts.item()

    # Algorithm 2 steps 14-15: L_align, a softmax classification of each part's feature against
    # that branch's FULL frozen prototype table (every identity is an implicit negative), summed
    # over the K PART branches only -- no foreground/global alignment term at all, matching Stage
    # 1's own scope exactly (fg_ctx is never trained there -- see progress.md's entry on that
    # rewrite -- so text_prototypes[:, 0, :] is meaningless noise; excluding branch 0 here means
    # that noise is never actually used for anything, resolving changes.md's flagged consequence as
    # a side effect of matching Algorithm 2's own scope, not a separate workaround).
    align_total = f_out.new_zeros(())
    for branch in range(1, num_branches):
        branch_prototypes = text_prototypes[:, branch, :]  # [num_identities, D], full table -- the
                                                             # negatives this loss classifies against
        w = vis[:, branch]  # continuous weighting, not the boolean vis_mask used for triplet
        align_total = align_total + align_loss(combined[:, branch, :], branch_prototypes, targets, weights=w)
    total = total + cfg.loss.align_weight * align_total
    log['align'] = align_total.item()

    # Algorithm 2 steps 9/16: L_attn, mandatory whenever masks are configured (see
    # configs/stage2_relational_finetune.yaml's own comment on why this defaults on now, and why
    # it still needs to stay optional in code for mask-less datasets like dukemtmc-reid).
    if use_masks:
        mask_targets = mask_to_pixel_targets(mask.cuda(), pixels_cls_scores)
        l_bpa, _ = bpa_loss(pixels_cls_scores, mask_targets)
        total = total + cfg.loss.bpa_weight * l_bpa
        log['bpa'] = l_bpa.item()

    return total, log


def main():
    parser = argparse.ArgumentParser(description="PCR Stage 2: supervised backbone finetune")
    parser.add_argument('--config', type=str, required=True, metavar='PATH')
    parser.add_argument('--setup-only', action='store_true',
                         help="build dataset/encoder/losses/loader, print shapes, exit before "
                              "the training loop")
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

    proto_path = osp.join(cfg.stage1.prompt_dir, 'text_prototypes.pth')
    proto = load_checkpoint(proto_path)
    assert proto['num_identities'] == num_identities, (
        "Stage 1's text_prototypes.pth was built for {} identities, this dataset has {} -- "
        "stage1 and stage2 must use the same dataset.".format(proto['num_identities'], num_identities))
    assert proto['num_branches'] == num_branches, (
        "Stage 1's text_prototypes.pth was built for {} branches (1 + parts_num), this config "
        "has parts_num={} ({} branches) -- stage1 and stage2 must agree on model.parts_num."
        .format(proto['num_branches'], cfg.model.parts_num, num_branches))
    text_prototypes = proto['text_prototypes'].cuda()  # [num_identities, num_branches, D]

    encoder = build_encoder(cfg)
    id_classifiers = PartIdClassifiers(num_identities, cfg.model.dim_reduce_output, branches=(0,)).cuda()

    vab = VisualAttentionBlock(dim=cfg.model.dim_reduce_output, num_heads=cfg.vab.num_heads,
                               num_layers=cfg.vab.num_layers).cuda()
    vab_path = osp.join(cfg.stage1.prompt_dir, 'vab.pth')
    vab.load_state_dict(load_checkpoint(vab_path))
    print('==> Loaded Stage 1 VisualAttentionBlock weights from {}'.format(vab_path))

    triplet_loss = PartTripletLoss(margin=cfg.loss.triplet_margin).cuda()
    id_loss = CrossEntropyLabelSmooth(num_identities).cuda()
    align_loss = CosineAlignLoss(temperature=cfg.loss.align_temperature).cuda()
    use_masks = bool(cfg.data.masks_dir)
    bpa_loss = BodyPartAttentionLoss().cuda() if use_masks else None

    train_set = sorted(dataset.train)
    train_loader = get_train_loader(dataset, cfg, train_set)
    test_loader = get_test_loader(dataset, cfg.data.height, cfg.data.width, cfg.data.batch_size,
                                   cfg.data.workers)

    if setup_only:
        print('==> Setup complete: {} identities, {} branches, {} images (no upstream visibility '
              'filtering -- every training image is used, weighted per-part inside the loss), '
              'masks {}. Exiting before the training loop (--setup-only).'.format(
                  num_identities, num_branches, len(train_set),
                  'ON (' + cfg.data.masks_dir + ')' if use_masks else 'off'))
        return

    params = list(encoder.parameters()) + list(id_classifiers.parameters()) + list(vab.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.optim.step_size, gamma=0.1)
    evaluator = Evaluator(encoder)

    best_mAP = 0
    for epoch in range(cfg.optim.epochs):
        encoder.train()
        vab.train()
        train_loader.new_epoch()
        train_iters = len(train_loader)

        epoch_start = time.time()
        for it in range(train_iters):
            inputs = train_loader.next()
            if use_masks:
                imgs, mask, targets, _, _ = inputs
            else:
                imgs, _, targets, _, _ = inputs
                mask = None
            imgs = imgs.cuda()
            targets = targets.cuda()

            optimizer.zero_grad()
            loss, log = compute_losses(encoder, vab, id_classifiers, triplet_loss, id_loss,
                                        align_loss, bpa_loss, text_prototypes, imgs, mask, targets, cfg)
            loss.backward()
            optimizer.step()

            if (it + 1) % cfg.logging.print_freq == 0:
                print('Epoch: [{}][{}/{}]\tLoss {:.3f}\tVAB gate {:.3f}\t{}'.format(
                    epoch, it + 1, train_iters, loss.item(), torch.tanh(vab.gate).item(),
                    '\t'.join('{} {:.3f}'.format(k, v) for k, v in log.items())))

        lr_scheduler.step()
        print('Epoch {} done in {:.1f}s'.format(epoch, time.time() - epoch_start))

        if (epoch + 1) % cfg.logging.eval_step == 0 or epoch == cfg.optim.epochs - 1:
            # float(): mean_ap() (pcr/evaluation_metrics/ranking.py) returns numpy.float64, not
            # a plain float. Found by actually round-tripping a saved checkpoint through
            # bpbreid's own torchreid.utils.load_pretrained_weights (the real consumer, via
            # train_uda.py/train_usl.py --checkpoint-path): its load_checkpoint doesn't pass
            # weights_only=False, so PyTorch 2.6+'s stricter weights_only=True default rejects a
            # numpy scalar sitting anywhere in the pickled checkpoint dict with an
            # UnpicklingError -- this repo's own pcr.utils.serialization.load_checkpoint already
            # passes weights_only=False and would have hidden this, so only testing the actual
            # cross-repo consumption path caught it.
            mAP = float(evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False))
            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            # Algorithm 2 step 20: one saved checkpoint containing {backbone, BPAM, VRB} -- 'vab_
            # state_dict' rides alongside 'state_dict' in the same file rather than a separate
            # vab.pth, harmless to any consumer reading only 'state_dict' (bpbreid's own
            # load_pretrained_weights explicitly reads that one key and ignores the rest).
            save_checkpoint({
                'state_dict': encoder.model.state_dict(),
                'vab_state_dict': vab.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
                'optimizer': optimizer.state_dict(),
            }, is_best, fpath=osp.join(cfg.logging.logs_dir, 'checkpoint.pth.tar'))
            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.format(
                epoch, mAP, best_mAP, ' *' if is_best else ''))

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
