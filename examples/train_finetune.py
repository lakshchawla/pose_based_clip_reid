"""Stage 2: supervised backbone finetune. BPBreID backbone + attention are trainable; Stage 1's
learned prompts and the CLIP text encoder are both frozen -- only Stage 1's precomputed
text_prototypes.pth (a frozen per-(identity,branch) lookup table) is used here, as the alignment
loss's fixed target. Combines four losses, matching reid_pipeline_plan.md section 3.2 generalized
per-branch and CLIP-ReID's actual stage-2 formula (loss/make_loss.py) for the alignment term:

  - id loss (foreground branch only, matching bpbreid's own default_losses_weights convention
    already documented in pcr/loss/gilt_loss.py): CrossEntropyLabelSmooth via a persistent
    PartIdClassifiers head (real, static labels here -- not per-epoch cluster centers, unlike the
    USL path, since Stage 2 has real identity labels throughout).
  - triplet loss (all branches): pcr/loss/part_triplet_loss.py::PartTripletLoss, visibility-gated
    internally.
  - alignment loss (all branches, visibility-gated per branch): pcr/loss/clip_i2t_loss.py::I2TLoss
    against Stage 1's frozen per-branch text prototypes.
  - BPA loss (optional, source-domain masks only): pcr/loss/body_part_attention_loss.py, off by
    default (data.masks_dir empty in the YAML).

Produces a checkpoint of just the BPBreID model's own state dict (not the whole encoder wrapper),
directly loadable by the existing examples/train_uda.py --checkpoint-path or
examples/train_usl.py --checkpoint-path unchanged.

Config-driven (YAML) -- see configs/stage2_backbone_finetune.yaml. Same deliberate deviation from
the rest of pcr2 as examples/train_prompts.py (train_uda.py/train_usl.py stay argparse-only).
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
from pcr.loss import PartTripletLoss, CrossEntropyLabelSmooth, I2TLoss, BodyPartAttentionLoss
from pcr.evaluators import Evaluator
from pcr.utils.config import load_yaml_config
from pcr.utils.data import IterLoader
from pcr.utils.data import transforms as T
from pcr.utils.data.sampler import RandomIdentitySampler
from pcr.utils.data.preprocessor import Preprocessor, PreprocessorMaskedSingleView
from pcr.utils.logging import Logger
from pcr.utils.osutils import mkdir_if_missing
from pcr.utils.part_distance import branch_visible_mask
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


def get_train_loader(dataset, cfg):
    train_set = sorted(dataset.train)
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


def compute_losses(encoder, id_classifiers, triplet_loss, id_loss, align_loss, bpa_loss,
                    text_prototypes, imgs, mask, targets, cfg):
    use_masks = bpa_loss is not None
    if use_masks:
        f_out, vis, pixels_cls_scores = encoder.forward_full(imgs)
    else:
        f_out, vis = encoder(imgs)
        pixels_cls_scores = None

    total = f_out.new_zeros(())
    log = {}

    id_logits = id_classifiers(f_out, 0)
    l_id = id_loss(id_logits, targets)
    total = total + cfg.loss.id_weight * l_id
    log['id'] = l_id.item()

    result = triplet_loss(f_out, targets, parts_visibility=vis)
    if result is not None:
        l_tri = result[0]
        total = total + cfg.loss.triplet_weight * l_tri
        log['triplet'] = l_tri.item()

    num_branches = f_out.size(1)
    align_total = f_out.new_zeros(())
    for branch in range(num_branches):
        visible = branch_visible_mask(vis[:, branch], cfg.loss.align_visibility_threshold)
        if visible.sum().item() < 1:
            continue
        branch_feats = f_out[visible, branch, :]
        branch_targets = targets[visible]
        branch_prototypes = text_prototypes[:, branch, :]
        align_total = align_total + align_loss(branch_feats, branch_prototypes, branch_targets)
    total = total + cfg.loss.align_weight * align_total
    log['align'] = align_total.item()

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

    triplet_loss = PartTripletLoss(margin=cfg.loss.triplet_margin).cuda()
    id_loss = CrossEntropyLabelSmooth(num_identities).cuda()
    align_loss = I2TLoss(num_identities).cuda()
    use_masks = bool(cfg.data.masks_dir)
    bpa_loss = BodyPartAttentionLoss().cuda() if use_masks else None

    train_loader = get_train_loader(dataset, cfg)
    test_loader = get_test_loader(dataset, cfg.data.height, cfg.data.width, cfg.data.batch_size,
                                   cfg.data.workers)

    if setup_only:
        print('==> Setup complete: {} identities, {} branches, masks {}. Exiting before the '
              'training loop (--setup-only).'.format(
                  num_identities, num_branches, 'ON (' + cfg.data.masks_dir + ')' if use_masks else 'off'))
        return

    params = list(encoder.parameters()) + list(id_classifiers.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.optim.step_size, gamma=0.1)
    evaluator = Evaluator(encoder)

    best_mAP = 0
    for epoch in range(cfg.optim.epochs):
        encoder.train()
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
            loss, log = compute_losses(encoder, id_classifiers, triplet_loss, id_loss, align_loss,
                                        bpa_loss, text_prototypes, imgs, mask, targets, cfg)
            loss.backward()
            optimizer.step()

            if (it + 1) % cfg.logging.print_freq == 0:
                print('Epoch: [{}][{}/{}]\tLoss {:.3f}\t{}'.format(
                    epoch, it + 1, train_iters, loss.item(),
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
            save_checkpoint({
                'state_dict': encoder.model.state_dict(),
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
