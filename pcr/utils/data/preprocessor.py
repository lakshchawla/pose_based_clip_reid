from __future__ import absolute_import
import os.path as osp
import random

import numpy as np
import torch
import torch.nn.functional as nnF
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomCrop
from torch.utils.data import Dataset
from PIL import Image


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, camid, index


class Preprocessor2View(Dataset):
    """Two independently-augmented views of the same image, both under `transform`. Used by
    the ICE-style USL pipeline (pcr/trainers_usl.py): one view feeds the student encoder, the
    other feeds the EMA teacher -- that's the whole "instance discrimination between two views"
    setup the hard-instance contrastive loss operates on. No index is returned: unlike the UDA
    pipeline's PartHybridMemory, this training loop keeps no per-instance memory slots, so the
    pseudo-label already baked into the dataset tuple's `pid` field is all that's needed."""

    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor2View, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')
        img1 = self.transform(img)
        img2 = self.transform(img.copy())
        return img1, img2, pid, camid


class PreprocessorMasked(Dataset):
    """Like Preprocessor2View, but view 1 is paired with its PifPaf ground-truth part mask for
    BodyPartAttentionLoss (view 2, feeding the EMA teacher, never needs a mask). The geometric
    steps (resize, pad, crop, flip) are applied identically to image and mask via shared random
    parameters -- deliberately hand-rolled instead of pulling in bpbreid's own albumentations-
    based mask pipeline, since only these 4 ops need pairing at all (colour-only augmentation --
    blur, erasing -- doesn't touch pixel *positions*, so it's applied to the image alone,
    afterward, same as the rest of this repo's transforms). Channel grouping (36 raw PifPaf
    channels -> 5 named parts) and the synthetic background channel are computed via bpbreid's
    own torchreid.data.masks_transforms classes, reused directly rather than re-deriving their
    channel-mapping tables by hand (imported lazily so plain Preprocessor/Preprocessor2View usage
    never requires torchreid's masks_transforms module, or albumentations, at all).
    """

    def __init__(self, dataset, masks_root, masks_dir, height, width, photometric_transform,
                 view2_transform, root=None, pad=10, flip_p=0.5, mask_suffix='.npy'):
        super(PreprocessorMasked, self).__init__()
        from torchreid.data.masks_transforms import CombinePifPafIntoFiveVerticalParts, AddBackgroundMask

        self.dataset = dataset
        self.root = root
        self.masks_root = masks_root
        self.masks_dir = masks_dir
        self.mask_suffix = mask_suffix
        self.height = height
        self.width = width
        self.pad = pad
        self.flip_p = flip_p
        self.photometric_transform = photometric_transform
        self.view2_transform = view2_transform
        self.group_masks = CombinePifPafIntoFiveVerticalParts()
        self.add_background = AddBackgroundMask()

    def __len__(self):
        return len(self.dataset)

    def _load_raw_mask(self, fname):
        split_folder = osp.basename(osp.dirname(fname))
        basename = osp.splitext(osp.basename(fname))[0]
        mask_path = osp.join(self.masks_root, 'masks', self.masks_dir, split_folder,
                              basename + self.mask_suffix)
        raw = np.load(mask_path)  # [C, H, W]
        return torch.from_numpy(raw).float()

    def __getitem__(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = osp.join(self.root, fname) if self.root is not None else fname

        img = Image.open(fpath).convert('RGB')
        img2 = self.view2_transform(img.copy())

        mask = self._load_raw_mask(fname)
        mask = nnF.interpolate(mask.unsqueeze(0), size=(self.height, self.width),
                                mode='bilinear', align_corners=True).squeeze(0)

        img1 = TF.resize(img, [self.height, self.width])
        img1 = TF.pad(img1, self.pad)
        mask = nnF.pad(mask, (self.pad, self.pad, self.pad, self.pad))

        i, j, h, w = RandomCrop.get_params(img1, output_size=(self.height, self.width))
        img1 = TF.crop(img1, i, j, h, w)
        mask = mask[:, i:i + h, j:j + w]

        if random.random() < self.flip_p:
            img1 = TF.hflip(img1)
            mask = mask.flip(dims=[-1])

        mask = self.group_masks.apply_to_mask(mask)
        mask = self.add_background.apply_to_mask(mask)  # [1+parts_num, height, width]

        img1 = self.photometric_transform(img1)

        return img1, img2, mask, pid, camid
