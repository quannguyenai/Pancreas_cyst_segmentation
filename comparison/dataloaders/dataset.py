"""dataset.py — Unified dataset classes for 3-D (Cyst) and 2-D slice (Cyst2D) loading.

Both classes read split CSV files produced by data/prepare_dataset.py.
Paths in those CSVs are absolute (resolved via configs/paths.yaml at prep time).

Key differences from original baseline/ code
---------------------------------------------
* Accepts a ``txt_path`` argument directly (no hardcoded base_dir string concat).
* ``Cyst2D`` in training mode avoids loading every volume at ``__init__`` time by
  reading a pre-built ``slice_index.json`` written by prepare_dataset.py.
  If that file does not exist it falls back to the original on-the-fly counting.
* Removed ``import pdb`` debug leftover.
"""

from __future__ import annotations

import itertools
import json
import os
import random
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from skimage import transform as sk_trans
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


# ─── 3-D dataset ──────────────────────────────────────────────────────────────

class Cyst(Dataset):
    """Load 3-D NIfTI volumes for training/validation.

    Parameters
    ----------
    txt_path : str | Path
        Path to a split CSV (header: image_path,mask_path).
    num : int, optional
        Limit training samples to the first ``num`` entries.
    transform : callable, optional
        Transform applied to each ``{'image': np.ndarray, 'label': np.ndarray}``
        sample dict.
    """

    def __init__(
        self,
        txt_path: str | Path,
        num: int | None = None,
        transform=None,
    ):
        self.transform = transform
        txt_path = Path(txt_path)

        with open(txt_path) as f:
            lines = f.readlines()[1:]  # skip header
        self.sample_list = [item.strip().split(",") for item in lines if item.strip()]

        if num is not None:
            self.sample_list = self.sample_list[:num]

        print(f"[Cyst] {txt_path.name}: {len(self.sample_list)} samples")

    def __len__(self) -> int:
        return len(self.sample_list)

    def __getitem__(self, idx: int) -> dict:
        image_path, mask_path = self.sample_list[idx]
        image = nib.load(image_path).get_fdata().astype("float32")
        label = nib.load(mask_path).get_fdata().astype("float32")
        sample = {"image": image, "label": label}
        if self.transform:
            sample = self.transform(sample)
        sample["case"] = image_path
        return sample


# ─── 2-D slice dataset ────────────────────────────────────────────────────────

class Cyst2D(Dataset):
    """Load 2-D axial slices extracted from 3-D NIfTI volumes.

    In training mode each slice is a separate sample. In validation mode
    the full 3-D volume is returned for volumetric metric computation.

    Parameters
    ----------
    txt_path : str | Path
        Path to a split CSV.
    split : {'train', 'val'}
    slice_index_json : str | Path, optional
        Path to a JSON file mapping ``image_path → num_slices`` written by
        ``prepare_dataset.py``. Avoids loading all volumes at ``__init__``.
    transform : callable, optional
    """

    def __init__(
        self,
        txt_path: str | Path,
        split: str = "train",
        slice_index_json: str | Path | None = None,
        transform=None,
    ):
        self.split = split
        self.transform = transform
        txt_path = Path(txt_path)

        with open(txt_path) as f:
            lines = f.readlines()[1:]
        self.sample_list = [item.strip().split(",") for item in lines if item.strip()]

        self.slice_list: list[tuple[str, str, int]] = []

        if split == "train":
            # Try fast path: pre-built slice index
            if slice_index_json and Path(slice_index_json).exists():
                depth_map: dict[str, int] = json.loads(
                    Path(slice_index_json).read_text()
                )
                for image_path, mask_path in self.sample_list:
                    depth = depth_map.get(image_path)
                    if depth is None:
                        depth = nib.load(image_path).shape[2]
                    for z in range(depth):
                        self.slice_list.append((image_path, mask_path, z))
            else:
                # Fallback: load every volume to count slices (slow on cold storage)
                for image_path, mask_path in self.sample_list:
                    depth = nib.load(image_path).shape[2]
                    for z in range(depth):
                        self.slice_list.append((image_path, mask_path, z))

        print(
            f"[Cyst2D/{split}] {txt_path.name}: "
            + (f"{len(self.slice_list)} slices" if split == "train"
               else f"{len(self.sample_list)} volumes")
        )

    def __len__(self) -> int:
        if self.split == "val":
            return len(self.sample_list)
        return len(self.slice_list)

    def __getitem__(self, idx: int) -> dict:
        if self.split == "val":
            image_path, mask_path = self.sample_list[idx]
            image = nib.load(image_path).get_fdata().astype(np.float32)
            label = nib.load(mask_path).get_fdata().astype(np.uint8)
            return {"image": image, "label": label, "case": image_path}

        image_path, mask_path, z = self.slice_list[idx]
        image = nib.load(image_path).get_fdata()[:, :, z].astype(np.float32)
        label = nib.load(mask_path).get_fdata()[:, :, z].astype(np.uint8)
        sample = {"image": image, "label": label}
        if self.transform:
            sample = self.transform(sample)
        sample["case"] = image_path
        return sample


# ─── Transforms ───────────────────────────────────────────────────────────────

def random_rot_flip(image: np.ndarray, label: np.ndarray):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image: np.ndarray, label: np.ndarray):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator:
    """2-D random augmentation: rot/flip + zoom to fixed output size."""

    def __init__(self, output_size: tuple[int, int]):
        self.output_size = output_size

    def __call__(self, sample: dict) -> dict:
        image, label = sample["image"], sample["label"]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        return {
            "image": torch.from_numpy(image.astype(np.float32)).unsqueeze(0),
            "label": torch.from_numpy(label.astype(np.uint8)).long(),
        }


class Resize:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample: dict) -> dict:
        image, label = sample["image"], sample["label"]
        label = label.astype(bool)
        image = sk_trans.resize(image, self.output_size, order=1, mode="constant", cval=0)
        label = sk_trans.resize(label, self.output_size, order=0)
        return {"image": image, "label": label}


class Resize3D:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample: dict) -> dict:
        image, label = sample["image"], sample["label"]
        label = label.astype(bool)
        image = sk_trans.resize(image, self.output_size, order=1, mode="constant", cval=0)
        label = sk_trans.resize(label, self.output_size, order=0)
        return {"image": image, "label": label}


class CenterCrop:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample: dict) -> dict:
        image, label = sample["image"], sample["label"]
        for dim, sz in enumerate(self.output_size):
            if image.shape[dim] <= sz:
                pad = [(0, 0)] * 3
                pad[dim] = (max((sz - image.shape[dim]) // 2 + 3, 0),) * 2
                image = np.pad(image, pad, mode="constant", constant_values=0)
                label = np.pad(label, pad, mode="constant", constant_values=0)
        w, h, d = image.shape
        w1 = int(round((w - self.output_size[0]) / 2.0))
        h1 = int(round((h - self.output_size[1]) / 2.0))
        d1 = int(round((d - self.output_size[2]) / 2.0))
        image = image[w1:w1+self.output_size[0], h1:h1+self.output_size[1], d1:d1+self.output_size[2]]
        label = label[w1:w1+self.output_size[0], h1:h1+self.output_size[1], d1:d1+self.output_size[2]]
        return {"image": image, "label": label}


class RandomCrop:
    def __init__(self, output_size, with_sdf: bool = False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample: dict) -> dict:
        image, label = sample["image"], sample["label"]
        sdf = sample.get("sdf")
        # Pad if needed
        pw = max((self.output_size[0] - image.shape[0]) // 2 + 3, 0)
        ph = max((self.output_size[1] - image.shape[1]) // 2 + 3, 0)
        pd = max((self.output_size[2] - image.shape[2]) // 2 + 3, 0)
        if pw or ph or pd:
            image = np.pad(image, [(pw,pw),(ph,ph),(pd,pd)], mode="constant", constant_values=0)
            label = np.pad(label, [(pw,pw),(ph,ph),(pd,pd)], mode="constant", constant_values=0)
            if sdf is not None:
                sdf = np.pad(sdf, [(pw,pw),(ph,ph),(pd,pd)], mode="constant", constant_values=0)
        w, h, d = image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])
        image = image[w1:w1+self.output_size[0], h1:h1+self.output_size[1], d1:d1+self.output_size[2]]
        label = label[w1:w1+self.output_size[0], h1:h1+self.output_size[1], d1:d1+self.output_size[2]]
        if sdf is not None:
            sdf = sdf[w1:w1+self.output_size[0], h1:h1+self.output_size[1], d1:d1+self.output_size[2]]
            return {"image": image, "label": label, "sdf": sdf}
        return {"image": image, "label": label}


class RandomRotFlip:
    def __call__(self, sample: dict) -> dict:
        image, label = random_rot_flip(sample["image"], sample["label"])
        return {"image": image, "label": label}


class RandomRot:
    def __call__(self, sample: dict) -> dict:
        image, label = random_rotate(sample["image"], sample["label"])
        return {"image": image, "label": label}


class RandomNoise:
    def __init__(self, mean: float = 0.0, std: float = 0.05, p: float = 0.5):
        self.mean, self.std, self.p = mean, std, p

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.p:
            noise = np.random.normal(self.mean, self.std, sample["image"].shape)
            sample["image"] = np.clip(sample["image"] + noise, 0, 1)
        return sample


class RandomBlur:
    def __init__(self, sigma_range=(0.5, 1.5), p: float = 0.5):
        self.sigma_range, self.p = sigma_range, p

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.p:
            sigma = random.uniform(*self.sigma_range)
            ksize = int(2 * round(3 * sigma) + 1)
            sample["image"] = cv2.GaussianBlur(sample["image"], (ksize, ksize), sigma)
        return sample


class RandomGamma:
    def __init__(self, gamma_range=(0.7, 1.5), p: float = 0.5):
        self.gamma_range, self.p = gamma_range, p

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.p:
            gamma = random.uniform(*self.gamma_range)
            sample["image"] = np.clip(np.power(sample["image"], gamma), 0, 1)
        return sample


class Normalize:
    """Foreground mean-std normalisation (foreground = voxels > 0)."""

    def __call__(self, sample: dict) -> dict:
        image = sample["image"].astype(np.float32)
        fg = image > 0
        mean = image[fg].mean() if fg.any() else 0.0
        std  = image[fg].std()  if fg.any() else 1.0
        image = (image - mean) / (std + 1e-8)
        return {"image": image, "label": sample["label"]}


class CreateOnehotLabel:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def __call__(self, sample: dict) -> dict:
        label = sample["label"]
        onehot = np.zeros((self.num_classes, *label.shape), dtype=np.float32)
        for i in range(self.num_classes):
            onehot[i] = (label == i).astype(np.float32)
        return {**sample, "onehot_label": onehot}


class ToTensor:
    """Convert 3-D sample arrays to torch tensors."""

    def __call__(self, sample: dict) -> dict:
        image = sample["image"].reshape(1, *sample["image"].shape).astype(np.float32)
        label = sample["label"].astype(np.int16)
        out = {
            "image": torch.from_numpy(image),
            "label": torch.from_numpy(label).long(),
        }
        if "onehot_label" in sample:
            out["onehot_label"] = torch.from_numpy(sample["onehot_label"]).long()
        return out


# ─── Batch samplers ───────────────────────────────────────────────────────────

class TwoStreamBatchSampler(Sampler):
    """Interleave labelled (primary) and unlabelled (secondary) indices."""

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter   = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            p + s
            for p, s in zip(
                grouper(primary_iter,   self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self) -> int:
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def _shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(_shuffles())


def grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)
