import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import itertools
from scipy import ndimage
import random
from torch.utils.data.sampler import Sampler
from skimage import transform as sk_trans
from scipy.ndimage import rotate, zoom
import pdb
import cv2
import nibabel as nib

class Cyst2D(Dataset):

    def __init__(self, base_dir=None, split='train', transform=None):

        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        self.slice_list = []
        self.split = split

        if split == 'train':
            txt_file = os.path.join(base_dir, 'train.txt')
        elif split == 'val':
            txt_file = os.path.join(base_dir, 'val.txt')

        with open(txt_file, 'r') as f:
            lines = f.readlines()[1:]

        self.sample_list = [item.strip().split(',') for item in lines]

        if split == 'train':
        # convert 3D volumes into slice index list
            for image_path, mask_path in self.sample_list:

                image = nib.load(image_path).get_fdata()

                depth = image.shape[2]

                for z in range(depth):
                    self.slice_list.append((image_path, mask_path, z))

        print("total {} slices".format(len(self.slice_list)))

    def __len__(self):
        if self.split == 'val':
            return len(self.sample_list)
        return len(self.slice_list)

    def __getitem__(self, idx):

        if self.split == 'val':
            image_path, mask_path = self.sample_list[idx]

            image = nib.load(image_path).get_fdata().astype(np.float32)
            label = nib.load(mask_path).get_fdata().astype(np.uint8)
            sample = {'image': image, 'label': label}
            return sample

        image_path, mask_path, z = self.slice_list[idx]

        image = nib.load(image_path).get_fdata()
        label = nib.load(mask_path).get_fdata()

        image = image[:, :, z]
        label = label[:, :, z]

        image = image.astype(np.float32)
        label = label.astype(np.uint8)

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        sample['case'] = image_path

        return sample

def random_rot_flip(image, label):

    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)

    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()

    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):

        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)

        x, y = image.shape

        image = zoom(image, (self.output_size[0] / x,
                             self.output_size[1] / y),
                     order=1)

        label = zoom(label, (self.output_size[0] / x,
                             self.output_size[1] / y),
                     order=0)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        return {'image': image, 'label': label}

class Resize(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        (w, h, d) = image.shape
        label = label.astype(bool)
        image = sk_trans.resize(image, self.output_size, order = 1, mode = 'constant', cval = 0)
        label = sk_trans.resize(label, self.output_size, order = 0)
        
        return {'image': image, 'label': label}
    
class Resize3D(object):
    
    def __init__(self, output_size):
        # output_size should be a tuple (new_w, new_h, new_d)
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # Ensure the image and label are 3D (w, h, d)
        assert len(image.shape) == 3, "Input image should be 3D (w, h, d)"
        assert len(label.shape) == 3, "Input label should be 3D (w, h, d)"
        
        (w, h, d) = image.shape
        label = label.astype(bool)

        # Resize image and label using skimage.transform.resize
        image = sk_trans.resize(image, self.output_size, order=1, mode='constant', cval=0)
        label = sk_trans.resize(label, self.output_size, order=0)  # Keep binary labels
        
        # Assert that the resized label is still binary (0 or 1)
        assert(np.max(label) == 1 and np.min(label) == 0)
        assert(np.unique(label).shape[0] == 2)
        
        return {'image': image, 'label': label}
    
    
class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = random_rot_flip(image, label)

        return {'image': image, 'label': label}

class RandomRot(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = random_rotate(image, label)

        return {'image': image, 'label': label}


# class RandomNoise(object):
#     def __init__(self, mu=0, sigma=0.1):
#         self.mu = mu
#         self.sigma = sigma

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
#         noise = noise + self.mu
#         image = image + noise
#         return {'image': image, 'label': label}
    
class RandomNoise(object):
    def __init__(self, mean=0.0, std=0.05, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            image = sample['image']
            noise = np.random.normal(self.mean, self.std, image.shape)
            image = image + noise
            sample['image'] = np.clip(image, 0, 1)
        return sample


class RandomBlur(object):
    def __init__(self, sigma_range=(0.5, 1.5), p=0.5):
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            image = sample['image']
            sigma = random.uniform(*self.sigma_range)
            ksize = int(2 * round(3 * sigma) + 1)
            image = cv2.GaussianBlur(image, (ksize, ksize), sigma)
            sample['image'] = image
        return sample


class RandomGamma(object):
    def __init__(self, gamma_range=(0.7, 1.5), p=0.5):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            image = sample['image']
            gamma = random.uniform(*self.gamma_range)
            image = np.power(image, gamma)
            sample['image'] = np.clip(image, 0, 1)
        return sample


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        label = sample['label'].astype(np.int16)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(label).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(label).long()}
        
class Normalize(object):

    def __call__(self, sample):

        image, label = sample['image'], sample['label']

        fg_mask = image > 0

        if np.sum(fg_mask) > 0:
            mean = image[fg_mask].mean()
            std = image[fg_mask].std()
            image = (image - mean) / (std + 1e-8)

        return {'image': image, 'label': label}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


class ThreeStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch + primary_batch
            for (primary_batch, secondary_batch, primary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size),
                    grouper(primary_iter, self.primary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
