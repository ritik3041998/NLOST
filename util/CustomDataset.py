import os
import glob
import numpy as np
import cv2
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader


class NLOSTCustomDataset(Dataset):
    """
    Dataset for custom NLOST training data.

    Expected folder structure:
        data_root/
        ├── transient/   ← .mat files, variable 'out', shape (64, 64, 1024)
        ├── img/         ← .bmp files, shape (64, 64), binary
        └── vol/         ← .mat files, variable 'vol', shape (1024, 64, 64)

    Files must share the same base name across folders.
    """

    def __init__(self, transient_dir, img_dir, vol_dir, file_list, augment=False):
        self.transient_dir = transient_dir
        self.img_dir       = img_dir
        self.vol_dir       = vol_dir
        self.file_list     = file_list
        self.augment       = augment

    def _load_transient(self, name):
        path = os.path.join(self.transient_dir, name + '.mat')
        data = sio.loadmat(path)['out']          # (64, 64, 1024)
        data = np.asarray(data, dtype=np.float32)
        data = data.transpose(2, 0, 1)            # (1024, 64, 64) — axes: (T, W, H)
        data = np.swapaxes(data, 1, 2)            # (1024, 64, 64) — axes: (T, H, W)
        T, H, W = data.shape
        data = data.reshape(T // 2, 2, H, W).sum(axis=1)   # (512, 64, 64)
        data = data[None]                          # (1, 512, 64, 64)
        dmax = data.max()
        if dmax > 0:
            data = data / dmax
        return torch.from_numpy(np.ascontiguousarray(data))

    def _load_image(self, name):
        path = os.path.join(self.img_dir, name + '.bmp')
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f'Image not found: {path}')
        img = img.astype(np.float32) / 255.0      # (64, 64)
        return torch.from_numpy(img)

    def _load_volume(self, name):
        path = os.path.join(self.vol_dir, name + '.mat')
        data = sio.loadmat(path)['vol']            # (1024, 64, 64) = (D, W, H) in scipy
        data = np.asarray(data, dtype=np.float32)
        data = np.swapaxes(data, 1, 2)            # (D, H, W)
        data = data[None]                           # (1, 1024, 64, 64)
        dmax = data.max()
        if dmax > 0:
            data = data / dmax
        return torch.from_numpy(np.ascontiguousarray(data))

    def _apply_augmentation(self, meas, img_gt, vol_gt):
        # random horizontal flip (H axis)
        if torch.rand(1).item() > 0.5:
            meas   = torch.flip(meas,   dims=[2])   # (1,T,H,W) → flip H
            img_gt = torch.flip(img_gt, dims=[0])   # (H,W)     → flip H
            vol_gt = torch.flip(vol_gt, dims=[2])   # (1,D,H,W) → flip H

        # random vertical flip (W axis)
        if torch.rand(1).item() > 0.5:
            meas   = torch.flip(meas,   dims=[3])   # flip W
            img_gt = torch.flip(img_gt, dims=[1])   # flip W
            vol_gt = torch.flip(vol_gt, dims=[3])   # flip W

        # random 90° rotation (0, 90, 180, 270) in H-W plane
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            meas   = torch.rot90(meas,   k, dims=[2, 3])
            img_gt = torch.rot90(img_gt, k, dims=[0, 1])
            vol_gt = torch.rot90(vol_gt, k, dims=[2, 3])

        return meas, img_gt, vol_gt

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        name   = self.file_list[idx]
        meas   = self._load_transient(name)   # (1, 512, 64, 64)
        img_gt = self._load_image(name)        # (64, 64)
        vol_gt = self._load_volume(name)       # (1, 1024, 64, 64)

        if self.augment:
            meas, img_gt, vol_gt = self._apply_augmentation(meas, img_gt, vol_gt)

        return {
            'ds_meas': meas,
            'img_gt':  img_gt,
            'vol_gt':  vol_gt,
        }


def _get_file_list(transient_dir):
    paths = sorted(glob.glob(os.path.join(transient_dir, '*.mat')))
    if len(paths) == 0:
        raise FileNotFoundError(f'No .mat files found in: {transient_dir}')
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]


def build_dataloaders(data_root, batch_size=2, num_workers=4,
                      train_ratio=0.8, seed=42):
    transient_dir = os.path.join(data_root, 'transient')
    img_dir       = os.path.join(data_root, 'img')
    vol_dir       = os.path.join(data_root, 'vol')

    file_list = _get_file_list(transient_dir)
    n_total   = len(file_list)
    n_train   = int(n_total * train_ratio)

    rng     = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    train_files = [file_list[i] for i in indices[:n_train]]
    val_files   = [file_list[i] for i in indices[n_train:]]

    print(f'Total samples : {n_total}')
    print(f'Train samples : {len(train_files)} (with augmentation)')
    print(f'Val samples   : {len(val_files)}')

    # augment=True only for training
    train_dataset = NLOSTCustomDataset(transient_dir, img_dir, vol_dir, train_files, augment=True)
    val_dataset   = NLOSTCustomDataset(transient_dir, img_dir, vol_dir, val_files,   augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
