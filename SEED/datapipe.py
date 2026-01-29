# datapipe.py

import os
import glob
import numpy as np
import torch
import scipy.io as sio
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset

SUBJECTS = 15
NUM_CLASSES = 3
VERSION = 1

# Update this path to your local SEED extracted-feature directory.
DATA_ROOT = "……/SEED/ExtractedFeatures/"


def to_categorical(y, num_classes=None, dtype="float32"):
    """Convert integer labels to one-hot vectors."""
    y = np.asarray(y, dtype=np.int64).ravel()
    if num_classes is None:
        num_classes = int(np.max(y) + 1)
    out = np.zeros((y.size, num_classes), dtype=dtype)
    out[np.arange(y.size), y] = 1
    return out


class EmotionDataset(InMemoryDataset):
    """Cached PyG dataset for a single LOSO split."""

    def __init__(self, stage, root, subjects, sub_i, X=None, Y=None, transform=None, pre_transform=None):
        self.stage = stage
        self.subjects = int(subjects)
        self.sub_i = int(sub_i)
        self.X = X
        self.Y = Y
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return [f"./V_{VERSION:.0f}_{self.stage}_CV{self.subjects:.0f}_{self.sub_i:.0f}.dataset"]

    def process(self):
        assert self.X is not None and self.Y is not None, "X/Y must be provided for processing."
        data_list = []
        for i in tqdm(range(self.Y.shape[0]), desc=f"Processing {self.stage} CV{self.sub_i}", leave=False):
            x = torch.tensor(self.X[i], dtype=torch.float32)          # [62,1325]
            y = torch.tensor(self.Y[i], dtype=torch.float32)          # [NUM_CLASSES]
            data_list.append(Data(x=x, y=y))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def normalize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / (std + 1e-7)


def get_data():
    """
    Load SEED DE moving-average features.
    """
    label = sio.loadmat(os.path.join(DATA_ROOT, "label.mat"))["label"]  # typically [1,15]
    files = sorted(glob.glob(os.path.join(DATA_ROOT, "*")))

    sub_mov, sub_label = [], []
    for sub_i in range(SUBJECTS):
        # Each subject has 3 sessions.
        sub_files = files[sub_i * 3: sub_i * 3 + 3]
        session_trials = []

        for f in sub_files:
            mat = sio.loadmat(f, verify_compressed_data_integrity=False)
            keys = list(mat.keys())
            de_keys = [k for k in keys if "de_movingAve" in k]

            trials = []
            for t in range(15):
                # Original layout varies; keep the same transpose/pad logic as the baseline.
                tmp = mat[de_keys[t]].transpose(0, 2, 1)  # [62,5,T] (after transpose)
                T = tmp.shape[-1]
                padded = np.zeros((62, 5, 265), dtype=tmp.dtype)
                padded[:, :, :T] = tmp
                trials.append(padded.reshape(62, -1))     # [62,1325]
            session_trials.append(np.asarray(trials))     # [15,62,1325]

        mov = np.vstack(session_trials)                   # [45,62,1325]
        mov = normalize(mov)
        sub_mov.append(mov)

        # Repeat labels for 3 sessions.
        sub_label.append(np.hstack([label, label, label]).squeeze())

    return np.asarray(sub_mov), np.asarray(sub_label)


def build_dataset(subjects):
    """Build cached LOSO datasets if missing."""
    loaded = False
    mov_coefs = labels = None

    for sub_i in range(subjects):
        train_cache = f"./processed/V_{VERSION:.0f}_Train_CV{subjects:.0f}_{sub_i:.0f}.dataset"
        if os.path.exists(train_cache):
            continue

        if not loaded:
            mov_coefs, labels = get_data()
            loaded = True

        train_idx = list(range(subjects))
        train_idx.remove(sub_i)

        X = mov_coefs[train_idx].reshape(-1, 62, 1325)
        Y = labels[train_idx].reshape(-1)
        testX = mov_coefs[sub_i].reshape(-1, 62, 1325)
        testY = labels[sub_i].reshape(-1)

        _, Y = np.unique(Y, return_inverse=True)
        _, testY = np.unique(testY, return_inverse=True)

        Y = to_categorical(Y, NUM_CLASSES)
        testY = to_categorical(testY, NUM_CLASSES)

        EmotionDataset("Train", "./", subjects, sub_i, X, Y)
        EmotionDataset("Test", "./", subjects, sub_i, testX, testY)


def get_dataset(subjects, sub_i):
    train_dataset = EmotionDataset("Train", "./", subjects, sub_i)
    test_dataset = EmotionDataset("Test", "./", subjects, sub_i)
    return train_dataset, test_dataset
