# datapipe.py

import os
import glob
from typing import List, Tuple

import numpy as np
import torch
import scipy.io as sio
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset
from torch.utils.data import ConcatDataset

SUBJECTS_SEED = 15
CLASSES = 3
VERSION = 2  # bump to avoid processed-cache collision

SEED_ROOT = r"……/SEED/ExtractedFeatures/"
SEEDVII_ROOT = r"……/SEED-VII/SEED-VII/EEG_features"

SEEDVII_SUBJECTS = 20
SEEDVII_TRIALS = 80

T_FIXED = 265  


# =========================
# SEED-VII label mapping
# =========================
seedvii_labels_7 = [
    3, 0, 5, 1, 4, 4, 1, 5, 0, 3,
    3, 0, 5, 1, 4, 4, 1, 5, 0, 3,
    4, 1, 2, 0, 6, 6, 0, 2, 1, 4,
    4, 1, 2, 0, 6, 6, 0, 2, 1, 4,
    3, 6, 5, 2, 4, 4, 2, 5, 6, 3,
    3, 6, 5, 2, 4, 4, 2, 5, 6, 3,
    5, 1, 2, 6, 3, 3, 6, 2, 1, 5,
    5, 1, 2, 6, 3, 3, 6, 2, 1, 5
]
label_map_7_to_3 = {0: 0, 1: -1, 2: -1, 3: 1, 4: -1, 5: -1, 6: 0}
label_map_3_to_index = {-1: 0, 0: 1, 1: 2}


def map_label7_to_index(label7: int) -> int:
    """Map SEED-VII 7-class labels to {0,1,2} for cross-dataset evaluation."""
    return int(label_map_3_to_index[label_map_7_to_3[int(label7)]])


# =========================
# Utils
# =========================
def normalize(data: np.ndarray) -> np.ndarray:
    """Z-score normalize over the sample dimension."""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / (std + 1e-7)


def pad_62_5_T(x: np.ndarray) -> np.ndarray:
    """x:[62,5,T] -> [62,1325] after padding/cropping to T_FIXED."""
    out = np.zeros((62, 5, T_FIXED), dtype=np.float32)
    useT = min(x.shape[-1], T_FIXED)
    out[:, :, :useT] = x[:, :, :useT]
    return out.reshape(62, -1)


# =========================
# Load SEED-VII (per subject)
# =========================
def load_seedvii_subject(subject_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    mat_files = sorted(glob.glob(os.path.join(SEEDVII_ROOT, "*.mat")))
    mat = sio.loadmat(mat_files[subject_idx - 1], verify_compressed_data_integrity=False)

    X, y = [], []
    for i in range(SEEDVII_TRIALS):
        arr = mat[f"de_LDS_{i + 1}"]          # [T,5,62]
        arr = np.transpose(arr, (2, 1, 0))    # [62,5,T]
        X.append(pad_62_5_T(arr))
        y.append(map_label7_to_index(seedvii_labels_7[i]))

    X = normalize(np.stack(X).astype(np.float32))
    y = np.asarray(y, dtype=np.int64)
    return X, y


# =========================
# Load SEED (per subject)
# =========================
def load_seed_subjects() -> Tuple[List[np.ndarray], List[np.ndarray]]:
    label = sio.loadmat(os.path.join(SEED_ROOT, "label.mat"))["label"].squeeze()  # [15]
    files = sorted(glob.glob(os.path.join(SEED_ROOT, "*")))

    X_list, y_list = [], []
    for sub in range(SUBJECTS_SEED):
        sub_files = files[sub * 3:(sub + 1) * 3]
        sub_X = []

        for f in sub_files:
            mat = sio.loadmat(f, verify_compressed_data_integrity=False)
            keys = [k for k in mat.keys() if "de_movingAve" in k]
            for t in range(15):
                arr = mat[keys[t]].transpose(0, 2, 1)  # [62,5,T]
                pad = np.zeros((62, 5, T_FIXED), dtype=np.float32)
                pad[:, :, :arr.shape[-1]] = arr
                sub_X.append(pad.reshape(62, -1))       # [62,1325]

        X_sub = normalize(np.asarray(sub_X, dtype=np.float32))
        y_sub = np.tile(label, 3).astype(np.int64)      # repeat labels for 3 sessions
        _, y_sub = np.unique(y_sub, return_inverse=True)  # map to 0..C-1

        X_list.append(X_sub)
        y_list.append(y_sub)

    return X_list, y_list


# =========================
# PyG Dataset
# =========================
class EmotionDataset(InMemoryDataset):
    def __init__(self, stage: str, root: str, tag: str, X=None, Y=None):
        self.stage = stage
        self.tag = tag
        self.X = X
        self.Y = Y
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return [f"V_{VERSION}_{self.stage}_{self.tag}.pt"]

    def process(self):
        assert self.X is not None and self.Y is not None
        data_list = []
        for i in tqdm(range(len(self.Y)), desc=f"Building {self.stage}:{self.tag}", leave=False):
            x = torch.tensor(self.X[i], dtype=torch.float32)        # [62,1325]
            y = torch.tensor([int(self.Y[i])], dtype=torch.long)    # [1], graph-level
            data_list.append(Data(x=x, y=y))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# =========================
# Cache builder
# =========================
def build_dataset():
    """Build processed caches if missing."""
    os.makedirs("./processed", exist_ok=True)

    # SEED: cache each subject separately (source pool for selection in main).
    seed_X_list, seed_y_list = None, None
    for s in range(1, SUBJECTS_SEED + 1):
        tag = f"SEED_S{s:02d}"
        path = os.path.join("./processed", f"37_V_{VERSION}_Train_{tag}.pt")
        if os.path.exists(path):
            continue
        if seed_X_list is None:
            seed_X_list, seed_y_list = load_seed_subjects()
        EmotionDataset("Train", "./", tag, seed_X_list[s - 1], seed_y_list[s - 1])

    # SEED-VII: cache each subject.
    for t in range(1, SEEDVII_SUBJECTS + 1):
        tag = f"SEEDVII_T{t:02d}"
        path = os.path.join("./processed", f"37_V_{VERSION}_Test_{tag}.pt")
        if os.path.exists(path):
            continue
        Xt, yt = load_seedvii_subject(t)
        EmotionDataset("Test", "./", tag, Xt, yt)


def get_seed_subject_dataset(subject_idx: int) -> EmotionDataset:
    """subject_idx: 1..15"""
    tag = f"SEED_S{subject_idx:02d}"
    return EmotionDataset("Train", "./", tag)


def get_seedvii_subject_dataset(subject_idx: int) -> EmotionDataset:
    """subject_idx: 1..20"""
    tag = f"SEEDVII_T{subject_idx:02d}"
    return EmotionDataset("Test", "./", tag)


def get_uda_datasets(target_test_subject: int):
    seed_subjects = [get_seed_subject_dataset(s) for s in range(1, SUBJECTS_SEED + 1)]

    tgt_train_parts = [
        get_seedvii_subject_dataset(t)
        for t in range(1, SEEDVII_SUBJECTS + 1)
        if t != target_test_subject
    ]
    target_train = ConcatDataset(tgt_train_parts)
    target_test = get_seedvii_subject_dataset(target_test_subject)
    return seed_subjects, target_train, target_test
