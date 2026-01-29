# EmoVAE
This repository provides a PyTorch/PyG implementation of a cross-dataset unsupervised domain adaptation (UDA) framework for EEG-based emotion recognition.  
Source domain: **SEED** (labeled). Target domain: **SEED-VII** (unlabeled during training, labels used only for evaluation).

## Repository Structure
```text
.
├── SEED/
│   ├── datapipe.py
│   ├── main.py
│   └── Net.py
├── SEED_to_SEED-VII/
│   ├── datapipe.py
│   ├── main.py
│   └── Net.py
└── README.md
```

## Requirements
- Python 3.9+ (recommended)
- PyTorch (CUDA optional)
- PyTorch Geometric (PyG)
- NumPy, SciPy, scikit-learn, pandas, tqdm

Example installation (adjust to your CUDA/PyTorch version):

```bash
pip install numpy scipy pandas tqdm scikit-learn
pip install torch
pip install torch-geometric
