# main.py

import os
import math
import random
import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from datapipe import build_dataset, get_dataset
from Net import PCDA


def set_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SUBJECTS = 15
EPOCHS = 200
NUM_CLASSES = 3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigmas):
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True).t()
    dist2 = torch.clamp(x2 + y2 - 2.0 * (x @ y.t()), min=0.0)

    K = 0.0
    for s in sigmas:
        gamma = 1.0 / (2.0 * (s ** 2))
        K = K + torch.exp(-gamma * dist2)
    return K


def _mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigmas=(1.0, 2.0, 4.0, 8.0)):
    N, M = x.size(0), y.size(0)
    if N < 2 or M < 2:
        return x.new_tensor(0.0)

    Kxx = _rbf_kernel(x, x, sigmas)
    Kyy = _rbf_kernel(y, y, sigmas)
    Kxy = _rbf_kernel(x, y, sigmas)

    Kxx = Kxx - torch.diag(torch.diag(Kxx))
    Kyy = Kyy - torch.diag(torch.diag(Kyy))
    return Kxx.sum() / (N * (N - 1)) + Kyy.sum() / (M * (M - 1)) - 2.0 * Kxy.mean()


def conditional_mmd(zs: torch.Tensor, zt: torch.Tensor,
                    ys: torch.Tensor, yt_hat: torch.Tensor,
                    num_classes: int,
                    sigmas=(1.0, 2.0, 4.0, 8.0),
                    min_per_class: int = 2):
    """Class-conditional MMD on flattened channel features: zs/zt must be [B,62*D]."""
    loss = zs.new_tensor(0.0)
    wsum = zs.new_tensor(0.0)

    for c in range(num_classes):
        idx_s = (ys == c)
        idx_t = (yt_hat == c)
        ns, nt = int(idx_s.sum()), int(idx_t.sum())
        if ns < min_per_class or nt < min_per_class:
            continue

        mmd_c = _mmd_rbf(zs[idx_s], zt[idx_t], sigmas=sigmas)
        w = zs.new_tensor(float(min(ns, nt)))
        loss = loss + w * mmd_c
        wsum = wsum + w

    return zs.new_tensor(0.0) if wsum.item() == 0 else loss / (wsum + 1e-8)


def dann_alpha(p: float) -> float:
    """Original DANN schedule: alpha = 2/(1+exp(-10p)) - 1."""
    return float(2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


def train_one_epoch(model, train_loader, target_loader,
                    cls_crit, domain_crit, optimizer,
                    epoch_idx: int, total_epochs: int,
                    lambda1=0.1, lambda2=1.0, lambda3=1.0, lambda4=0.1):
    model.train()
    total_loss_sum = 0.0
    domain_loss_sum = 0.0

    target_iter = itertools.cycle(target_loader)
    num_steps = len(train_loader)

    for step_idx, source_data in enumerate(train_loader):
        target_data = next(target_iter)

        source_data = source_data.to(DEVICE)
        target_data = target_data.to(DEVICE)
        optimizer.zero_grad()

        p = (epoch_idx * num_steps + step_idx) / float(total_epochs * num_steps)
        model.set_grl_alpha(dann_alpha(p))

        # Source forward
        ys = torch.argmax(source_data.y.view(-1, NUM_CLASSES), dim=1)
        s_logits, s_pred, s_dom, s_aux = model(source_data.x, source_data.batch)
        loss_cls = cls_crit(s_logits, ys)

        # Target forward (pseudo-labels from current classifier)
        t_logits, t_pred, t_dom, t_aux = model(target_data.x, target_data.batch)
        yt_hat = torch.argmax(t_pred.detach(), dim=1)

        # Domain loss (binary: source=0, target=1)
        src_lbl = torch.zeros(source_data.num_graphs, device=DEVICE)
        tgt_lbl = torch.ones(target_data.num_graphs, device=DEVICE)
        dom_pred = torch.cat([s_dom[:source_data.num_graphs], t_dom[:target_data.num_graphs]], dim=0)
        dom_lbl = torch.cat([src_lbl, tgt_lbl], dim=0)
        loss_domain = domain_crit(dom_pred, dom_lbl)

        # Recon + KL 
        loss_recon = F.mse_loss(s_aux["recon_tokens"], s_aux["cnn_tokens"]) + \
                     F.mse_loss(t_aux["recon_tokens"], t_aux["cnn_tokens"])
        loss_kl = s_aux["vq_loss"] + t_aux["vq_loss"]

        # CMMD 
        loss_cmmd = conditional_mmd(s_aux["quant_flat"], t_aux["quant_flat"], ys, yt_hat, num_classes=NUM_CLASSES)

        loss = loss_cls + lambda1 * loss_domain + lambda2 * loss_kl + lambda3 * loss_recon + lambda4 * loss_cmmd
        loss.backward()
        optimizer.step()

        total_loss_sum += loss.item() * source_data.num_graphs
        domain_loss_sum += loss_domain.item() * dom_lbl.size(0)

    return total_loss_sum / len(train_loader.dataset), domain_loss_sum / len(train_loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, labels = [], []

    for data in loader:
        label_oh = data.y.view(-1, NUM_CLASSES)
        data = data.to(DEVICE)

        _, pred, _, _ = model(data.x, data.batch)
        preds.append(pred.detach().cpu().numpy())
        labels.append(label_oh.cpu().numpy())

    preds = np.vstack(preds)
    labels = np.vstack(labels)

    try:
        auc = roc_auc_score(labels, preds, average="macro")
    except Exception:
        auc = float("nan")

    y_true = np.argmax(labels, axis=1)
    y_pred = np.argmax(preds, axis=1)
    return auc, accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")


def main():
    set_random_seed(42)
    os.makedirs("./result", exist_ok=True)

    # Auto-increment result file name.
    version = 1
    while os.path.exists(f"./result/Q3_{version}.csv"):
        version += 1
    out_csv = f"./result/Q3_{version}.csv"
    pd.DataFrame().to_csv(out_csv, index=False)

    build_dataset(SUBJECTS)

    cls_crit = torch.nn.CrossEntropyLoss()
    domain_crit = torch.nn.BCELoss()

    # Loss weights
    lambda1 = 0.1    # domain
    lambda2 = 0.001  # KL
    lambda3 = 1.0    # recon
    lambda4 = 0.1    # CMMD

    result_rows = []
    best_accs = []

    for cv_n in range(SUBJECTS):
        best_acc = 0.0
        best_epoch = 0

        train_dataset, target_dataset = get_dataset(SUBJECTS, cv_n)
        test_dataset = target_dataset  # same split; different loader configs

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        target_loader = DataLoader(target_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        model = PCDA(classes=NUM_CLASSES).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for epoch in range(EPOCHS):
            loss, dom_loss = train_one_epoch(
                model, train_loader, target_loader,
                cls_crit, domain_crit, optimizer,
                epoch_idx=epoch, total_epochs=EPOCHS,
                lambda1=lambda1, lambda2=lambda2, lambda3=lambda3, lambda4=lambda4,
            )
            val_auc, val_acc, val_f1 = evaluate(model, test_loader)

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch + 1

            print(
                f"V{cv_n:01d},EP{epoch + 1:03d},"
                f"domain_loss:{dom_loss:.4f},loss:{loss:.4f},"
                f"VAUC:{val_auc:.4f},Vacc:{val_acc:.4f}|BestVacc:{best_acc:.4f}(EP{best_epoch:03d})"
            )

        best_accs.append(best_acc)
        result_rows.append([cv_n, best_epoch, best_acc])
        pd.DataFrame(result_rows, columns=["Subject", "Best_Epoch", "Best_Vacc"]).to_csv(out_csv, index=False)

    print("\n=== Final Results ===")
    print(f"MeanVacc:{np.mean(best_accs):.4f}±{np.std(best_accs):.4f}")
    for subj, acc in enumerate(best_accs):
        print(f"Subject{subj:02d}:{acc:.4f}")


if __name__ == "__main__":
    main()
