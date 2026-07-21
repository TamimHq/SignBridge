"""
Train the ASL sign classifier (same Bi-LSTM as BdSL) and export it for the server.

Small dataset -> trains in a couple of minutes on CPU, no GPU needed.

Usage:
  py train_asl.py --data "processed/asl_training" --server_models "../server/models"
"""

import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

FEATURE_DIM = 144
TARGET_FRAMES = 30
N_POINTS = FEATURE_DIM // 3


# ── Model (identical architecture to the BdSL model) ─────────────────────────
class BiLSTMClassifier(nn.Module):
    def __init__(self, feature_dim=FEATURE_DIM, hidden=160, num_layers=2,
                 num_classes=20, dropout=0.3):
        super().__init__()
        self.input_norm = nn.LayerNorm(feature_dim)
        self.lstm = nn.LSTM(
            input_size=feature_dim, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.norm = nn.LayerNorm(hidden * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        x = self.input_norm(x)
        out, _ = self.lstm(x)
        pooled = self.dropout(self.norm(out.mean(dim=1)))
        return self.fc(pooled)


# ── Augmentation (same as BdSL training) ─────────────────────────────────────
def aug_mirror(k):
    k = k.copy()
    for i in range(N_POINTS):
        k[:, i * 3] = -k[:, i * 3]
    lh = k[:, 18:81].copy()
    rh = k[:, 81:144].copy()
    k[:, 18:81] = rh
    k[:, 81:144] = lh
    return k


def aug_scale(k, lo=0.9, hi=1.1):
    s = np.random.uniform(lo, hi)
    k = k.copy()
    for i in range(N_POINTS):
        k[:, i * 3] *= s
        k[:, i * 3 + 1] *= s
    return k


def aug_time_warp(k):
    T = len(k)
    new_T = max(5, int(T * np.random.uniform(0.8, 1.2)))
    k2 = k[np.linspace(0, T - 1, new_T).astype(int)]
    return k2[np.linspace(0, new_T - 1, T).astype(int)]


def aug_jitter(k, sigma=0.01):
    return k + np.random.normal(0, sigma, k.shape).astype(np.float32)


class SignDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X, self.y, self.augment = X, y, augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        k = self.X[i].copy()
        if self.augment:
            if np.random.rand() < 0.5:
                k = aug_mirror(k)
            if np.random.rand() < 0.4:
                k = aug_scale(k)
            if np.random.rand() < 0.4:
                k = aug_time_warp(k)
            if np.random.rand() < 0.25:
                k = aug_jitter(k)
        return torch.tensor(k, dtype=torch.float32), int(self.y[i])


# ── Stratified split ─────────────────────────────────────────────────────────
def stratified_split(X, y, test_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    tr_idx, te_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_frac)))
        n_test = min(n_test, len(idx) - 1)   # always leave >=1 for training
        te_idx.extend(idx[:n_test])
        tr_idx.extend(idx[n_test:])
    return np.array(tr_idx), np.array(te_idx)


def main(data_dir, server_models, epochs, hidden):
    X = np.load(os.path.join(data_dir, "X.npy"))
    y = np.load(os.path.join(data_dir, "y.npy"))
    with open(os.path.join(data_dir, "classes.json"), encoding="utf-8") as f:
        idx_to_word = {int(k): v for k, v in json.load(f).items()}

    num_classes = len(idx_to_word)
    print(f"\n=== Training ASL model ===")
    print(f"Samples: {len(X)}   Classes: {num_classes}")

    tr, te = stratified_split(X, y)
    print(f"Train: {len(tr)}   Test: {len(te)}")

    train_loader = DataLoader(SignDataset(X[tr], y[tr], augment=True),
                              batch_size=16, shuffle=True)
    test_loader = DataLoader(SignDataset(X[te], y[te], augment=False),
                             batch_size=16)

    device = torch.device("cpu")
    model = BiLSTMClassifier(hidden=hidden, num_classes=num_classes).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="max", factor=0.5, patience=8)

    def evaluate():
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for kps, labels in test_loader:
                pred = model(kps).argmax(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        return correct / max(total, 1)

    best_acc, best_state = 0.0, None
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for kps, labels in train_loader:
            optim.zero_grad()
            loss = criterion(model(kps), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tot += loss.item()
        acc = evaluate()
        sched.step(acc)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"Epoch {ep+1:3d}/{epochs} | loss {tot/len(train_loader):.3f} "
                  f"| acc {acc*100:.1f}% | best {best_acc*100:.1f}%")

    print(f"\n✓ Best test accuracy: {best_acc*100:.1f}%")

    # ── Export (CPU-traced, like the BdSL fix) ──
    model.load_state_dict(best_state)
    model.eval().cpu()

    os.makedirs(server_models, exist_ok=True)
    example = torch.randn(1, TARGET_FRAMES, FEATURE_DIM)
    traced = torch.jit.trace(model, example)

    model_path = os.path.join(server_models, "asl_bilstm_scripted.pt")
    labels_path = os.path.join(server_models, "asl_idx_to_gloss.json")
    traced.save(model_path)
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(idx_to_word, f, ensure_ascii=False, indent=2)

    print(f"\nExported for the server:")
    print(f"  {model_path}")
    print(f"  {labels_path}")
    print("\nRestart the server — it should now report  ASL model : ✓")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="processed/asl_training")
    ap.add_argument("--server_models", default="../server/models")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--hidden", type=int, default=160)
    a = ap.parse_args()
    main(a.data, a.server_models, a.epochs, a.hidden)