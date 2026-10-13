import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import reader
from preprocess.mpii_norm import gaze_to_3d

ROOT = Path.cwd()
DATA_ROOT = ROOT / "data" / "EyeBased" / "MPIIGaze"
LABEL_ROOT = DATA_ROOT / "Label"
IMAGE_ROOT = DATA_ROOT / "Image"
PERSONS = [f"p{k:02d}" for k in range(15)]


def angular_error(pred, gt):
    a, b = gaze_to_3d(pred), gaze_to_3d(gt)
    cos = np.sum(a * b, axis=-1) / (
        np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    )
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


class GazeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, eye, head_pose):
        feat = self.cnn(eye).flatten(1)
        fused = torch.cat([feat, head_pose], dim=1)
        return self.head(fused)


def compute_stats_for_persons(persons):
    label_paths = [str(LABEL_ROOT / f"{p}.label") for p in persons]
    dataset = reader.loader(label_paths, str(IMAGE_ROOT))
    N = len(dataset)
    eyes = []
    heads = []
    for i in range(N):
        info, label = dataset[i]
        eyes.append(info["eye"])
        heads.append(info["head_pose"])
    eyes = np.concatenate(eyes, axis=0)  # (N,1,H,W)
    heads = np.stack(heads, axis=0)
    img_mean = eyes.mean(axis=0, keepdims=True)
    img_std = eyes.std(axis=0, keepdims=True)
    img_std = np.maximum(img_std, 1e-6)
    head_mean = heads.mean(axis=0)
    head_std = heads.std(axis=0)
    head_std = np.maximum(head_std, 1e-6)
    return img_mean, img_std, head_mean, head_std


def make_collate_fn(img_mean, img_std, head_mean, head_std, augment=False):
    def collate_torch(batch):
        info, labels = reader.collate(batch)
        eyes = info["eye"]
        heads = info["head_pose"]
        if augment:
            mask = np.random.rand(len(eyes)) < 0.5
            if mask.any():
                eyes[mask] = eyes[mask, :, :, ::-1]
                labels[mask, 0] = -labels[mask, 0]
                heads[mask, 0] = -heads[mask, 0]
        if img_mean is not None:
            eyes = (eyes - img_mean) / img_std
        if head_mean is not None:
            heads = (heads - head_mean) / head_std
        return (
            {
                "eye": torch.from_numpy(eyes).float(),
                "head_pose": torch.from_numpy(heads).float(),
            },
            torch.from_numpy(labels).float(),
        )

    return collate_torch


def fit_ridge_from_Xy(X, y, alpha=1000.0):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    N = X.shape[0]
    mx = X.mean(0)
    my = y.mean(0)
    Xc = X - mx
    cxx = Xc.T @ Xc
    cxy = Xc.T @ (y - my)
    std = np.sqrt(np.maximum(np.diag(cxx) / N, 1e-12))
    A = cxx / np.outer(std, std) + alpha * np.eye(len(std))
    W = np.linalg.solve(A, cxy / std[:, None]) / std[:, None]
    b = my - mx @ W
    return W, b


def init_batchnorm(model, freeze=False):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)
            if hasattr(m, "running_mean"):
                m.running_mean.zero_()
            if hasattr(m, "running_var"):
                m.running_var.fill_(1.0)
            if freeze:
                m.eval()
                if hasattr(m, "weight") and m.weight is not None:
                    m.weight.requires_grad = False
                if hasattr(m, "bias") and m.bias is not None:
                    m.bias.requires_grad = False


def linear_probe_init(model, train_people, alpha=1000.0, device=None, batch_size=256):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_mean, img_std, head_mean, head_std = compute_stats_for_persons(train_people)
    loader = DataLoader(
        reader.loader(
            [str(LABEL_ROOT / f"{p}.label") for p in train_people], str(IMAGE_ROOT)
        ),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(
            img_mean, img_std, head_mean, head_std, augment=False
        ),
    )
    model.eval()
    feats_list = []
    heads_list = []
    y_list = []
    with torch.no_grad():
        for inputs, targets in loader:
            eye = inputs["eye"].to(device)
            head = inputs["head_pose"].to(device)
            feat = model.cnn(eye).flatten(1).cpu().numpy()
            feats_list.append(feat)
            heads_list.append(head.cpu().numpy())
            y_list.append(targets.numpy())
    X_feat = np.concatenate(feats_list, axis=0)
    X_head = np.concatenate(heads_list, axis=0)
    Y = np.concatenate(y_list, axis=0)
    X = np.concatenate([X_feat, X_head], axis=1)
    with torch.no_grad():
        X_t = torch.from_numpy(X).float().to(device)
        T = model.head[:-1](X_t).cpu().numpy()
    W_last, b_last = fit_ridge_from_Xy(T, Y, alpha=alpha)
    fc_last = model.head[-1]
    with torch.no_grad():
        fc_last.weight.copy_(torch.from_numpy(W_last.T).float())
        fc_last.bias.copy_(torch.from_numpy(b_last).float())
    return W_last, b_last


def train_one_fold(
    test_person,
    epochs=3,
    batch_size=128,
    lr=1e-3,
    ridge_alpha=1000.0,
    apply_bn_init=False,
    apply_linprobe=False,
    freeze_bn=False,
):
    train_people = [p for p in PERSONS if p != test_person]
    img_mean, img_std, head_mean, head_std = compute_stats_for_persons(train_people)
    train_loader = DataLoader(
        reader.loader(
            [str(LABEL_ROOT / f"{p}.label") for p in train_people], str(IMAGE_ROOT)
        ),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(
            img_mean, img_std, head_mean, head_std, augment=True
        ),
    )
    test_loader = DataLoader(
        reader.loader([str(LABEL_ROOT / f"{test_person}.label")], str(IMAGE_ROOT)),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(
            img_mean, img_std, head_mean, head_std, augment=False
        ),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GazeCNN().to(device)
    if apply_bn_init:
        init_batchnorm(model, freeze=freeze_bn)
    if apply_linprobe:
        linear_probe_init(
            model, train_people, alpha=ridge_alpha, device=device, batch_size=batch_size
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n = 0
        for inputs, targets in train_loader:
            eye = inputs["eye"].to(device)
            head = inputs["head_pose"].to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            pred = model(eye, head)
            # cosine loss on 3D vectors
            pred_vec = (
                gaze_to_3d(torch.from_numpy(pred.cpu().detach().numpy()))
                if isinstance(pred, np.ndarray)
                else None
            )
            # easier: use torch conversion
            pred_vec = torch.stack(
                [
                    -torch.cos(pred[:, 1]) * torch.sin(pred[:, 0]),
                    -torch.sin(pred[:, 1]),
                    -torch.cos(pred[:, 1]) * torch.cos(pred[:, 0]),
                ],
                dim=1,
            )
            gt_vec = torch.stack(
                [
                    -torch.cos(targets[:, 1]) * torch.sin(targets[:, 0]),
                    -torch.sin(targets[:, 1]),
                    -torch.cos(targets[:, 1]) * torch.cos(targets[:, 0]),
                ],
                dim=1,
            )
            cos = (pred_vec * gt_vec).sum(dim=1).clamp(-1.0, 1.0)
            loss = torch.mean(1.0 - cos)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * eye.size(0)
            n += eye.size(0)
        print(f"Epoch {epoch}/{epochs} train loss: {running_loss/n:.6f}")
    # eval
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            eye = inputs["eye"].to(device)
            head = inputs["head_pose"].to(device)
            pred = model(eye, head).cpu().numpy()
            preds.append(pred)
            gts.append(targets.numpy())
    pred_arr = np.concatenate(preds, axis=0)
    gt_arr = np.concatenate(gts, axis=0)
    err = angular_error(pred_arr, gt_arr)
    return err.mean()


if __name__ == "__main__":
    test_person = "p00"
    configs = [
        ("base", False, False),
        ("bn_init", True, False),
        ("lp_init", False, True),
        ("both", True, True),
    ]
    results = {}
    for name, bn, lp in configs:
        print("Running config:", name)
        err = train_one_fold(
            test_person,
            epochs=3,
            batch_size=128,
            lr=1e-3,
            apply_bn_init=bn,
            apply_linprobe=lp,
            freeze_bn=False,
        )
        print(f"{name} mean angular err: {err:.2f} deg")
        results[name] = err
    print("\nSummary:")
    for k, v in results.items():
        print(k, v)
