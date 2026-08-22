#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# Use project-local root (directory containing this script)
ROOT = Path(__file__).resolve().parent


def first_existing_path(*candidates: Path) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


DEFAULT_DATA_DIR = first_existing_path(ROOT / "data", ROOT.parent / "data")
DEFAULT_ESM_DIR = first_existing_path(ROOT / "esm2_embedding", ROOT / "plm_embeddings" / "esm2_embedding", ROOT.parent / "plm_embeddings" / "esm2_embedding")
DEFAULT_PROTT5_DIR = first_existing_path(ROOT / "prott5_embedding", ROOT / "plm_embeddings" / "prott5_embedding", ROOT.parent / "plm_embeddings" / "prott5_embedding")
DEFAULT_STRUCTURE_DIR = first_existing_path(ROOT / "structure_embedding", ROOT / "structure" / "embedding", ROOT.parent / "structure" / "embedding")

DATA_DIR = DEFAULT_DATA_DIR
ESM_DIR = DEFAULT_ESM_DIR
PROTT5_DIR = DEFAULT_PROTT5_DIR
STRUCTURE_DIR = DEFAULT_STRUCTURE_DIR

PTM_TYPES = sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir() and (p / "train_80.csv").exists()])
VARIANT = "mt_esm_prott5_gated_structure"
VARIANTS = [VARIANT]
METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc"]


@dataclass
class Config:
    experiment_name: str = "mt_esm_prott5_custom"
    window_size: int = 31
    esm_dim: int = 1280
    prott5_dim: int = 1024
    structure_dim: int = 112
    cnn_channels: int = 200
    kernel_sizes: tuple[int, ...] = (1, 9, 11)
    prott5_out: int = 256
    structure_out: int = 128
    dropout: float = 0.5
    epochs: int = 30
    patience: int = 6
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    log_every: int = 200
    seed: int = 0
    num_folds: int = 5
    device: str = "cuda"
    amp: bool = True


def config_from_dict(raw: dict) -> Config:
    """Load supported fields from a saved configuration."""
    valid = {field.name for field in fields(Config)}
    return Config(**{key: value for key, value in raw.items() if key in valid})


def load_compatible_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Load weights saved by the current or an earlier implementation."""
    model_keys = set(model.state_dict())
    cleaned = {key: value for key, value in state_dict.items() if key in model_keys}
    incompatible = model.load_state_dict(cleaned, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Incompatible checkpoint: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone ESM-ProtT5 gated-structure lysine PTM model.")
    p.add_argument("--experiment-name", default="mt_esm_prott5_custom")
    p.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    p.add_argument("--ptm", nargs="+", default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--log-every", type=int, default=200, help="Print training progress every N batches; <=0 disables batch logs.")
    p.add_argument("--num-folds", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--esm-dir", type=Path, default=DEFAULT_ESM_DIR)
    p.add_argument("--prott5-dir", type=Path, default=DEFAULT_PROTT5_DIR)
    p.add_argument("--structure-dir", type=Path, default=DEFAULT_STRUCTURE_DIR)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-root", default=str(ROOT / "experiment"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--overwrite-variant", action="store_true")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def safe_torch_load(path: Path) -> torch.Tensor:
    try:
        x = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        x = torch.load(path, map_location="cpu")
    if x.ndim == 3 and x.shape[0] == 1:
        x = x.squeeze(0)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D tensor in {path}, got {tuple(x.shape)}")
    return x.float()


@lru_cache(maxsize=4096)
def load_esm(protein: str) -> torch.Tensor:
    return safe_torch_load(ESM_DIR / f"{protein}.pt")


@lru_cache(maxsize=4096)
def load_prott5(protein: str) -> torch.Tensor:
    return safe_torch_load(PROTT5_DIR / f"{protein}.pt")


@lru_cache(maxsize=4096)
def load_structure(protein: str) -> torch.Tensor:
    return safe_torch_load(STRUCTURE_DIR / f"{protein}.pt")


def slice_window(x: torch.Tensor, pos_1based: int, window: int) -> torch.Tensor:
    half = window // 2
    center = int(pos_1based) - 1
    out = torch.zeros((window, x.shape[1]), dtype=x.dtype)
    src_start = max(center - half, 0)
    src_end = min(center + half + 1, x.shape[0])
    dst_start = src_start - (center - half)
    if src_start < src_end:
        out[dst_start : dst_start + src_end - src_start] = x[src_start:src_end]
    return out


def read_split(ptm: str, fold: int, split: str) -> pd.DataFrame:
    if split == "test":
        path = DATA_DIR / ptm / "test_20.csv"
    else:
        path = DATA_DIR / ptm / "cv_folds" / f"fold_{fold}_{split}.csv"
    df = pd.read_csv(path).reset_index(drop=True)
    df["ptm"] = ptm
    return df


def collect_df(ptms: list[str], fold: int, split: str) -> pd.DataFrame:
    return pd.concat([read_split(ptm, fold, split) for ptm in ptms], ignore_index=True)


class PTMDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, task_to_id: dict[str, int], cfg: Config, variant: str):
        self.df = df.reset_index(drop=True)
        self.task_to_id = task_to_id
        self.cfg = cfg
        self.variant = variant

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        protein = str(row["protein"])
        pos = int(row["Position"])
        item = {
            "esm": slice_window(load_esm(protein), pos, self.cfg.window_size),
            "y": torch.tensor(float(row["y"]), dtype=torch.float32),
            "task_id": torch.tensor(self.task_to_id[str(row["ptm"])], dtype=torch.long),
            "ptm": str(row["ptm"]),
            "protein": protein,
            "Position": pos,
        }
        if "prott5" in self.variant:
            item["prott5"] = slice_window(load_prott5(protein), pos, self.cfg.window_size)
        if "structure" in self.variant:
            item["structure"] = slice_window(load_structure(protein), pos, self.cfg.window_size)
        return item


def collate_fn(batch: list[dict]) -> dict:
    out = {
        "esm": torch.stack([b["esm"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "task_id": torch.stack([b["task_id"] for b in batch]),
        "ptm": [b["ptm"] for b in batch],
        "protein": [b["protein"] for b in batch],
        "Position": [b["Position"] for b in batch],
    }
    if "prott5" in batch[0]:
        out["prott5"] = torch.stack([b["prott5"] for b in batch])
    if "structure" in batch[0]:
        out["structure"] = torch.stack([b["structure"] for b in batch])
    return out


class MultiScaleESMBackbone(nn.Module):
    """
    Core branch:
      ESM2 window [B, 31, 1280]
      CNN(k=1,9,11) -> CNN(k=1,9,11)
      take center lysine representation -> [B, 600]
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.conv1 = nn.ModuleList(
            [nn.Conv1d(cfg.esm_dim, cfg.cnn_channels, k, padding="same") for k in cfg.kernel_sizes]
        )
        out_dim = cfg.cnn_channels * len(cfg.kernel_sizes)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.conv2 = nn.ModuleList([nn.Conv1d(out_dim, cfg.cnn_channels, k, padding="same") for k in cfg.kernel_sizes])
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(cfg.dropout)
        self.output_dim = out_dim

    def forward(self, esm: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        # task_ids remains in the interface because the downstream classifier
        # head is task-specific; the shared ESM2 encoder itself is task-agnostic.
        del task_ids
        x = esm.permute(0, 2, 1)
        x = self.drop(self.relu(self.bn1(torch.cat([conv(x) for conv in self.conv1], dim=1))))
        x = self.relu(self.bn2(torch.cat([conv(x) for conv in self.conv2], dim=1)))
        x = x.permute(0, 2, 1)
        return x[:, self.cfg.window_size // 2, :]


class WindowGRUEncoder(nn.Module):
    """
    Window encoder:
      PLM/structure window [B, 31, D]
      Linear + ReLU + Dropout
      BiGRU
      mean pooling -> [B, out_dim]
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        hidden = max(out_dim, 128)
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2))
        self.gru = nn.GRU(hidden, out_dim // 2, batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x, _ = self.gru(x)
        return x.mean(dim=1)


class CustomPTMModel(nn.Module):
    """
    Standalone mt_esm_prott5_gated_structure model, without importing MTPrompt-PTM code:

    1. ESM branch:
       ESM2 site window -> two-layer multi-scale CNN -> [600].

    2. ProtT5 branch:
       ProtT5 site window -> Linear + BiGRU + mean pooling -> [256].

    3. ESM-ProtT5 gate:
       gate = sigmoid(MLP([ESM, ProtT5]))
       fused = gate * ESM + (1 - gate) * Linear(ProtT5) -> [600].

    4. Structure gate:
       structure window -> Linear + BiGRU + mean pooling -> [128] -> Linear -> [600]
       final = structure_gate * fused + (1 - structure_gate) * projected_structure.

    5. Task-specific binary classifier heads.
    """

    def __init__(self, cfg: Config, n_tasks: int, variant: str):
        super().__init__()
        self.variant = variant
        self.esm = MultiScaleESMBackbone(cfg)
        self.base_dim = self.esm.output_dim
        self.prott5_encoder = None
        self.structure_encoder = None
        fusion_dim = self.base_dim

        if "prott5" in variant:
            self.prott5_encoder = WindowGRUEncoder(cfg.prott5_dim, cfg.prott5_out)
            if variant == "mt_esm_prott5_concat":
                fusion_dim = self.base_dim + cfg.prott5_out
            else:
                self.prott5_to_base = nn.Linear(cfg.prott5_out, self.base_dim)
                self.prott5_gate = nn.Sequential(nn.Linear(self.base_dim + cfg.prott5_out, self.base_dim), nn.Sigmoid())

        if "structure" in variant:
            self.structure_encoder = WindowGRUEncoder(cfg.structure_dim, cfg.structure_out)
            self.structure_to_base = nn.Linear(cfg.structure_out, self.base_dim)
            self.structure_gate = nn.Sequential(nn.Linear(self.base_dim + cfg.structure_out, self.base_dim), nn.Sigmoid())

        self.heads = nn.ModuleList([nn.Linear(fusion_dim, 2) for _ in range(n_tasks)])

    def forward(
        self,
        esm: torch.Tensor,
        task_ids: torch.Tensor,
        prott5: torch.Tensor | None = None,
        structure: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base = self.esm(esm, task_ids)
        if self.variant == "mt_esm_prott5_concat":
            prott5_repr = self.prott5_encoder(prott5)
            fused = torch.cat([base, prott5_repr], dim=1)
        else:
            prott5_repr = self.prott5_encoder(prott5)
            prott5_gate = self.prott5_gate(torch.cat([base, prott5_repr], dim=1))
            fused = prott5_gate * base + (1.0 - prott5_gate) * self.prott5_to_base(prott5_repr)
            if self.variant == "mt_esm_prott5_gated_structure":
                structure_repr = self.structure_encoder(structure)
                structure_gate = self.structure_gate(torch.cat([fused, structure_repr], dim=1))
                fused = structure_gate * fused + (1.0 - structure_gate) * self.structure_to_base(structure_repr)

        logits = []
        for i, task_id in enumerate(task_ids):
            two_class = self.heads[int(task_id)](fused[i])
            logits.append(two_class[1] - two_class[0])
        return torch.stack(logits)


def binary_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict:
    pred = (prob >= 0.5).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, pred),
        "roc_auc": roc_auc_score(y_true, prob) if len(np.unique(y_true)) == 2 else float("nan"),
        "pr_auc": average_precision_score(y_true, prob) if len(np.unique(y_true)) == 2 else float("nan"),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return out


def batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {
        "esm": batch["esm"].to(device, non_blocking=True),
        "y": batch["y"].to(device, non_blocking=True),
        "task_id": batch["task_id"].to(device, non_blocking=True),
    }
    if "prott5" in batch:
        out["prott5"] = batch["prott5"].to(device, non_blocking=True)
    if "structure" in batch:
        out["structure"] = batch["structure"].to(device, non_blocking=True)
    return out


def run_model(model: nn.Module, batch: dict) -> torch.Tensor:
    return model(batch["esm"], batch["task_id"], batch.get("prott5"), batch.get("structure"))


def evaluate(model, loader, criterion, device, amp):
    model.eval()
    losses, rows = [], []
    with torch.no_grad():
        for raw in loader:
            batch = batch_to_device(raw, device)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                logits = run_model(model, batch)
                loss = criterion(logits, batch["y"])
            prob = torch.sigmoid(logits.float()).detach().cpu().numpy()
            labels = batch["y"].detach().cpu().numpy().astype(int)
            losses.append(float(loss.item()) * len(labels))
            rows.extend(
                {
                    "ptm": ptm,
                    "protein": protein,
                    "Position": pos,
                    "y": int(y),
                    "prob": float(p),
                    "pred": int(p >= 0.5),
                }
                for ptm, protein, pos, y, p in zip(raw["ptm"], raw["protein"], raw["Position"], labels, prob)
            )
    pred_df = pd.DataFrame(rows)
    metric_rows = []
    for ptm, group in pred_df.groupby("ptm", sort=False):
        row = {"ptm": ptm}
        row.update(binary_metrics(group["y"].to_numpy(), group["prob"].to_numpy()))
        metric_rows.append(row)
    return float(np.sum(losses) / max(len(pred_df), 1)), pd.DataFrame(metric_rows), pred_df


def train_epoch(model, loader, criterion, optimizer, scaler, device, amp, log_every: int = 0, epoch: int | None = None, fold: int | None = None):
    model.train()
    total, n = 0.0, 0
    num_batches = len(loader)
    for step, raw in enumerate(loader, start=1):
        batch = batch_to_device(raw, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            loss = criterion(run_model(model, batch), batch["y"])
        if amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += float(loss.item()) * batch["y"].numel()
        n += batch["y"].numel()
        if log_every and log_every > 0 and (step == 1 or step % log_every == 0 or step == num_batches):
            prefix = []
            if fold is not None:
                prefix.append(f"fold {fold}")
            if epoch is not None:
                prefix.append(f"epoch {epoch}")
            prefix.append(f"batch {step}/{num_batches}")
            print(
                f"{' '.join(prefix)} train_loss_running={total / max(n, 1):.5f}",
                flush=True,
            )
    return total / max(n, 1)


def make_loader(ds, cfg, shuffle):
    return torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )


def train_fold(variant: str, fold: int, ptms: list[str], cfg: Config, variant_dir: Path, device: torch.device):
    task_to_id = {ptm: i for i, ptm in enumerate(ptms)}
    train_df = collect_df(ptms, fold, "train")
    val_df = collect_df(ptms, fold, "val")
    test_df = collect_df(ptms, fold, "test")
    train_loader = make_loader(PTMDataset(train_df, task_to_id, cfg, variant), cfg, True)
    val_loader = make_loader(PTMDataset(val_df, task_to_id, cfg, variant), cfg, False)
    test_loader = make_loader(PTMDataset(test_df, task_to_id, cfg, variant), cfg, False)

    model = CustomPTMModel(cfg, len(ptms), variant).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    fold_dir = variant_dir / "folds" / f"fold_{fold}"
    (fold_dir / "models").mkdir(parents=True, exist_ok=True)

    best_state, best_score, best_epoch, stale = None, -math.inf, 0, 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, cfg.amp, cfg.log_every, epoch, fold)
        val_loss, val_metrics, _ = evaluate(model, val_loader, criterion, device, cfg.amp)
        mean_mcc = float(pd.to_numeric(val_metrics["mcc"], errors="coerce").mean())
        mean_auc = float(pd.to_numeric(val_metrics["roc_auc"], errors="coerce").mean())
        score = mean_mcc if not math.isnan(mean_mcc) else mean_auc
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mean_mcc": mean_mcc,
                "val_mean_roc_auc": mean_auc,
            }
        )
        print(
            f"{variant} fold {fold} epoch {epoch}: "
            f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"val_mcc={mean_mcc:.4f} val_auc={mean_auc:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
    _, test_metrics, predictions = evaluate(model, test_loader, criterion, device, cfg.amp)
    predictions.to_csv(fold_dir / "test_predictions.csv", index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "variant": variant,
            "ptm_types": ptms,
            "task_to_id": task_to_id,
            "fold": fold,
            "best_epoch": best_epoch,
        },
        fold_dir / "models" / "best_model.pt",
    )
    test_metrics.insert(1, "fold", fold)
    test_metrics.insert(2, "best_epoch", best_epoch)
    return test_metrics


def write_artifacts(all_folds: pd.DataFrame, variant_dir: Path) -> pd.DataFrame:
    all_folds.to_csv(variant_dir / "all_ptm_independent_test_metrics_5fold.csv", index=False, float_format="%.6f")
    summaries = []
    for ptm, group in all_folds.groupby("ptm", sort=False):
        task_dir = variant_dir / "tasks" / ptm
        task_dir.mkdir(parents=True, exist_ok=True)
        group.to_csv(task_dir / f"{ptm}_independent_test_metrics_5fold.csv", index=False, float_format="%.6f")
        summary = group[METRIC_COLUMNS].mean().to_frame().T
        summary.columns = [f"{c}_mean" for c in summary.columns]
        summary.insert(0, "ptm", ptm)
        summary.to_csv(task_dir / f"{ptm}_task_mean_metrics.csv", index=False, float_format="%.6f")
        summaries.append(summary)
    summary_df = pd.concat(summaries, ignore_index=True)
    summary_df.to_csv(variant_dir / "all_ptm_task_mean_metrics.csv", index=False, float_format="%.6f")
    return summary_df


def main() -> None:
    global DATA_DIR, ESM_DIR, PROTT5_DIR, STRUCTURE_DIR, PTM_TYPES

    args = parse_args()
    args.variants = [VARIANT]
    DATA_DIR = args.data_dir.resolve()
    ESM_DIR = args.esm_dir.resolve()
    PROTT5_DIR = args.prott5_dir.resolve()
    STRUCTURE_DIR = args.structure_dir.resolve()
    for label, path in {
        "data_dir": DATA_DIR,
        "esm_dir": ESM_DIR,
        "prott5_dir": PROTT5_DIR,
        "structure_dir": STRUCTURE_DIR,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    PTM_TYPES = sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir() and (p / "train_80.csv").exists()])
    if not PTM_TYPES:
        raise FileNotFoundError(f"No PTM task directories with train_80.csv found in {DATA_DIR}")

    cfg = Config(
        experiment_name=args.experiment_name,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        seed=args.seed,
        num_folds=args.num_folds,
        device=args.device,
        amp=not args.no_amp,
    )
    set_seed(cfg.seed)
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but torch.cuda.is_available() is False.")
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Data dir: {DATA_DIR}", flush=True)
    print(f"ESM2 dir: {ESM_DIR}", flush=True)
    print(f"ProtT5 dir: {PROTT5_DIR}", flush=True)
    print(f"Structure dir: {STRUCTURE_DIR}", flush=True)

    ptms = args.ptm or PTM_TYPES
    exp_root = Path(args.out_root) / args.experiment_name
    if args.overwrite and exp_root.exists():
        raise RuntimeError(
            "This standalone script writes into the existing mt_esm_prott5_custom experiment. "
            "Use --overwrite-variant to rerun only mt_esm_prott5_gated_structure."
        )
    exp_root.mkdir(parents=True, exist_ok=True)
    save_json(
        exp_root / "config.json",
        {
            "config": asdict(cfg),
            "variants": args.variants,
            "ptm_types": ptms,
            "data_dir": str(DATA_DIR),
            "embedding_dirs": {
                "esm2": str(ESM_DIR),
                "prott5": str(PROTT5_DIR),
                "structure": str(STRUCTURE_DIR),
            },
        },
    )

    variant_summaries = []
    for variant in args.variants:
        variant_dir = exp_root / variant
        if args.overwrite_variant and variant_dir.exists():
            keep_names = {
                "run_mt_esm_prott5_gated_structure.py",
                "run_gated_structure.sh",
                "README_STRUCTURE.md",
                "config_gated_structure.json",
            }
            backup = {p.name: p.read_bytes() for p in variant_dir.iterdir() if p.is_file() and p.name in keep_names}
            shutil.rmtree(variant_dir)
            variant_dir.mkdir(parents=True, exist_ok=True)
            for name, content in backup.items():
                (variant_dir / name).write_bytes(content)
        variant_dir.mkdir(parents=True, exist_ok=True)
        fold_frames = []
        for fold in range(1, cfg.num_folds + 1):
            fold_frames.append(train_fold(variant, fold, ptms, cfg, variant_dir, device))
        summary = write_artifacts(pd.concat(fold_frames, ignore_index=True), variant_dir)
        summary.insert(0, "variant", variant)
        variant_summaries.append(summary)

    new_summary = pd.concat(variant_summaries, ignore_index=True)
    summary_path = exp_root / "all_variants_task_mean_metrics.csv"
    if summary_path.exists():
        old = pd.read_csv(summary_path)
        old = old[~old["variant"].isin(args.variants)]
        new_summary = pd.concat([old, new_summary], ignore_index=True)
    new_summary.to_csv(summary_path, index=False, float_format="%.6f")
    print(new_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

