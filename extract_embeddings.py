#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_embeddings.py
整理 / 验证 plm/structure .pt 特征文件为训练期望格式（2D float tensor）。
用法示例:
  python extract_embeddings.py --root "E:/多位点/ProSIA" --proteins "E:/多位点/ProSIA/data/all_2811_proteins.csv"
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
import torch

def parse_args():
    p = argparse.ArgumentParser(description="Collect/validate/copy ESM/ProtT5/Structure .pt embeddings.")
    p.add_argument("--root", default="E:/多位点/ProSIA", help="工作根目录（默认 E:/多位点/ProSIA）")
    p.add_argument("--proteins", default=None, help="包含蛋白名的 CSV 或文本文件（默认尝试 data/all_2811_proteins.csv）")
    p.add_argument("--modalities", nargs="+", default=["esm","prott5","structure"], choices=["esm","prott5","structure"])
    p.add_argument("--src-esm", default=None, help="源 ESM embedding 目录（默认 root/plm_embeddings/esm2_embedding）")
    p.add_argument("--src-prott5", default=None, help="源 ProtT5 embedding 目录（默认 root/plm_embeddings/prott5_embedding）")
    p.add_argument("--src-structure", default=None, help="源 structure embedding 目录（默认 root/structure/embedding）")
    p.add_argument("--out-esm", default=None, help="输出 ESM 目录（默认 src-esm）")
    p.add_argument("--out-prott5", default=None, help="输出 ProtT5 目录（默认 src-prott5）")
    p.add_argument("--out-structure", default=None, help="输出 structure 目录（默认 src-structure）")
    p.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出文件")
    return p.parse_args()

def safe_torch_load(path: Path) -> torch.Tensor:
    try:
        x = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        x = torch.load(path, map_location="cpu")
    if isinstance(x, dict) and "embedding" in x:
        x = x["embedding"]
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"{path}: not a tensor (type={type(x)})")
    if x.ndim == 3 and x.shape[0] == 1:
        x = x.squeeze(0)
    if x.ndim != 2:
        raise ValueError(f"{path}: expected 2D tensor, got shape {tuple(x.shape)}")
    return x.float()

def save_tensor(path: Path, tensor: torch.Tensor, overwrite: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return False
    torch.save(tensor, path)
    return True

def read_protein_list(path: Path | None, root: Path) -> list[str]:
    if path is None:
        default = root / "data" / "all_2811_proteins.csv"
        if default.exists():
            path = default
        else:
            raise FileNotFoundError("未提供 --proteins，且未找到默认文件 data/all_2811_proteins.csv")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Proteins file not found: {path}")
    try:
        df = pd.read_csv(path)
        if "protein" in df.columns:
            return df["protein"].astype(str).tolist()
        first_col = df.columns[0]
        return df[first_col].astype(str).tolist()
    except Exception:
        with path.open("r", encoding="utf-8") as f:
            return [line.strip().split()[0] for line in f if line.strip()]

def main():
    args = parse_args()
    root = Path(args.root)
    src = {
        "esm": Path(args.src_esm) if args.src_esm else root / "plm_embeddings" / "esm2_embedding",
        "prott5": Path(args.src_prott5) if args.src_prott5 else root / "plm_embeddings" / "prott5_embedding",
        "structure": Path(args.src_structure) if args.src_structure else root / "structure" / "embedding",
    }
    out = {
        "esm": Path(args.out_esm) if args.out_esm else src["esm"],
        "prott5": Path(args.out_prott5) if args.out_prott5 else src["prott5"],
        "structure": Path(args.out_structure) if args.out_structure else src["structure"],
    }

    proteins = read_protein_list(args.proteins, root)
    total = {m:0 for m in args.modalities}
    copied = {m:0 for m in args.modalities}
    skipped = {m:0 for m in args.modalities}
    failed = {m:0 for m in args.modalities}

    for prot in proteins:
        for m in args.modalities:
            total[m] += 1
            src_path = src[m] / f"{prot}.pt"
            dst_path = out[m] / f"{prot}.pt"
            if not src_path.exists():
                failed[m] += 1
                print(f"[WARN] {m} missing for {prot}: {src_path}", file=sys.stderr)
                continue
            try:
                tensor = safe_torch_load(src_path)
            except Exception as e:
                failed[m] += 1
                print(f"[ERROR] load failed {src_path}: {e}", file=sys.stderr)
                continue
            ok = save_tensor(dst_path, tensor, overwrite=args.overwrite)
            if ok:
                copied[m] += 1
            else:
                skipped[m] += 1

    print("Summary:")
    for m in args.modalities:
        print(f"  {m}: total={total[m]} copied={copied[m]} skipped(existing)={skipped[m]} failed={failed[m]}")

if __name__ == "__main__":
    main()
