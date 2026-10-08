from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "datasets" / "HyperLeaf2024" / "train.csv"
RESULTS_DIR = ROOT / "results"
CLASS_COLUMNS = ["Heerup", "Kvium", "Rembrandt", "Sheriff"]


def stratified_split(df: pd.DataFrame, seed: int = 42) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["label_idx"] = np.argmax(df[CLASS_COLUMNS].values, axis=1)

    split_ids = {"train": [], "val": [], "test": []}
    for _, group in df.groupby("label_idx"):
        ids = group["ImageId"].astype(int).to_numpy()
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.15))
        split_ids["train"].extend(ids[:n_train].tolist())
        split_ids["val"].extend(ids[n_train : n_train + n_val].tolist())
        split_ids["test"].extend(ids[n_train + n_val :].tolist())

    for key in split_ids:
        split_ids[key] = sorted(int(v) for v in split_ids[key])
    return split_ids


def class_counts(df: pd.DataFrame, ids: list[int]) -> dict[str, int]:
    part = df[df["ImageId"].astype(int).isin(ids)]
    return {name: int(part[name].sum()) for name in CLASS_COLUMNS}


def write_markdown(df: pd.DataFrame, split_ids: dict[str, list[int]], path: Path) -> None:
    lines = [
        "# Train/Validation/Test Split",
        "",
        "A deterministic 70/15/15 stratified split was generated from the 1,590 labeled HyperLeaf2024 training rows.",
        "",
        "| Split | Samples | Heerup | Kvium | Rembrandt | Sheriff |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ["train", "val", "test"]:
        counts = class_counts(df, split_ids[split])
        lines.append(
            f"| {split} | {len(split_ids[split])} | {counts['Heerup']} | {counts['Kvium']} | "
            f"{counts['Rembrandt']} | {counts['Sheriff']} |"
        )
    lines.extend(
        [
            "",
            "The split file stores ImageId values rather than row positions so it can be reused safely if CSV row order changes.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(TRAIN_CSV)
    split_ids = stratified_split(df, seed=42)
    payload = {
        "dataset": "HyperLeaf2024 local labeled train.csv",
        "random_state": 42,
        "split_policy": "stratified 70/15/15 by cultivar label",
        "class_columns": CLASS_COLUMNS,
        "splits": split_ids,
        "counts": {
            split: {
                "samples": len(ids),
                "class_counts": class_counts(df, ids),
            }
            for split, ids in split_ids.items()
        },
    }
    (RESULTS_DIR / "split_train_val_test.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(df, split_ids, RESULTS_DIR / "split_train_val_test.md")
    print(json.dumps(payload["counts"], indent=2))


if __name__ == "__main__":
    main()
