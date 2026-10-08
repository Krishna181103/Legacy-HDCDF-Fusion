from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
EVAL_RESULTS_DIR = ROOT / "evaluation" / "results"


def write(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    stat_path = EVAL_RESULTS_DIR / "BEST_HSI_HYBRID_CENTER_statistical_validation.json"
    stats = json.loads(stat_path.read_text(encoding="utf-8")) if stat_path.exists() else {}
    ci = stats.get("bootstrap_confidence_intervals", {})

    write(
        RESULTS_DIR / "loss_ablation.md",
        """
# Loss Function Ablation

No prior complete logs for the three requested loss variants were found in the repository.
The training script now exposes `--loss {ce_smooth,focal,multitask}` while preserving the
default HSI_Hybrid architecture and hyperparameters for supported single-output losses.

| Loss | Status | Accuracy | Macro F1 | Notes |
|---|---|---:|---:|---|
| Label-smoothed CE | Existing best stored run | 0.8774 | 0.8747 | Original best result, validation split only |
| Focal loss | Not rerun in current environment | N/A | N/A | TensorFlow/scikit-learn/tifffile/OpenCV unavailable locally |
| Multi-task loss | Not implemented in current single-output model | N/A | N/A | Requires adding regression head and regression-target pipeline |

Recommended rerun command after restoring the ML environment:

```bash
python training/train_best_hsi_model.py --loss focal --run-tag hsi_hybrid_full_focal --skip-submission
```
""",
    )

    write(
        RESULTS_DIR / "branch_ablation.md",
        """
# Multi-Branch Kernel Ablation

The training script now supports the requested single-branch variants without changing downstream
residual, SE, reshape, or classification layers.

| Variant | Branch kernels | Status | Accuracy | Macro F1 |
|---|---|---|---:|---:|
| Full HSI_Hybrid | (3,3,3), (3,3,5), (1,1,7) | Existing best stored run | 0.8774 | 0.8747 |
| k333 only | (3,3,3) | Script-ready, not rerun locally | N/A | N/A |
| k335 only | (3,3,5) | Script-ready, not rerun locally | N/A | N/A |
| k117 only | (1,1,7) | Script-ready, not rerun locally | N/A | N/A |

Recommended reduced-budget commands:

```bash
python training/train_best_hsi_model.py --branch-variant k333 --run-tag hsi_hybrid_k333_e20 --epoch-budget 20 --skip-submission
python training/train_best_hsi_model.py --branch-variant k335 --run-tag hsi_hybrid_k335_e20 --epoch-budget 20 --skip-submission
python training/train_best_hsi_model.py --branch-variant k117 --run-tag hsi_hybrid_k117_e20 --epoch-budget 20 --skip-submission
```
""",
    )

    write(
        RESULTS_DIR / "test_set_metrics.md",
        """
# Held-Out Test-Set Metrics

A deterministic 70/15/15 stratified split has been generated in `results/split_train_val_test.json`.
The model has not been retrained on this split in the current runtime because TensorFlow,
scikit-learn, tifffile, and OpenCV are unavailable.

| Evaluation | Split | Accuracy | Macro F1 | Notes |
|---|---|---:|---:|---|
| Original stored result | old 80/20 validation split | 0.8774 | 0.8747 | Optimistic because validation was used for model selection |
| New held-out test result | new 15% test split | N/A | N/A | Requires retraining on the new train split and selecting on new val split |
""",
    )

    write(
        RESULTS_DIR / "multiseed_results.md",
        """
# Multi-Seed Results

Multi-seed retraining has not been executed in the current environment because the ML dependencies
required by the project are unavailable. The split itself is fixed in `results/split_train_val_test.json`;
future runs should vary only initialization and data shuffling seed.

| Seed | Held-out test accuracy | Held-out test macro F1 | Status |
|---:|---:|---:|---|
| 11 | N/A | N/A | Not run |
| 22 | N/A | N/A | Not run |
| 33 | N/A | N/A | Not run |

Report mean plus/minus standard deviation after these runs complete.
""",
    )

    write(
        RESULTS_DIR / "latency_test.md",
        """
# Flask Latency Test

The Flask app could not be executed in the current runtime because Flask, TensorFlow, tifffile,
OpenCV, and scikit-learn are unavailable. The app code has been hardened with server-side upload
size limits and safer filename handling.

Run the latency script after restoring the ML environment:

```bash
python evaluation/web_sanity_checks.py --latency --requests 50
```
""",
    )

    write(
        RESULTS_DIR / "robustness_test.md",
        """
# Flask Robustness Test

Static review found two issues and patched them:

| Check | Result |
|---|---|
| Server-side file size limit | Added `MAX_CONTENT_LENGTH` to Flask config |
| Path traversal through uploaded filename | Patched with `secure_filename` before display/logging |
| Empty upload | Already handled with a clean error message |
| Unsupported extension | Already handled with a clean error message |

Runtime malformed-file tests require the Flask/TensorFlow environment to be restored.
""",
    )

    acc = ci.get("accuracy", {})
    f1 = ci.get("macro_f1", {})
    write(
        RESULTS_DIR / "gap_closure_summary.md",
        f"""
# Gap Closure Summary

| Task | What was done | Output | Remaining work |
|---|---|---|---|
| 1. Loss ablation | Added `--loss` support for label-smoothed CE and focal loss; documented multi-task as requiring a model-head change | `results/loss_ablation.md` | Run focal and any true multi-task variant in a restored ML environment |
| 2. Train/val/test split | Generated deterministic stratified 70/15/15 split | `results/split_train_val_test.json`, `results/split_train_val_test.md` | Retrain HSI_Hybrid and report held-out test metrics |
| 3. Branch ablation | Added single-branch variants `k333`, `k335`, and `k117` to the training script | `results/branch_ablation.md` | Run each variant with identical epoch budget |
| 4. PCA explained variance | Existing PCA loading remains blocked by missing scikit-learn; NumPy fallback script still requires TIFF loading support | Existing limitation retained | Restore tifffile/scikit-learn or run on project ML environment |
| 5. Multi-seed variance | Added summary template tied to the new fixed split | `results/multiseed_results.md` | Run 3-5 seeds on held-out split |
| 6. Web checks | Hardened Flask upload handling and documented static robustness checks | `results/latency_test.md`, `results/robustness_test.md` | Run live latency/malformed upload tests once Flask/TensorFlow are available |
| 7. External validation | Not attempted, as requested | Paper limitation | Use only if official labels/leaderboard are available |

Current bootstrap validation from the stored best confusion matrix:

- Accuracy: {acc.get('point', 0):.4f}, 95% CI {acc.get('ci95_low', 0):.4f}-{acc.get('ci95_high', 0):.4f}
- Macro F1: {f1.get('point', 0):.4f}, 95% CI {f1.get('ci95_low', 0):.4f}-{f1.get('ci95_high', 0):.4f}

Local runtime blocker: TensorFlow, scikit-learn, tifffile, OpenCV, and Flask are not installed in the available Python environment, and the repository `venv` points to a missing Miniforge interpreter.
""",
    )


if __name__ == "__main__":
    main()
