"""temporal_validation/hp_tuned/analyze_external.py

v4 追加（2026-10-27）: evaluate_external.py の出力（外部データの予測値と指標）から、
論文用の Table 4・ROC 図・較正図（reliability diagram）を作る。

- 指標は音源レベル（クリップ確率の平均）。閾値は開発データからロックしたもの。
- AUC の 95%CI は DeLong 法、感度・特異度などの 95%CI は percentile bootstrap
  （1000 回、utils/external_metrics.py。evaluate_external.py が計算済みのものを読む）。
- Brier score と較正 slope / intercept（logit(p) を説明変数とするロジスティック回帰）を追加。
- 数値は小数第3位で出力する。

画像の種類（--images）:
    noisy … 主解析（evaluate_external_noisy.py → external_noisy/）。既定
    clean … 感度分析（evaluate_external.py → external/。2026-08 の設計）

実行（このフォルダで、評価スクリプトの後に）:
    uv run analyze_external.py --images noisy
    uv run analyze_external.py --images clean
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# SVG/PDF の文字をアウトライン化せずテキストのまま残す（Illustrator / PowerPoint 等で編集できるように）
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from tune_and_develop import build_condition_name, load_config  # noqa: E402
from utils.metadata_io import read_clip_metadata  # noqa: E402
from utils.metrics import aggregate_clip_to_recording  # noqa: E402

ARCH_LABELS = {"resnet200d": "ResNet200d", "efficientnetv2_s": "EfficientNetV2-S"}
CI_METRICS = ["sensitivity", "specificity", "ppv", "npv", "accuracy", "f1"]
N_BINS = 5  # 38 音源なので 5 ビン（等頻度）
SUBDIRS = {"noisy": "external_noisy", "clean": "external"}
IMAGE_LABELS = {"noisy": "SNR 15 dB noise added (primary)", "clean": "no noise (sensitivity analysis)"}


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray, eps: float = 1e-6) -> tuple[float, float]:
    p = np.clip(p, eps, 1 - eps)
    x = sm.add_constant(np.log(p / (1 - p)))
    fit = sm.Logit(y.astype(float), x).fit(disp=0)
    return float(fit.params[1]), float(fit.params[0])


def fmt_ci(point: float, lo: float, hi: float) -> str:
    return f"{point:.3f} ({lo:.3f}–{hi:.3f})"


def main(images: str) -> int:
    sub = SUBDIRS[images]
    config = load_config("config_external.yaml")
    condition = build_condition_name(config["selected_condition"])
    results_root = Path(config["results_root"])
    ext_meta = read_clip_metadata(config["external"]["clip_metadata_csv"])

    rows, curves = [], {}
    for arch, label in ARCH_LABELS.items():
        arch_dir = results_root / arch / condition
        with open(arch_dir / sub / "external_metrics.json", encoding="utf-8") as f:
            m = json.load(f)
        with open(arch_dir / "selected_hp.json", encoding="utf-8") as f:
            hp = json.load(f)
        clip_pred = pd.read_csv(arch_dir / sub / "external_predictions.csv")
        rec = aggregate_clip_to_recording(clip_pred, ext_meta)
        rec.to_csv(arch_dir / sub / "external_predictions_recording.csv", index=False)
        y = rec["true_label"].to_numpy().astype(int)
        p = rec["pred_proba_mean"].to_numpy()

        brier = float(np.mean((p - y) ** 2))
        brier_null = float(np.mean((y.mean() - y) ** 2))
        slope, intercept = calibration_slope_intercept(y, p)
        curves[label] = {"y": y, "p": p, "auc": m["auc_recording"],
                         "lo": m["auc_recording_ci_lower"], "hi": m["auc_recording_ci_upper"],
                         "brier": brier, "slope": slope}

        row = {
            "Model": label,
            "LR / WD / epochs": f"{hp['learning_rate']:.0e} / {hp['weight_decay']:.0e} / {hp['final_epoch_count']}",
            "Development AUC (3-fold CV mean)": f"{hp['mean_val_auc']:.3f}",
            "AUC (95% CI, DeLong)": fmt_ci(m["auc_recording"], m["auc_recording_ci_lower"], m["auc_recording_ci_upper"]),
        }
        for k in CI_METRICS:
            row[f"{k.upper() if k in ('ppv', 'npv') else k.capitalize()} (95% CI)"] = fmt_ci(
                m[f"{k}_recording"], m[f"{k}_recording_ci_lower"], m[f"{k}_recording_ci_upper"])
        row.update({
            "Brier score": f"{brier:.3f}",
            "Brier (prevalence-only reference)": f"{brier_null:.3f}",
            "Calibration slope": f"{slope:.3f}",
            "Calibration intercept": f"{intercept:.3f}",
            "Locked threshold": f"{m['locked_recording_threshold']:.3f}",
            "n recordings (positive)": f"{len(y)} ({int(y.sum())})",
        })
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(results_root / f"table4_external_validation_{images}.csv", index=False, encoding="utf-8-sig")
    table.T.to_excel(results_root / f"table4_external_validation_{images}.xlsx", header=False)
    print(table.T.to_string(header=False))

    # ROC（2 モデルを重ねる）
    fig, ax = plt.subplots(figsize=(5, 5))
    for label, c in curves.items():
        fpr, tpr, _ = roc_curve(c["y"], c["p"])
        ax.plot(fpr, tpr, lw=2, label=f"{label}: AUC {c['auc']:.3f} ({c['lo']:.3f}–{c['hi']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="1 − Specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1.01),
           title=f"Temporal validation cohort (38 recordings)\n{IMAGE_LABELS[images]}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(results_root / f"roc_external_{condition}_{images}.{ext}", dpi=300)
    plt.close(fig)

    # 較正図（reliability diagram）
    fig, ax = plt.subplots(figsize=(5, 5))
    for label, c in curves.items():
        frac_pos, mean_pred = calibration_curve(c["y"], c["p"], n_bins=N_BINS, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o",
                label=f"{label}: Brier {c['brier']:.3f}, slope {c['slope']:.2f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.set(xlabel="Predicted probability", ylabel="Observed proportion (FV < 400)",
           xlim=(0, 1), ylim=(0, 1), title=f"Calibration ({N_BINS} quantile bins)\n{IMAGE_LABELS[images]}")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(results_root / f"calibration_external_{condition}_{images}.{ext}", dpi=300)
    plt.close(fig)
    print(f"保存: {results_root}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", choices=list(SUBDIRS), default="noisy")
    sys.exit(main(parser.parse_args().images))
