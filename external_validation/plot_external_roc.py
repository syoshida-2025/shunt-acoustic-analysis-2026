"""external_validation/plot_external_roc.py

外部hold-outテストセットにおける recording-level ROC曲線を、複数アーキテクチャ
重ね書きで1枚のFigureとして出力する。

入力: results/{architecture}/{condition_name}/external/external_predictions.csv
      (evaluate_external.py の出力。clip_file, patient_id, true_label, pred_proba)

出力: results/roc_external_{condition_name}.png / .pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metadata_io import read_clip_metadata
from utils.metrics import aggregate_clip_to_recording

from tune_and_develop import build_condition_name, load_config

ARCHITECTURE_LABELS = {
    "resnet200d": "ResNet200d",
    "efficientnetv2_s": "EfficientNetV2-S",
}

# アーキテクチャごとに固定の色・線種を割り当てる(重ね書き時の見分けやすさのため)
ARCHITECTURE_STYLES = {
    "resnet200d": {"color": "#1f77b4", "linestyle": "-"},
    "efficientnetv2_s": {"color": "#d62728", "linestyle": "--"},
}


def load_recording_level_predictions(
    predictions_csv: Path, external_clip_df: pd.DataFrame
) -> pd.DataFrame:
    predictions = pd.read_csv(predictions_csv)
    recording_pred = aggregate_clip_to_recording(predictions, external_clip_df)
    return recording_pred


def plot_roc(config_path: str, architectures: list[str], output_path: str | None) -> None:
    config = load_config(config_path)
    condition_name = build_condition_name(config["selected_condition"])
    external_clip_df = read_clip_metadata(config["external"]["clip_metadata_csv"])

    fig, ax = plt.subplots(figsize=(6, 6))

    for arch_key in architectures:
        predictions_csv = (
            Path(config["results_root"]) / arch_key / condition_name
            / "external" / "external_predictions.csv"
        )
        if not predictions_csv.exists():
            print(f"[skip] {arch_key}: {predictions_csv} が見つかりません(evaluate_external.py未実行?)")
            continue

        recording_pred = load_recording_level_predictions(predictions_csv, external_clip_df)
        fpr, tpr, _ = roc_curve(recording_pred["true_label"], recording_pred["pred_proba_mean"])
        roc_auc = sk_auc(fpr, tpr)

        label = ARCHITECTURE_LABELS.get(arch_key, arch_key)
        style = ARCHITECTURE_STYLES.get(arch_key, {})
        ax.plot(
            fpr, tpr,
            label=f"{label} (AUC = {roc_auc:.3f})",
            linewidth=2,
            **style,
        )
        print(f"{arch_key}: recording-level AUC = {roc_auc:.4f} (n={len(recording_pred)})")

    ax.plot([0, 1], [0, 1], linestyle=":", color="gray", linewidth=1, label="Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"External Test Set ROC Curve\n({condition_name})")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if output_path is None:
        output_dir = Path(config["results_root"])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"roc_external_{condition_name}.png")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = str(Path(output_path).with_suffix(".pdf"))
    fig.savefig(pdf_path, bbox_inches="tight")
    svg_path = str(Path(output_path).with_suffix(".svg"))
    fig.savefig(svg_path, bbox_inches="tight")
    print(f"保存しました: {output_path}")
    print(f"保存しました: {pdf_path}")
    print(f"保存しました: {svg_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="外部テストセットのROC曲線を重ね書きで作図する")
    parser.add_argument("--config", default="config_external.yaml")
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=None,
        help="重ね書きするアーキテクチャキー(config_external.yaml の architectures)。省略時は全アーキテクチャ。",
    )
    parser.add_argument("--output", default=None, help="出力パス(省略時は results/roc_external_{condition}.png)")
    args = parser.parse_args()

    config = load_config(args.config)
    architectures = args.architectures or list(config["architectures"].keys())
    plot_roc(args.config, architectures, args.output)
