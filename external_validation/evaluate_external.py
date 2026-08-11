"""external_validation/evaluate_external.py

外部検証パイプライン Phase B: ロック済みモデル・閾値を外部hold-outデータに
1回だけ適用し、汎化性能を評価する。

前提:
    - tune_and_develop.py が完了しており、
      results/{architecture}/{condition_name}/final_model.pth と
      locked_threshold.json が存在すること。
    - 外部データのクリップメタデータ(config_external.yaml の external セクション
      で指定した clip_metadata_csv)が、scripts/generate_metadata.py と
      scripts/create_clipped_audio.py を外部データ向けの設定で実行済みであること
      (本スクリプトはメタデータ生成そのものは行わない)。

このスクリプトは外部データに対して以下を一度だけ行う:
    1. 開発データと外部データの間で患者IDの重複がないことを確認する。
    2. 選択した条件のメルスペクトログラム画像を外部データに対して生成する(未生成の場合のみ)。
    3. ロック済みモデルで推論する(データ拡張は適用しない)。
    4. recording-level AUC(主指標、閾値非依存)と、ロック済み閾値による
       二次指標(F1・感度・特異度等)を算出する。
    5. 予測確率をCSVに保存する(ROC曲線等は事後的にこのCSVから作図できる)。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.dataset import build_dataloader
from utils.external_metrics import evaluate_with_locked_threshold
from utils.metadata_io import read_clip_metadata
from utils.sound2image import process_audio_to_images
from utils.trainer import create_model, predict_on_loader

from tune_and_develop import build_condition_name, build_runtime_config, load_config


def check_no_patient_overlap(dev_clip_df: pd.DataFrame, external_clip_df: pd.DataFrame) -> None:
    dev_patients = set(dev_clip_df["patient_id"].astype(str).unique())
    external_patients = set(external_clip_df["patient_id"].astype(str).unique())
    overlap = dev_patients & external_patients
    if overlap:
        raise RuntimeError(
            f"開発データと外部データで患者IDが重複しています: {overlap}\n"
            f"外部検証の前提(開発・外部が独立したコホートであること)が崩れています。"
        )
    print(
        f"[check_no_patient_overlap] 重複なしを確認しました "
        f"(開発: {len(dev_patients)}人, 外部: {len(external_patients)}人)"
    )


def ensure_external_images(config: dict, condition_name: str) -> Path:
    """外部データについて、選択した条件のメルスペクトログラム画像を生成する(未生成の場合のみ)。"""
    images_root = Path(config["external"]["images_root"])
    condition_dir = images_root / condition_name
    condition_dir.mkdir(parents=True, exist_ok=True)

    if (condition_dir / ".complete").exists():
        print(f"[ensure_external_images] {condition_name} は生成済み。スキップ。")
        return condition_dir

    clip_df = read_clip_metadata(config["external"]["clip_metadata_csv"])
    df_for_process = clip_df.copy()
    df_for_process["audio_file"] = df_for_process["clip_file"]

    sc = config["selected_condition"]
    aug = config["augmentation"]
    hop_length = int(sc["n_fft"] * config["mel_conditions"]["hop_length_ratio"])

    print(f"[ensure_external_images] {condition_name} の画像を生成中(外部データ)...")
    # 外部データは評価専用のため、ノイズ拡張は適用しない(apply_noise=False, n_aug=1)。
    # 拡張の有無によらず build_dataloader(mode="test") は無加工画像だけを使うが、
    # 不要な拡張コピーを外部データ側で作らないよう、生成段階から素の画像のみにする。
    process_audio_to_images(
        df=df_for_process,
        audio_folder=config["external"]["clip_dir"],
        output_folder=str(condition_dir),
        label_name="label",
        threshold=0.5,
        start_time=0.0,
        clip_duration=config["audio"]["clip_duration_sec"],
        target_sr=config["audio"]["sample_rate"],
        use_mel=True,
        n_mels=sc["n_mels"],
        n_fft_direct=sc["n_fft"],
        hop_length_direct=hop_length,
        fmax=sc["fmax"],
        target_shape=tuple(config["mel_conditions"]["image_size"]),
        cmap=config["mel_conditions"]["cmap"],
        n_aug=1,
        apply_silence=False,
        max_silence_duration=0.0,
        apply_noise=False,
        min_snr_db=aug["noise_snr_db_min"],
        max_snr_db=aug["noise_snr_db_max"],
        strength=1.0,
        use_adaptive_snr=False,
        enable_multiclip=False,
        base_seed=42,
    )
    (condition_dir / ".complete").touch()
    return condition_dir


def load_final_model(config: dict, architecture_key: str, condition_name: str) -> torch.nn.Module:
    output_dir = Path(config["results_root"]) / architecture_key / condition_name
    checkpoint = torch.load(output_dir / "final_model.pth", map_location="cpu", weights_only=False)

    runtime_config = build_runtime_config(config, architecture_key)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = create_model(runtime_config["model"]["main"], device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def run_architecture(config: dict, architecture_key: str, condition_name: str) -> dict:
    output_dir = Path(config["results_root"]) / architecture_key / condition_name
    external_dir = output_dir / "external"
    external_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "locked_threshold.json", "r", encoding="utf-8") as f:
        locked_thresholds = json.load(f)

    external_clip_df = read_clip_metadata(config["external"]["clip_metadata_csv"])

    runtime_config = build_runtime_config(
        config, architecture_key, images_root=config["external"]["images_root"]
    )
    test_loader = build_dataloader(external_clip_df, condition_name, runtime_config, mode="test")

    model = load_final_model(config, architecture_key, condition_name)
    predictions = predict_on_loader(model, test_loader, runtime_config)
    predictions.to_csv(external_dir / "external_predictions.csv", index=False)

    metrics = evaluate_with_locked_threshold(
        predictions=predictions,
        clip_metadata=external_clip_df,
        clip_threshold=locked_thresholds["clip_threshold"],
        recording_threshold=locked_thresholds["recording_threshold"],
    )
    with open(external_dir / "external_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(output_dir / "selected_hp.json", "r", encoding="utf-8") as f:
        selected_hp = json.load(f)

    summary_lines = [
        f"アーキテクチャ: {architecture_key}",
        f"条件: {condition_name}",
        f"開発フェーズ(3-fold探索, 採用HP): mean_val_auc = {selected_hp['mean_val_auc']:.4f}",
        f"外部データ: auc_recording = {metrics['auc_recording']:.4f} "
        f"[{metrics['auc_recording_ci_lower']:.4f}, {metrics['auc_recording_ci_upper']:.4f}] "
        f"(n_recordings={metrics['n_recordings']})",
        f"外部データ(ロック済み閾値={metrics['locked_recording_threshold']:.4f}): "
        f"F1={metrics['f1_recording']:.4f}, Sens={metrics['sensitivity_recording']:.4f}, "
        f"Spec={metrics['specificity_recording']:.4f}",
    ]
    summary = "\n".join(summary_lines)
    with open(external_dir / "external_vs_dev_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(summary)

    return metrics


def main(config_path: str, architecture: str | None) -> int:
    config = load_config(config_path)
    condition_name = build_condition_name(config["selected_condition"])

    dev_clip_df = read_clip_metadata(config["dev"]["clip_metadata_csv"])
    external_clip_df = read_clip_metadata(config["external"]["clip_metadata_csv"])
    check_no_patient_overlap(dev_clip_df, external_clip_df)

    ensure_external_images(config, condition_name)

    architectures = [architecture] if architecture else list(config["architectures"].keys())
    for arch_key in architectures:
        print(f"\n{'=' * 60}\nアーキテクチャ: {arch_key}\n{'=' * 60}")
        run_architecture(config, arch_key, condition_name)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="外部検証Phase B: ロック済みモデルの外部データへの適用")
    parser.add_argument("--config", default="config_external.yaml")
    parser.add_argument(
        "--architecture",
        default=None,
        help="config_external.yaml の architectures キーの1つ。省略時は全アーキテクチャを実行。",
    )
    args = parser.parse_args()
    sys.exit(main(args.config, args.architecture))
