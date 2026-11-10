"""temporal_validation/hp_tuned/evaluate_external_noisy.py

v4 追加（2026-10-27）: temporal cohort への適用を「SNR 15 dB ノイズを1回付加した画像」で行う（主解析）。

理由:
    v2 の内部検証では、検証・テスト画像にもノイズを1回付加した画像（_aug0）が使われ、
    HP 探索の検証 fold と閾値ロックも同じ扱いだった。temporal cohort にも同じ前処理を
    かけることで、開発から評価まで前処理が一貫する。
    ノイズなし画像での評価（2026-08 の設計、evaluate_external.py → external/）は感度分析とする。

処理は evaluate_external.py（2026-08 版）と同じで、違いは次の2点だけ:
    - 外部画像の生成で apply_noise=True, n_aug=1（SNR 15 dB、base_seed=42。
      10/20 の fixed_hp_1020/01_prepare_images.py の noisy 画像と同じ作り方）
    - 出力先が external_noisy/、画像フォルダが images_external_noisy/

実行（このフォルダで、retrain_final_seeded.py の後に）:
    uv run evaluate_external_noisy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from evaluate_external import check_no_patient_overlap, load_final_model  # noqa: E402
from tune_and_develop import build_condition_name, build_runtime_config, load_config  # noqa: E402
from utils.dataset import build_dataloader  # noqa: E402
from utils.external_metrics import evaluate_with_locked_threshold  # noqa: E402
from utils.metadata_io import read_clip_metadata  # noqa: E402
from utils.sound2image import process_audio_to_images  # noqa: E402
from utils.trainer import predict_on_loader  # noqa: E402

IMAGES_ROOT = "./images_external_noisy"
OUT_SUBDIR = "external_noisy"


def ensure_external_noisy_images(config: dict, condition_name: str) -> None:
    condition_dir = Path(IMAGES_ROOT) / condition_name
    condition_dir.mkdir(parents=True, exist_ok=True)
    if (condition_dir / ".complete").exists():
        print(f"[noisy images] {condition_name} は生成済み。スキップ。")
        return

    clip_df = read_clip_metadata(config["external"]["clip_metadata_csv"])
    df_for_process = clip_df.copy()
    df_for_process["audio_file"] = df_for_process["clip_file"]
    sc = config["selected_condition"]
    aug = config["augmentation"]

    print(f"[noisy images] {condition_name} の外部画像を生成中（SNR {aug['noise_snr_db_min']} dB を1回付加）...")
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
        hop_length_direct=int(sc["n_fft"] * config["mel_conditions"]["hop_length_ratio"]),
        fmax=sc["fmax"],
        target_shape=tuple(config["mel_conditions"]["image_size"]),
        cmap=config["mel_conditions"]["cmap"],
        n_aug=1,  # 1枚だけ → ファイル名に _aug サフィックスは付かない
        apply_silence=False,
        max_silence_duration=0.0,
        apply_noise=True,
        min_snr_db=aug["noise_snr_db_min"],
        max_snr_db=aug["noise_snr_db_max"],
        strength=1.0,
        use_adaptive_snr=False,
        enable_multiclip=False,
        base_seed=42,
    )
    (condition_dir / ".complete").touch()


def run_architecture(config: dict, architecture_key: str, condition_name: str) -> None:
    output_dir = Path(config["results_root"]) / architecture_key / condition_name
    out_dir = output_dir / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "locked_threshold.json", encoding="utf-8") as f:
        locked = json.load(f)
    external_clip_df = read_clip_metadata(config["external"]["clip_metadata_csv"])

    runtime_config = build_runtime_config(config, architecture_key, images_root=IMAGES_ROOT)
    test_loader = build_dataloader(external_clip_df, condition_name, runtime_config, mode="test")
    model = load_final_model(config, architecture_key, condition_name)
    predictions = predict_on_loader(model, test_loader, runtime_config)
    predictions.to_csv(out_dir / "external_predictions.csv", index=False)

    metrics = evaluate_with_locked_threshold(
        predictions=predictions,
        clip_metadata=external_clip_df,
        clip_threshold=locked["clip_threshold"],
        recording_threshold=locked["recording_threshold"],
    )
    with open(out_dir / "external_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(
        f"[{architecture_key}] noisy: AUC {metrics['auc_recording']:.3f} "
        f"({metrics['auc_recording_ci_lower']:.3f}-{metrics['auc_recording_ci_upper']:.3f}), "
        f"Sens {metrics['sensitivity_recording']:.3f}, Spec {metrics['specificity_recording']:.3f}"
    )


def main() -> int:
    config = load_config("config_external.yaml")
    condition_name = build_condition_name(config["selected_condition"])
    check_no_patient_overlap(read_clip_metadata(config["dev"]["clip_metadata_csv"]),
                             read_clip_metadata(config["external"]["clip_metadata_csv"]))
    ensure_external_noisy_images(config, condition_name)
    for arch_key in config["architectures"]:
        run_architecture(config, arch_key, condition_name)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
