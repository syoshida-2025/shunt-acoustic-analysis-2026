"""scripts/generate_melspectrogram.py

1条件分のメルスペクトログラム画像を生成する。

condition_name は "nfft{N}_mel{M}_fmax{F}" 形式（例: "nfft256_mel64_fmax1000"）。
このスクリプトは1条件ずつ呼び出される。

train_model.py から条件ごとに subprocess で呼び出されるか、ユーザーが手動で実行する。

注意: clip_metadata.csv の clip_file カラムを process_audio_to_images に渡す際、
      audio_file カラムとして渡す必要がある（関数の入力仕様に合わせるため）。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import read_clip_metadata, process_audio_to_images


def main(config_path: str, condition_name: str) -> int:
    """エントリーポイント。"""
    config = load_config(config_path)

    # condition_name からパラメータ復元
    try:
        params = parse_condition_name(condition_name)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    n_fft = params["n_fft"]
    n_mels = params["n_mels"]
    fmax = params["fmax"]
    hop_length = int(n_fft * config["mel_conditions"]["hop_length_ratio"])

    print(f"[generate_melspectrogram] 条件: {condition_name}")
    print(f"  n_fft={n_fft}, n_mels={n_mels}, fmax={fmax}, hop_length={hop_length}")

    # 入力
    try:
        clip_df = read_clip_metadata(config["paths"]["clip_metadata_csv"])
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    clip_dir = Path(config["paths"]["clip_dir"])
    images_root = Path(config["paths"]["images_root"])
    condition_dir = images_root / condition_name
    condition_dir.mkdir(parents=True, exist_ok=True)

    # 既に完了している場合はスキップ
    if is_condition_complete(condition_dir, clip_df):
        print(f"[generate_melspectrogram] {condition_name} は既に完了しています。スキップ。")
        return 0

    # process_audio_to_images の入力DataFrameを構築
    # clip_metadata の clip_file を audio_file として渡す（process_audio_to_images の仕様に合わせる）
    df_for_process = clip_df.copy()
    df_for_process["audio_file"] = df_for_process["clip_file"]

    # 拡張パラメータ
    aug = config["augmentation"]

    process_audio_to_images(
        df=df_for_process,
        audio_folder=str(clip_dir),
        output_folder=str(condition_dir),
        label_name="label",
        threshold=0.5,      # label は 0/1 の int なので > 0.5 で positive
        start_time=0.0,     # クリップ済みなのでオフセット 0
        clip_duration=config["audio"]["clip_duration_sec"],
        target_sr=config["audio"]["sample_rate"],
        use_mel=True,
        n_mels=n_mels,
        # 直接モード（Part 2 で拡張）
        n_fft_direct=n_fft,
        hop_length_direct=hop_length,
        fmax=fmax,
        target_shape=tuple(config["mel_conditions"]["image_size"]),
        cmap=config["mel_conditions"]["cmap"],
        # 拡張
        n_aug=aug["noise_n_aug"],
        apply_silence=aug["apply_silence"],
        max_silence_duration=0.0,
        apply_noise=aug["apply_noise"],
        min_snr_db=aug["noise_snr_db_min"],
        max_snr_db=aug["noise_snr_db_max"],
        strength=1.0,
        use_adaptive_snr=False,
        enable_multiclip=False,  # クリップ済みなのでマルチクリップ不要
        clip_step=aug["clip_step_sec"],
        max_clip=aug["max_clips_per_recording"],
        base_seed=42,
    )

    # 完了マーカー
    (condition_dir / ".complete").touch()

    # サマリー
    pos_dir = condition_dir / "positive"
    neg_dir = condition_dir / "negative"
    n_positive = len(list(pos_dir.glob("*.png"))) if pos_dir.exists() else 0
    n_negative = len(list(neg_dir.glob("*.png"))) if neg_dir.exists() else 0
    print(f"[generate_melspectrogram] 完了:")
    print(f"  positive: {n_positive} 画像")
    print(f"  negative: {n_negative} 画像")
    print(f"  合計: {n_positive + n_negative} 画像")

    return 0


def parse_condition_name(condition_name: str) -> dict[str, int]:
    """condition_name から n_fft, n_mels, fmax を復元する。

    例:
        "nfft256_mel64_fmax1000" → {"n_fft": 256, "n_mels": 64, "fmax": 1000}

    Raises:
        ValueError: フォーマットが不正な場合。
    """
    pattern = r"nfft(\d+)_mel(\d+)_fmax(\d+)"
    m = re.match(pattern, condition_name)
    if not m:
        raise ValueError(
            f"condition_name の形式が不正: '{condition_name}'\n"
            f"期待形式: 'nfft{{N}}_mel{{M}}_fmax{{F}}' (例: 'nfft256_mel64_fmax1000')"
        )
    return {
        "n_fft": int(m.group(1)),
        "n_mels": int(m.group(2)),
        "fmax": int(m.group(3)),
    }


def is_condition_complete(condition_dir: Path, clip_df: pd.DataFrame) -> bool:
    """指定条件のメルスペクトログラム生成が完了しているかを判定する。

    判定基準: condition_dir/.complete マーカーが存在する
    """
    return (condition_dir / ".complete").exists()


def load_config(config_path: str) -> dict:
    """config.yaml を読み込んで dict として返す。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="メルスペクトログラム画像生成スクリプト")
    parser.add_argument("--config", default="config.yaml", help="config.yaml へのパス")
    parser.add_argument(
        "--condition",
        required=True,
        help="condition_name (例: nfft256_mel64_fmax1000)",
    )
    args = parser.parse_args()

    sys.exit(main(args.config, args.condition))
