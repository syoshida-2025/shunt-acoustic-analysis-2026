"""scripts/generate_27condition_examples.py

論文用: 1つのWAVデータ（5秒クリップ1個分）から、
n_fft x n_mels x fmax の全27条件のメルスペクトログラム画像を生成する。

生成される画像は generate_melspectrogram.py / utils.sound2image.wave2image と
同一のロジック（実際の学習パイプラインと同じ画像）を使用する。

各画像には条件名（例: "nfft256_mel64_fmax1000"）をタイトルとして付与し、
個別のPNGに加えて、27枚をまとめたグリッド画像も出力する。

使い方:
    # デフォルト: clip_metadata.csv の先頭クリップを使用
    uv run scripts/generate_27condition_examples.py

    # data/all_sounds 内の生録音を指定（config の skip_initial_sec から5秒切り出し）
    uv run scripts/generate_27condition_examples.py --wav 00101.wav

    # data/all_clips 内の既存5秒クリップを指定
    uv run scripts/generate_27condition_examples.py --wav 00101_clip0.wav
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import clip_audio, wave2image


def load_config(config_path: str) -> dict:
    """config.yaml を読み込んで dict として返す。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_wave(
    config: dict, wav_arg: str | None, start_sec_arg: float | None
) -> tuple[np.ndarray, int, str]:
    """対象WAVを解決し、5秒分の波形データを返す。

    Parameters:
    -----------
    wav_arg : str | None
        WAVファイル名。None の場合は clip_metadata.csv の先頭クリップを使用する。
        data/all_clips 内のファイル名を指定した場合はそのまま読み込む（既に5秒）。
        data/all_sounds 内のファイル名を指定した場合は skip_initial_sec (または
        --start-sec) から5秒切り出す。
    start_sec_arg : float | None
        生録音から切り出す場合の開始秒。None なら config の skip_initial_sec を使う。

    Returns:
    --------
    tuple[np.ndarray, int, str]
        (波形データ, サンプリングレート, 参照元ファイル名の説明文字列)
    """
    audio_dir = Path(config["paths"]["audio_dir"])
    clip_dir = Path(config["paths"]["clip_dir"])
    target_sr = config["audio"]["sample_rate"]
    clip_duration = config["audio"]["clip_duration_sec"]
    skip_initial = config["audio"]["skip_initial_sec"]

    if wav_arg is None:
        clip_csv = Path(config["paths"]["clip_metadata_csv"])
        df = pd.read_csv(clip_csv)
        clip_file = df.iloc[0]["clip_file"]
        wav_path = clip_dir / clip_file
        wave_data, sr = librosa.load(str(wav_path), sr=target_sr)
        return wave_data, sr, clip_file

    wav_path = Path(wav_arg)
    if wav_path.exists():
        pass
    elif (clip_dir / wav_path.name).exists():
        wav_path = clip_dir / wav_path.name
    elif (audio_dir / wav_path.name).exists():
        wav_path = audio_dir / wav_path.name
    else:
        raise FileNotFoundError(
            f"WAVファイルが見つかりません: {wav_arg} "
            f"(data/all_clips, data/all_sounds のいずれにも存在しません)"
        )

    is_raw_recording = wav_path.parent.resolve() == audio_dir.resolve()
    if is_raw_recording:
        start_sec = start_sec_arg if start_sec_arg is not None else skip_initial
        wave_data, sr = clip_audio(
            file_path=str(wav_path),
            start_time=start_sec,
            duration=clip_duration,
            target_sr=target_sr,
        )
    else:
        wave_data, sr = librosa.load(str(wav_path), sr=target_sr)
        target_length = int(clip_duration * sr)
        if len(wave_data) < target_length:
            wave_data = np.pad(wave_data, (0, target_length - len(wave_data)))
        elif len(wave_data) > target_length:
            wave_data = wave_data[:target_length]

    return wave_data, sr, wav_path.name


def main() -> int:
    parser = argparse.ArgumentParser(
        description="1WAVデータから全27条件のメルスペクトログラム例示画像を生成する"
    )
    parser.add_argument("--config", default="config.yaml", help="config.yaml へのパス")
    parser.add_argument(
        "--wav",
        default=None,
        help="対象WAVファイル名（data/all_sounds または data/all_clips 内）。"
        "省略時は clip_metadata.csv の先頭クリップを使用",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=None,
        help="生録音から切り出す場合の開始秒（省略時は config.audio.skip_initial_sec）",
    )
    parser.add_argument(
        "--output-dir",
        default="./results/example_27conditions",
        help="出力先フォルダ",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    wave_data, sr, source_name = resolve_wave(config, args.wav, args.start_sec)

    mel_cfg = config["mel_conditions"]
    n_fft_list = sorted(mel_cfg["n_fft"])
    n_mels_list = sorted(mel_cfg["n_mels"])
    fmax_list = sorted(mel_cfg["fmax"])
    hop_ratio = mel_cfg["hop_length_ratio"]
    target_shape = tuple(mel_cfg["image_size"])
    cmap = mel_cfg["cmap"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = list(itertools.product(n_fft_list, n_mels_list, fmax_list))
    print(f"対象WAV: {source_name} (sr={sr}Hz, {len(wave_data) / sr:.2f}秒)")
    print(f"生成条件数: {len(conditions)}")

    images: dict[str, np.ndarray] = {}
    for n_fft, n_mels, fmax in conditions:
        hop_length = int(n_fft * hop_ratio)
        condition_name = f"nfft{n_fft}_mel{n_mels}_fmax{fmax}"
        img_array = wave2image(
            wave_data=wave_data,
            sr=sr,
            n_mels=n_mels,
            n_fft_direct=n_fft,
            hop_length_direct=hop_length,
            fmax=fmax,
            target_shape=target_shape,
            cmap=cmap,
        )
        images[condition_name] = img_array
        Image.fromarray(img_array).save(output_dir / f"{condition_name}.png")
        print(f"  {condition_name}: 保存済み")

    # 論文用グリッド画像（縦: n_fft x fmax, 横: n_mels）
    n_rows = len(n_fft_list) * len(fmax_list)
    n_cols = len(n_mels_list)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    for row_idx, (n_fft, fmax) in enumerate(itertools.product(n_fft_list, fmax_list)):
        for col_idx, n_mels in enumerate(n_mels_list):
            condition_name = f"nfft{n_fft}_mel{n_mels}_fmax{fmax}"
            ax = axes[row_idx, col_idx]
            ax.imshow(images[condition_name])
            ax.set_title(condition_name, fontsize=9)
            ax.axis("off")

    fig.suptitle(
        f"Mel-spectrogram examples across 27 conditions (source: {source_name})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    grid_path = output_dir / "grid_27conditions.png"
    fig.savefig(grid_path, dpi=200)
    plt.close(fig)

    print(f"\n個別画像を {len(conditions)} 枚保存しました: {output_dir}")
    print(f"グリッド画像を保存しました: {grid_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
