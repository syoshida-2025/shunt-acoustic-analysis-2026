"""scripts/create_clipped_audio.py

5秒クリップWAVファイルを生成する。

clip_metadata.csv の各行に対応する WAV ファイルを data/all_clips/ に保存する。
切り出しは librosa.load で行い、soundfile.write で int16 として保存する。

このスクリプトはプロジェクト全体で1回だけ実行する。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import read_clip_metadata


def main(config_path: str) -> int:
    """エントリーポイント。"""
    config = load_config(config_path)

    # 入力
    try:
        clip_df = read_clip_metadata(config["paths"]["clip_metadata_csv"])
    except FileNotFoundError as e:
        print(f"ERROR: {e}\n先に generate_metadata.py を実行してください。", file=sys.stderr)
        return 1

    audio_dir = Path(config["paths"]["audio_dir"])
    clip_dir = Path(config["paths"]["clip_dir"])
    clip_dir.mkdir(parents=True, exist_ok=True)

    clip_duration = config["audio"]["clip_duration_sec"]
    target_sr = config["audio"]["sample_rate"]

    # 元WAVをキャッシュ（同じ録音から複数クリップを切り出すため）
    wave_cache: dict[str, tuple[np.ndarray, int]] = {}
    n_success = 0
    n_skip = 0
    errors: list[str] = []

    for _, clip in tqdm(clip_df.iterrows(), total=len(clip_df), desc="クリップ生成"):
        clip_path = clip_dir / clip["clip_file"]

        # すでに存在する場合はスキップ（再実行対応）
        if clip_path.exists():
            n_skip += 1
            continue

        # 元WAVをキャッシュから取得 or 読み込み
        audio_file = clip["audio_file"]
        if audio_file not in wave_cache:
            try:
                wave, sr = librosa.load(audio_dir / audio_file, sr=target_sr, mono=True)
                wave_cache[audio_file] = (wave, sr)
            except Exception as e:
                errors.append(f"{audio_file}: {type(e).__name__}: {e}")
                continue

        wave, sr = wave_cache[audio_file]

        # 切り出し
        start_sample = int(clip["start_sec"] * sr)
        end_sample = start_sample + int(clip_duration * sr)
        if end_sample > len(wave):
            errors.append(
                f"{clip['clip_file']}: end_sample ({end_sample}) > "
                f"wave length ({len(wave)})"
            )
            continue
        clip_wave = wave[start_sample:end_sample]

        # int16 で保存
        clip_int16 = (clip_wave * 32767.0).clip(-32768, 32767).astype(np.int16)
        sf.write(clip_path, clip_int16, sr, subtype="PCM_16")
        n_success += 1

        # メモリ節約：同じ audio_file の最後のクリップを処理した後にキャッシュを削除
        same_audio = clip_df[clip_df["audio_file"] == audio_file]
        if clip["clip_idx"] == same_audio["clip_idx"].max():
            del wave_cache[audio_file]

    # サマリー
    print(f"\n[create_clipped_audio] 完了:")
    print(f"  生成: {n_success} クリップ")
    print(f"  スキップ（既存）: {n_skip} クリップ")
    if errors:
        print(f"  エラー: {len(errors)} 件")
        for err in errors[:10]:
            print(f"    - {err}")
        if len(errors) > 10:
            print(f"    ... 他 {len(errors) - 10} 件")

    # ディスク使用量
    wav_files = list(clip_dir.glob("*.wav"))
    if wav_files:
        total_bytes = sum(p.stat().st_size for p in wav_files)
        print(f"  合計ディスク使用量: {total_bytes / 1024 / 1024:.1f} MB")

    return 0 if not errors else 1


def load_config(config_path: str) -> dict:
    """config.yaml を読み込んで dict として返す。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="5秒クリップWAV生成スクリプト")
    parser.add_argument("--config", default="config.yaml", help="config.yaml へのパス")
    args = parser.parse_args()

    sys.exit(main(args.config))
