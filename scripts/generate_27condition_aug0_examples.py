"""scripts/generate_27condition_aug0_examples.py

検証用: 実際の学習パイプライン（utils.sound2image.process_audio_to_images）を
config.yaml の augmentation 設定そのまま（apply_noise=True, n_aug=100）で
1クリップに対して呼び出し、val/test で実際に使われる "_aug0.png" が
本当にノイズ入りかどうかを27条件すべてで確認する。

generate_27condition_examples.py で生成した「ノイズなし(clean)」画像と
同じWAV・同じ条件で並べて比較できるよう、ファイル名を揃えて出力する。
さらに clean 画像との平均絶対差(0-255スケール)も併せて表示し、
ノイズが乗っているかどうかを数値でも確認する。

使い方:
    uv run scripts/generate_27condition_aug0_examples.py
    uv run scripts/generate_27condition_aug0_examples.py --wav 00101.wav
"""

from __future__ import annotations

import argparse
import itertools
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import process_audio_to_images
from scripts.generate_27condition_examples import load_config, resolve_wave  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="実パイプライン(process_audio_to_images, apply_noise=True)で"
        "aug0画像を27条件分生成し、ノイズが乗っているか検証する"
    )
    parser.add_argument("--config", default="config.yaml", help="config.yaml へのパス")
    parser.add_argument(
        "--wav",
        default=None,
        help="対象WAVファイル名。省略時は clip_metadata.csv の先頭クリップ"
        "（generate_27condition_examples.py のデフォルトと同一クリップ）",
    )
    parser.add_argument("--start-sec", type=float, default=None)
    parser.add_argument(
        "--output-dir",
        default="./results/example_27conditions_aug0",
        help="出力先フォルダ",
    )
    parser.add_argument(
        "--clean-dir",
        default="./results/example_27conditions",
        help="比較対象となる clean(ノイズなし)画像のフォルダ"
        "（generate_27condition_examples.py の出力先）",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    wave_data, sr, source_name = resolve_wave(config, args.wav, args.start_sec)

    mel_cfg = config["mel_conditions"]
    aug_cfg = config["augmentation"]
    n_fft_list = sorted(mel_cfg["n_fft"])
    n_mels_list = sorted(mel_cfg["n_mels"])
    fmax_list = sorted(mel_cfg["fmax"])
    hop_ratio = mel_cfg["hop_length_ratio"]
    target_shape = tuple(mel_cfg["image_size"])
    cmap = mel_cfg["cmap"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_dir = Path(args.clean_dir)

    # process_audio_to_images 用に単一クリップのWAVを一時保存
    # （関数がファイルパスからの読み込みを前提としているため）
    tmp_audio_dir = output_dir / "_tmp_audio_in"
    tmp_audio_dir.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    single_clip_name = "target_clip.wav"
    sf.write(str(tmp_audio_dir / single_clip_name), wave_data, sr)
    df = pd.DataFrame([{"audio_file": single_clip_name, "label": 0}])

    conditions = list(itertools.product(n_fft_list, n_mels_list, fmax_list))
    print(f"対象WAV: {source_name} (sr={sr}Hz, {len(wave_data) / sr:.2f}秒)")
    print(f"検証条件数: {len(conditions)}")
    print(
        f"augmentation設定: apply_noise={aug_cfg['apply_noise']}, "
        f"snr={aug_cfg['noise_snr_db_min']}-{aug_cfg['noise_snr_db_max']}dB, "
        f"n_aug={aug_cfg['noise_n_aug']}"
    )

    images: dict[str, np.ndarray] = {}
    diffs: dict[str, float] = {}

    tmp_out_dir = output_dir / "_tmp_process_out"

    for n_fft, n_mels, fmax in conditions:
        hop_length = int(n_fft * hop_ratio)
        condition_name = f"nfft{n_fft}_mel{n_mels}_fmax{fmax}"

        shutil.rmtree(tmp_out_dir, ignore_errors=True)

        # 実パイプラインと全く同じ関数・同じconfig値で生成
        # （generate_melspectrogram.py が呼んでいるのと同一の process_audio_to_images）
        process_audio_to_images(
            df=df,
            audio_folder=str(tmp_audio_dir),
            output_folder=str(tmp_out_dir),
            label_name="label",
            threshold=0.5,
            start_time=0.0,
            clip_duration=config["audio"]["clip_duration_sec"],
            target_sr=sr,
            use_mel=True,
            n_mels=n_mels,
            n_fft_direct=n_fft,
            hop_length_direct=hop_length,
            fmax=fmax,
            target_shape=target_shape,
            cmap=cmap,
            n_aug=aug_cfg["noise_n_aug"],
            apply_silence=aug_cfg["apply_silence"],
            apply_noise=aug_cfg["apply_noise"],
            min_snr_db=aug_cfg["noise_snr_db_min"],
            max_snr_db=aug_cfg["noise_snr_db_max"],
            strength=1.0,
            use_adaptive_snr=False,
            enable_multiclip=False,
            base_seed=42,
        )

        aug0_path = tmp_out_dir / "negative" / "target_clip_aug0.png"
        if not aug0_path.exists():
            raise FileNotFoundError(f"aug0画像が生成されませんでした: {aug0_path}")

        img_array = np.array(Image.open(aug0_path).convert("RGB"))
        images[condition_name] = img_array

        out_path = output_dir / f"{condition_name}_aug0.png"
        Image.fromarray(img_array).save(out_path)

        # clean版との差分を数値確認
        clean_path = clean_dir / f"{condition_name}.png"
        if clean_path.exists():
            clean_img = np.array(Image.open(clean_path).convert("L"))
            aug0_gray = img_array[:, :, 0]
            diff = np.abs(aug0_gray.astype(int) - clean_img.astype(int))
            diffs[condition_name] = float(diff.mean())
            print(
                f"  {condition_name}: 保存済み (clean比 平均絶対差={diff.mean():.2f}/255)"
            )
        else:
            print(f"  {condition_name}: 保存済み (clean画像が無いため差分比較スキップ)")

    shutil.rmtree(tmp_out_dir, ignore_errors=True)
    shutil.rmtree(tmp_audio_dir, ignore_errors=True)

    # グリッド画像
    n_rows = len(n_fft_list) * len(fmax_list)
    n_cols = len(n_mels_list)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    for row_idx, (n_fft, fmax) in enumerate(itertools.product(n_fft_list, fmax_list)):
        for col_idx, n_mels in enumerate(n_mels_list):
            condition_name = f"nfft{n_fft}_mel{n_mels}_fmax{fmax}"
            ax = axes[row_idx, col_idx]
            ax.imshow(images[condition_name])
            title = condition_name
            if condition_name in diffs:
                title += f"\ndiff={diffs[condition_name]:.1f}"
            ax.set_title(title, fontsize=8)
            ax.axis("off")

    fig.suptitle(
        f"aug0 images (apply_noise=True, SNR={aug_cfg['noise_snr_db_min']}dB) "
        f"— source: {source_name}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    grid_path = output_dir / "grid_27conditions_aug0.png"
    fig.savefig(grid_path, dpi=200)
    plt.close(fig)

    if diffs:
        mean_diff = float(np.mean(list(diffs.values())))
        print(f"\n27条件平均の clean-aug0 絶対差: {mean_diff:.2f} / 255")
        print("→ 0に近ければノイズなし、大きいほどノイズの影響が強いことを意味する")

    print(f"\n個別画像を {len(conditions)} 枚保存しました: {output_dir}")
    print(f"グリッド画像を保存しました: {grid_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
