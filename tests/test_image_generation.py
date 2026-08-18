"""tests/test_image_generation.py

process_audio_to_images / ShuntMelDataset の画像選択ロジックの回帰テスト。

背景:
    utils/sound2image.py::process_audio_to_images は、かつて apply_noise=True の
    場合に aug_idx=0 を含む全ての保存画像へノイズを混入させており、
    ノイズなしの「素の」画像が一枚も生成されないバグがあった。
    utils/dataset.py::ShuntMelDataset は val/test 時に「素の画像」（無ければ
    _aug0）へフォールバックする設計だったが、_aug0 自体がノイズ入りだったため
    val/testの評価も実質ノイズ入り画像で行われていた。

    本テストは、修正後の仕様（クリップごとに素の画像を必ず1枚生成し、
    train はそれに加えて拡張画像 n_aug 枚を使用、val/test は素の画像のみを
    使用する）を固定する。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.dataset import ShuntMelDataset, build_transform
from utils.sound2image import process_audio_to_images


def _write_dummy_wav(path: Path, sr: int = 4000, duration_sec: float = 5.0) -> None:
    rng = np.random.default_rng(0)
    n_samples = int(sr * duration_sec)
    wave = rng.standard_normal(n_samples).astype(np.float32) * 0.1
    sf.write(str(path), wave, sr)


def _generate_images(output_dir: Path, audio_dir: Path, n_aug: int = 3) -> pd.DataFrame:
    df = pd.DataFrame([
        {"audio_file": "clipA.wav", "label": 0},
        {"audio_file": "clipB.wav", "label": 1},
    ])
    process_audio_to_images(
        df=df,
        audio_folder=str(audio_dir),
        output_folder=str(output_dir),
        label_name="label",
        threshold=0.5,
        start_time=0.0,
        clip_duration=5.0,
        target_sr=4000,
        use_mel=True,
        n_mels=64,
        n_fft_direct=256,
        hop_length_direct=64,
        fmax=1000,
        target_shape=(64, 64),
        cmap=None,
        n_aug=n_aug,
        apply_silence=False,
        apply_noise=True,
        min_snr_db=15,
        max_snr_db=15,
        base_seed=42,
        enable_multiclip=False,
    )
    return df


# ===== process_audio_to_images: ファイル生成 =====

def test_process_audio_to_images_saves_clean_plus_augmented():
    """apply_noise=True, n_aug=3 のとき、1クリップにつき「素の画像1枚 + _aug0..2の3枚」= 4枚保存される。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        audio_dir = tmpdir / "audio"
        audio_dir.mkdir()
        _write_dummy_wav(audio_dir / "clipA.wav")
        _write_dummy_wav(audio_dir / "clipB.wav")

        output_dir = tmpdir / "images"
        _generate_images(output_dir, audio_dir, n_aug=3)

        negative_files = sorted(p.name for p in (output_dir / "negative").glob("*.png"))
        assert negative_files == [
            "clipA.png", "clipA_aug0.png", "clipA_aug1.png", "clipA_aug2.png",
        ]

        positive_files = sorted(p.name for p in (output_dir / "positive").glob("*.png"))
        assert positive_files == [
            "clipB.png", "clipB_aug0.png", "clipB_aug1.png", "clipB_aug2.png",
        ]


def test_clean_image_differs_from_augmented_images():
    """素の画像（サフィックスなし）は、ノイズが加算された _aug* 画像とは画素が異なる。"""
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        audio_dir = tmpdir / "audio"
        audio_dir.mkdir()
        _write_dummy_wav(audio_dir / "clipA.wav")
        _write_dummy_wav(audio_dir / "clipB.wav")

        output_dir = tmpdir / "images"
        _generate_images(output_dir, audio_dir, n_aug=2)

        clean = np.array(Image.open(output_dir / "negative" / "clipA.png"))
        aug0 = np.array(Image.open(output_dir / "negative" / "clipA_aug0.png"))

        assert not np.array_equal(clean, aug0), (
            "素の画像とノイズ拡張画像(_aug0)が完全一致しています。"
            "ノイズが正しく加算されていない可能性があります。"
        )


def test_no_augmentation_saves_only_clean_image():
    """apply_noise=False, apply_silence=False のときは素の画像のみ保存される（拡張画像なし）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        audio_dir = tmpdir / "audio"
        audio_dir.mkdir()
        _write_dummy_wav(audio_dir / "clipA.wav")

        output_dir = tmpdir / "images"
        df = pd.DataFrame([{"audio_file": "clipA.wav", "label": 0}])
        process_audio_to_images(
            df=df,
            audio_folder=str(audio_dir),
            output_folder=str(output_dir),
            label_name="label",
            threshold=0.5,
            start_time=0.0,
            clip_duration=5.0,
            target_sr=4000,
            use_mel=True,
            n_mels=64,
            n_fft_direct=256,
            hop_length_direct=64,
            fmax=1000,
            target_shape=(64, 64),
            cmap=None,
            apply_silence=False,
            apply_noise=False,
            enable_multiclip=False,
        )

        files = sorted(p.name for p in (output_dir / "negative").glob("*.png"))
        assert files == ["clipA.png"]


# ===== ShuntMelDataset: train / val / test の画像選択 =====

def _minimal_config() -> dict:
    return {
        "mel_conditions": {"image_size": [64, 64]},
        "augmentation": {"brightness_jitter": 0.4},
        "normalization": {
            "imagenet_normalization": True,
            "imagenet_mean": [0.485, 0.456, 0.406],
            "imagenet_std": [0.229, 0.224, 0.225],
        },
    }


def test_dataset_train_mode_uses_clean_plus_all_augmented():
    """train モードはクリップごとに 素の画像1枚 + _aug*(n_aug枚) を全て使用する。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        audio_dir = tmpdir / "audio"
        audio_dir.mkdir()
        _write_dummy_wav(audio_dir / "clipA.wav")
        _write_dummy_wav(audio_dir / "clipB.wav")

        images_root = tmpdir / "images"
        condition_name = "nfft256_mel64_fmax1000"
        _generate_images(images_root / condition_name, audio_dir, n_aug=3)

        df = pd.DataFrame([
            {"clip_file": "clipA.wav", "patient_id": "P1", "label": 0},
            {"clip_file": "clipB.wav", "patient_id": "P2", "label": 1},
        ])
        transform = build_transform("train", _minimal_config())
        dataset = ShuntMelDataset(df, condition_name, images_root, transform, mode="train")

        # 2クリップ x (素の画像1枚 + _aug0..2の3枚) = 8サンプル
        assert len(dataset) == 8


def test_dataset_val_and_test_mode_use_only_clean_image():
    """val/test モードはクリップごとに素の画像（サフィックスなし）を1枚だけ使用する。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        audio_dir = tmpdir / "audio"
        audio_dir.mkdir()
        _write_dummy_wav(audio_dir / "clipA.wav")
        _write_dummy_wav(audio_dir / "clipB.wav")

        images_root = tmpdir / "images"
        condition_name = "nfft256_mel64_fmax1000"
        _generate_images(images_root / condition_name, audio_dir, n_aug=3)

        df = pd.DataFrame([
            {"clip_file": "clipA.wav", "patient_id": "P1", "label": 0},
            {"clip_file": "clipB.wav", "patient_id": "P2", "label": 1},
        ])

        for mode in ("val", "test"):
            transform = build_transform(mode, _minimal_config())
            dataset = ShuntMelDataset(df, condition_name, images_root, transform, mode=mode)

            assert len(dataset) == 2, f"mode={mode}"
            for path in dataset.image_paths:
                assert "_aug" not in path.stem, (
                    f"mode={mode} でノイズ拡張画像({path.name})が使われています。"
                    "素の画像のみを使用する必要があります。"
                )
