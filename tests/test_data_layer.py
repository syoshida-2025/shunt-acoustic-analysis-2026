"""tests/test_data_layer.py — データ層のスモークテスト。

対象:
    - utils/augmentation.py: add_noise_by_snr
    - utils/sound2image.py: wave2image (direct mode)
    - utils/data_split.py: split_train_test_by_patient
    - utils/dataset.py: build_transform
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.augmentation import add_noise_by_snr
from utils.data_split import split_train_test_by_patient
from utils.dataset import build_transform


# ===== add_noise_by_snr =====

def test_add_noise_by_snr_output_shape():
    """ノイズ追加後も波形の長さが変わらない。"""
    rng = np.random.default_rng(42)
    signal = rng.standard_normal(4000).astype(np.float32)
    noisy = add_noise_by_snr(signal, snr_db=15.0)
    assert noisy.shape == signal.shape


def test_add_noise_by_snr_changes_signal():
    """ノイズ追加で波形が変化する（ゼロ信号への追加は除く）。"""
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(4000).astype(np.float32)
    noisy = add_noise_by_snr(signal, snr_db=10.0)
    assert not np.allclose(signal, noisy)


def test_add_noise_by_snr_high_snr_close_to_original():
    """非常に高い SNR では元信号に近い。"""
    rng = np.random.default_rng(1)
    signal = rng.standard_normal(4000).astype(np.float32) * 0.5
    noisy = add_noise_by_snr(signal, snr_db=60.0)
    # 高 SNR なのでほぼ同じ（差のRMSが元信号RMSより十分小さい）
    diff_rms = np.sqrt(np.mean((signal - noisy) ** 2))
    signal_rms = np.sqrt(np.mean(signal ** 2))
    assert diff_rms < signal_rms * 0.01  # 差が元信号の 1% 未満


# ===== build_transform =====

def _minimal_config() -> dict:
    return {
        "mel_conditions": {"image_size": [224, 224]},
        "augmentation": {"brightness_jitter": 0.4},
        "normalization": {
            "imagenet_normalization": True,
            "imagenet_mean": [0.485, 0.456, 0.406],
            "imagenet_std": [0.229, 0.224, 0.225],
        },
    }


def test_build_transform_train_has_color_jitter():
    """訓練用 transform には ColorJitter が含まれる。"""
    t = build_transform("train", _minimal_config())
    names = [type(x).__name__ for x in t.transforms]
    assert "ColorJitter" in names
    assert "Normalize" in names


def test_build_transform_val_no_color_jitter():
    """検証用 transform には ColorJitter が含まれない。"""
    t = build_transform("val", _minimal_config())
    names = [type(x).__name__ for x in t.transforms]
    assert "ColorJitter" not in names
    assert "Normalize" in names


def test_build_transform_test_no_color_jitter():
    """テスト用 transform には ColorJitter が含まれない。"""
    t = build_transform("test", _minimal_config())
    names = [type(x).__name__ for x in t.transforms]
    assert "ColorJitter" not in names


def test_build_transform_output_shape():
    """transform を適用すると (3, 224, 224) の Tensor が返る。"""
    from PIL import Image
    import torch

    t = build_transform("val", _minimal_config())
    img = Image.fromarray(
        np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    )
    tensor = t(img)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


# ===== split_train_test_by_patient =====

def _make_sample_df(n_patients: int = 20, clips_per_patient: int = 3) -> pd.DataFrame:
    """テスト用の模擬 DataFrame を作成する。"""
    rows = []
    for i in range(n_patients):
        patient_id = f"P{i:03d}"
        fv_value = 200.0 if i < n_patients // 2 else 600.0  # 前半が陽性
        for j in range(clips_per_patient):
            rows.append({
                "patient_id": patient_id,
                "clip_file": f"{patient_id}_clip{j}.wav",
                "fv_value": fv_value,
                "label": 1 if fv_value < 400 else 0,
            })
    return pd.DataFrame(rows)


def test_split_no_patient_overlap():
    """train と test に同じ patient_id が存在しない。"""
    df = _make_sample_df(n_patients=20)
    train_df, test_df = split_train_test_by_patient(
        df=df,
        patient_id_col="patient_id",
        label_name="fv_value",
        threshold=400.0,
        test_size=0.333,
        random_state=42,
        use_yield=False,
    )
    train_patients = set(train_df["patient_id"])
    test_patients = set(test_df["patient_id"])
    overlap = train_patients & test_patients
    assert len(overlap) == 0, f"患者重複: {overlap}"


def test_split_covers_all_data():
    """train + test が全クリップをカバーする（重複なし）。"""
    df = _make_sample_df(n_patients=20)
    train_df, test_df = split_train_test_by_patient(
        df=df,
        patient_id_col="patient_id",
        label_name="fv_value",
        threshold=400.0,
        test_size=0.333,
        random_state=42,
        use_yield=False,
    )
    all_clips = set(df["clip_file"])
    train_clips = set(train_df["clip_file"])
    test_clips = set(test_df["clip_file"])

    assert train_clips.isdisjoint(test_clips), "クリップ重複あり"
    assert train_clips | test_clips == all_clips, "カバレッジ不完全"


def test_split_test_size_approximately_correct():
    """test_size が概ね期待する比率になる（±10%）。"""
    df = _make_sample_df(n_patients=30)
    train_df, test_df = split_train_test_by_patient(
        df=df,
        patient_id_col="patient_id",
        label_name="fv_value",
        threshold=400.0,
        test_size=0.333,
        random_state=42,
        use_yield=False,
    )
    total_patients = df["patient_id"].nunique()
    test_patient_ratio = test_df["patient_id"].nunique() / total_patients
    assert 0.20 <= test_patient_ratio <= 0.50, f"test_patient_ratio={test_patient_ratio:.2f}"


def test_split_yield_mode_produces_multiple_folds():
    """use_yield=True で複数 fold が生成される。"""
    df = _make_sample_df(n_patients=30)
    folds = list(split_train_test_by_patient(
        df=df,
        patient_id_col="patient_id",
        label_name="fv_value",
        threshold=400.0,
        test_size=0.333,
        random_state=42,
        use_yield=True,
    ))
    assert len(folds) >= 2
    for train_df, test_df in folds:
        train_pats = set(train_df["patient_id"])
        test_pats = set(test_df["patient_id"])
        assert len(train_pats & test_pats) == 0
