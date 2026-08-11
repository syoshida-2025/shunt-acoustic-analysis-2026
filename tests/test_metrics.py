"""tests/test_metrics.py — utils/metrics.py のスモークテスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import (
    aggregate_clip_to_recording,
    compute_all_metrics,
    compute_auc_with_ci,
    compute_binary_metrics,
    find_youden_threshold,
)


# ===== aggregate_clip_to_recording =====

def test_aggregate_basic():
    """同音源の複数クリップが平均化される。"""
    clip_pred = pd.DataFrame({
        "clip_file": ["A0101_clip0.wav", "A0101_clip1.wav", "A0102_clip0.wav"],
        "true_label": [1, 1, 0],
        "pred_proba": [0.8, 0.6, 0.2],
    })
    clip_meta = pd.DataFrame({
        "clip_file": ["A0101_clip0.wav", "A0101_clip1.wav", "A0102_clip0.wav"],
        "audio_file": ["A0101.wav", "A0101.wav", "A0102.wav"],
        "patient_id": ["A01", "A01", "A01"],
    })
    result = aggregate_clip_to_recording(clip_pred, clip_meta)
    assert len(result) == 2
    a0101 = result[result["audio_file"] == "A0101.wav"].iloc[0]
    assert abs(a0101["pred_proba_mean"] - 0.7) < 1e-6
    assert a0101["n_clips"] == 2


def test_aggregate_single_clip_per_recording():
    """クリップが1枚だけの場合、pred_proba_mean = pred_proba。"""
    clip_pred = pd.DataFrame({
        "clip_file": ["A0101_clip0.wav"],
        "true_label": [1],
        "pred_proba": [0.75],
    })
    clip_meta = pd.DataFrame({
        "clip_file": ["A0101_clip0.wav"],
        "audio_file": ["A0101.wav"],
    })
    result = aggregate_clip_to_recording(clip_pred, clip_meta)
    assert len(result) == 1
    assert abs(result.iloc[0]["pred_proba_mean"] - 0.75) < 1e-6


def test_aggregate_missing_column_raises():
    """必須カラムが欠けると ValueError を出す。"""
    clip_pred = pd.DataFrame({
        "clip_file": ["A0101_clip0.wav"],
        "true_label": [1],
        # pred_proba が欠落
    })
    clip_meta = pd.DataFrame({
        "clip_file": ["A0101_clip0.wav"],
        "audio_file": ["A0101.wav"],
    })
    with pytest.raises(ValueError, match="必須カラム"):
        aggregate_clip_to_recording(clip_pred, clip_meta)


# ===== find_youden_threshold =====

def test_youden_threshold_perfect_separation():
    """完全分離データで Youden 閾値が 0.3 と 0.7 の間に設定される。"""
    y_true = [0, 0, 0, 1, 1, 1]
    y_proba = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    th = find_youden_threshold(y_true, y_proba)
    assert 0.3 < th <= 0.7


def test_youden_threshold_single_class():
    """片側クラスのみの場合、0.5 を返す。"""
    y_true = [1, 1, 1, 1]
    y_proba = [0.6, 0.7, 0.8, 0.9]
    assert find_youden_threshold(y_true, y_proba) == 0.5

    y_true_neg = [0, 0, 0]
    y_proba_neg = [0.1, 0.2, 0.3]
    assert find_youden_threshold(y_true_neg, y_proba_neg) == 0.5


# ===== compute_binary_metrics =====

def test_compute_binary_metrics_basic():
    """混同行列の基本ケース。TP=2, FN=1, TN=2, FP=1。"""
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 1, 0, 0, 0, 1]
    m = compute_binary_metrics(y_true, y_pred)
    assert m["n_tp"] == 2
    assert m["n_fn"] == 1
    assert m["n_tn"] == 2
    assert m["n_fp"] == 1
    assert m["sensitivity"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(2 / 3)
    assert m["ppv"] == pytest.approx(2 / 3)
    assert m["npv"] == pytest.approx(2 / 3)


def test_compute_binary_metrics_all_correct():
    """完全正解の場合、全指標が 1.0。"""
    y_true = [1, 1, 0, 0]
    y_pred = [1, 1, 0, 0]
    m = compute_binary_metrics(y_true, y_pred)
    assert m["sensitivity"] == pytest.approx(1.0)
    assert m["specificity"] == pytest.approx(1.0)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)


def test_compute_binary_metrics_no_positive_pred():
    """全て陰性と予測した場合、PPV は NaN。"""
    y_true = [1, 1, 0, 0]
    y_pred = [0, 0, 0, 0]
    m = compute_binary_metrics(y_true, y_pred)
    assert np.isnan(m["ppv"])
    assert m["sensitivity"] == pytest.approx(0.0)
    assert m["specificity"] == pytest.approx(1.0)


# ===== compute_auc_with_ci =====

def test_auc_with_ci_single_class():
    """片側クラスのみで NaN を返す。"""
    y_true = [1, 1, 1, 1]
    y_proba = [0.6, 0.7, 0.8, 0.9]
    result = compute_auc_with_ci(y_true, y_proba)
    assert np.isnan(result["auc"])
    assert np.isnan(result["auc_ci_lower"])
    assert np.isnan(result["auc_ci_upper"])
    assert np.isnan(result["auc_se"])


def test_auc_with_ci_perfect_classifier_delong():
    """完全分類器で AUC=1.0 (DeLong)。"""
    y_true = [0, 0, 0, 1, 1, 1]
    y_proba = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    result = compute_auc_with_ci(y_true, y_proba, method="delong")
    assert result["auc"] == pytest.approx(1.0)
    assert 0.0 <= result["auc_ci_lower"] <= 1.0
    assert 0.0 <= result["auc_ci_upper"] <= 1.0


def test_auc_with_ci_perfect_classifier_bootstrap():
    """完全分類器で AUC=1.0 (bootstrap)。"""
    y_true = [0, 0, 0, 1, 1, 1]
    y_proba = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    result = compute_auc_with_ci(y_true, y_proba, method="bootstrap", n_bootstrap=200)
    assert result["auc"] == pytest.approx(1.0)


def test_auc_with_ci_ci_bounds():
    """CI は [0, 1] の範囲内。"""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=50).tolist()
    y_proba = (rng.random(50) * 0.4 + 0.3).tolist()
    result = compute_auc_with_ci(y_true, y_proba, method="delong")
    if not np.isnan(result["auc"]):
        assert result["auc_ci_lower"] >= 0.0
        assert result["auc_ci_upper"] <= 1.0
        assert result["auc_ci_lower"] <= result["auc_ci_upper"]


# ===== compute_all_metrics =====

def test_compute_all_metrics_returns_expected_keys():
    """compute_all_metrics が期待されるキーを含む辞書を返す。"""
    clip_pred = pd.DataFrame({
        "clip_file": [f"A010{i}_clip0.wav" for i in range(6)],
        "patient_id": [f"A0{i}" for i in range(6)],
        "true_label": [1, 1, 1, 0, 0, 0],
        "pred_proba": [0.8, 0.7, 0.6, 0.3, 0.2, 0.1],
    })
    clip_meta = pd.DataFrame({
        "clip_file": [f"A010{i}_clip0.wav" for i in range(6)],
        "audio_file": [f"A010{i}.wav" for i in range(6)],
        "patient_id": [f"A0{i}" for i in range(6)],
    })
    config = {"evaluation": {"threshold_method": "youden"}}

    result = compute_all_metrics(clip_pred, clip_meta, config)

    expected_keys = [
        "auc_clip", "auc_clip_ci_lower", "auc_recording",
        "f1_clip", "sensitivity_clip", "specificity_clip",
        "f1_recording", "sensitivity_recording",
        "youden_threshold_clip", "youden_threshold_recording",
        "n_test_clips", "n_test_recordings",
    ]
    for key in expected_keys:
        assert key in result, f"キー '{key}' が結果にありません"


def test_compute_all_metrics_perfect_classifier():
    """完全分類器の場合、AUC=1.0。"""
    clip_pred = pd.DataFrame({
        "clip_file": [f"A010{i}_clip0.wav" for i in range(6)],
        "patient_id": [f"A0{i}" for i in range(6)],
        "true_label": [1, 1, 1, 0, 0, 0],
        "pred_proba": [0.9, 0.85, 0.8, 0.2, 0.15, 0.1],
    })
    clip_meta = pd.DataFrame({
        "clip_file": [f"A010{i}_clip0.wav" for i in range(6)],
        "audio_file": [f"A010{i}.wav" for i in range(6)],
    })
    config = {"evaluation": {"threshold_method": "youden"}}

    result = compute_all_metrics(clip_pred, clip_meta, config)
    assert result["auc_clip"] == pytest.approx(1.0)
    assert result["auc_recording"] == pytest.approx(1.0)
