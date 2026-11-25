"""tests/test_external_metrics.py — utils/external_metrics.py のスモークテスト。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.external_metrics import (
    bootstrap_binary_metrics_ci,
    evaluate_with_locked_threshold,
    lock_thresholds_from_oof_predictions,
)


def _make_clip_metadata(n_recordings: int, clips_per_recording: int = 2) -> pd.DataFrame:
    rows = []
    for i in range(n_recordings):
        audio_file = f"A{i:03d}.wav"
        for c in range(clips_per_recording):
            rows.append({
                "clip_file": f"A{i:03d}_clip{c}.wav",
                "audio_file": audio_file,
                "patient_id": f"P{i:03d}",
            })
    return pd.DataFrame(rows)


def _make_predictions(clip_metadata: pd.DataFrame, true_label_by_recording: dict, proba_by_recording: dict) -> pd.DataFrame:
    rows = []
    for _, row in clip_metadata.iterrows():
        rows.append({
            "clip_file": row["clip_file"],
            "true_label": true_label_by_recording[row["audio_file"]],
            "pred_proba": proba_by_recording[row["audio_file"]],
        })
    return pd.DataFrame(rows)


def test_lock_thresholds_from_oof_predictions_perfect_separation() -> None:
    clip_metadata = _make_clip_metadata(n_recordings=8)
    true_label = {f"A{i:03d}.wav": int(i >= 4) for i in range(8)}
    # 完全に分離可能: 陽性は0.9台、陰性は0.1台
    proba = {f"A{i:03d}.wav": (0.9 if i >= 4 else 0.1) for i in range(8)}
    predictions = _make_predictions(clip_metadata, true_label, proba)

    thresholds = lock_thresholds_from_oof_predictions(predictions, clip_metadata)

    assert 0.1 < thresholds["clip_threshold"] <= 0.9
    assert 0.1 < thresholds["recording_threshold"] <= 0.9


def test_evaluate_with_locked_threshold_uses_supplied_threshold_not_recomputed() -> None:
    """渡した閾値がそのまま使われ、データ自身から再計算されないことを確認する。"""
    clip_metadata = _make_clip_metadata(n_recordings=8)
    true_label = {f"A{i:03d}.wav": int(i >= 4) for i in range(8)}
    proba = {f"A{i:03d}.wav": (0.9 if i >= 4 else 0.1) for i in range(8)}
    predictions = _make_predictions(clip_metadata, true_label, proba)

    # 通常のYouden閾値なら 0.1〜0.9 の間になるはずだが、意図的に極端な閾値(0.99)を渡す。
    metrics = evaluate_with_locked_threshold(
        predictions=predictions,
        clip_metadata=clip_metadata,
        clip_threshold=0.99,
        recording_threshold=0.99,
    )

    # 閾値0.99なら、pred_proba=0.9の陽性例は全て陰性判定になり、感度は0になるはず。
    assert metrics["sensitivity_recording"] == 0.0
    assert metrics["locked_recording_threshold"] == 0.99
    assert metrics["locked_clip_threshold"] == 0.99
    # AUCは閾値に依存しないため、判別可能な並びであれば1.0に近いはず。
    assert metrics["auc_recording"] > 0.9


def test_bootstrap_binary_metrics_ci_contains_point_estimate() -> None:
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 20 + [1] * 20)
    # 完全ではないが分離可能なスコア
    y_proba = np.concatenate([
        rng.normal(0.3, 0.15, 20),
        rng.normal(0.7, 0.15, 20),
    ])
    threshold = 0.5

    ci = bootstrap_binary_metrics_ci(y_true, y_proba, threshold, n_bootstrap=200, random_state=0)

    for key in ("sensitivity", "specificity", "ppv", "npv", "accuracy", "f1"):
        assert key in ci
        assert ci[key]["n_valid_bootstrap"] >= 10
        assert 0.0 <= ci[key]["ci_lower"] <= ci[key]["ci_upper"] <= 1.0
        assert ci[key]["se"] >= 0.0


def test_bootstrap_binary_metrics_ci_handles_tiny_sample() -> None:
    """陽性・陰性が極端に少ない場合、有効なリサンプリングが少なくCIがNaNになり得ることを確認する。"""
    y_true = np.array([0, 1])
    y_proba = np.array([0.2, 0.8])

    ci = bootstrap_binary_metrics_ci(y_true, y_proba, threshold=0.5, n_bootstrap=50, random_state=0)

    for key in ("sensitivity", "specificity", "ppv", "npv", "accuracy", "f1"):
        assert key in ci
        # NaNになっていてもクラッシュしないことが重要(有効試行数が少ない場合はNaNを許容)
        assert isinstance(ci[key]["ci_lower"], float)


def test_evaluate_with_locked_threshold_includes_bootstrap_ci_keys() -> None:
    clip_metadata = _make_clip_metadata(n_recordings=10)
    true_label = {f"A{i:03d}.wav": int(i >= 5) for i in range(10)}
    proba = {f"A{i:03d}.wav": (0.8 if i >= 5 else 0.2) for i in range(10)}
    predictions = _make_predictions(clip_metadata, true_label, proba)

    metrics = evaluate_with_locked_threshold(
        predictions=predictions,
        clip_metadata=clip_metadata,
        clip_threshold=0.5,
        recording_threshold=0.5,
    )

    for key in ("sensitivity", "specificity", "ppv", "npv", "accuracy", "f1"):
        assert f"{key}_recording_ci_lower" in metrics
        assert f"{key}_recording_ci_upper" in metrics
        assert f"{key}_recording_se" in metrics
        assert f"{key}_clip_ci_lower" in metrics
        assert f"{key}_clip_ci_upper" in metrics
        assert f"{key}_clip_se" in metrics

    assert "auc_recording_se" in metrics
    assert "auc_clip_se" in metrics


def test_evaluate_with_locked_threshold_returns_expected_keys() -> None:
    clip_metadata = _make_clip_metadata(n_recordings=6)
    true_label = {f"A{i:03d}.wav": int(i >= 3) for i in range(6)}
    proba = {f"A{i:03d}.wav": (0.8 if i >= 3 else 0.2) for i in range(6)}
    predictions = _make_predictions(clip_metadata, true_label, proba)

    metrics = evaluate_with_locked_threshold(
        predictions=predictions,
        clip_metadata=clip_metadata,
        clip_threshold=0.5,
        recording_threshold=0.5,
    )

    expected_keys = {
        "auc_clip", "auc_clip_ci_lower", "auc_clip_ci_upper", "locked_clip_threshold",
        "f1_clip", "sensitivity_clip", "specificity_clip", "ppv_clip", "npv_clip",
        "accuracy_clip", "n_clips",
        "auc_recording", "auc_recording_ci_lower", "auc_recording_ci_upper",
        "locked_recording_threshold", "f1_recording", "sensitivity_recording",
        "specificity_recording", "ppv_recording", "npv_recording", "accuracy_recording",
        "n_recordings",
    }
    assert expected_keys.issubset(metrics.keys())
    assert metrics["n_recordings"] == 6
