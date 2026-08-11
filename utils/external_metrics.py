"""utils/external_metrics.py

外部検証パイプライン専用の指標計算。

内部検証の utils/metrics.py:compute_all_metrics は、評価対象のデータ自身から
Youden閾値を事後決定する設計になっている（テスト分割内での閾値最適化）。
これは主指標であるAUCには影響しないが、二次指標（F1・感度・特異度等）を
楽観的にする可能性がある。

外部検証では、この問題を避けるため「開発データ側で決定した閾値をロックし、
外部データにはその値をそのまま適用するだけで、外部データから閾値を再計算しない」
という設計にする。本モジュールはそのための関数を提供する。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .metrics import (
    aggregate_clip_to_recording,
    compute_auc_with_ci,
    compute_binary_metrics,
    find_youden_threshold,
)


def lock_thresholds_from_oof_predictions(
    oof_predictions: pd.DataFrame,
    clip_metadata: pd.DataFrame,
) -> dict[str, float]:
    """開発データのout-of-fold予測から、クリップ・音源レベルのYouden閾値を決定する。

    HP探索の3-fold CVにおいて、選定されたHPで学習した3つのモデルそれぞれが
    「自分が学習に使わなかった1foldの患者」に対して出した予測を1つにプールしたもの
    （out-of-fold予測。全開発患者について、学習に使われていないモデルからの予測が
    1つずつ揃う）を入力として、Youden閾値を決定する。

    この閾値は「開発データのみ」から決まっており、外部データには一切依存しない。
    以降、evaluate_with_locked_threshold に渡して外部データへ固定適用する。

    Args:
        oof_predictions: out-of-fold予測 DataFrame。
            必須カラム: clip_file, true_label, pred_proba
        clip_metadata: 開発データのクリップメタデータ（clip_file, audio_file, patient_id）

    Returns:
        {"clip_threshold": float, "recording_threshold": float}
    """
    clip_threshold = find_youden_threshold(
        oof_predictions["true_label"].values,
        oof_predictions["pred_proba"].values,
    )

    recording_pred = aggregate_clip_to_recording(oof_predictions, clip_metadata)
    recording_threshold = find_youden_threshold(
        recording_pred["true_label"].values,
        recording_pred["pred_proba_mean"].values,
    )

    return {
        "clip_threshold": float(clip_threshold),
        "recording_threshold": float(recording_threshold),
    }


BOOTSTRAP_METRIC_KEYS = ("sensitivity", "specificity", "ppv", "npv", "accuracy", "f1")


def bootstrap_binary_metrics_ci(
    y_true: np.ndarray | list[int],
    y_proba: np.ndarray | list[float],
    threshold: float,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    """固定閾値での二値指標(感度・特異度・PPV・NPV・正確度・F1)の95%CIとブートストラップ
    標準誤差(SE)を、percentile bootstrapで算出する。

    AUCはDeLong法で解析式によりSE・CIを求めているが、二値指標は固定閾値での
    混同行列に基づく比率のため、ここではリサンプリングでSE・CIを推定する
    (ブートストラップ分布の標準偏差は、その点推定値のSEの推定値として扱う。
    「個々のデータのばらつき」を表すSDとは概念が異なるため、キー名は "se" とする)。
    サンプル(音源またはクリップ)を復元抽出し、そのつど同じ閾値で指標を
    再計算する。

    Args:
        y_true: 真のラベル(0 or 1)
        y_proba: 予測確率
        threshold: ロック済みの固定閾値(リサンプリングの度に再計算しない)
        n_bootstrap: リサンプリング回数
        ci_level: 信頼区間の水準
        random_state: 乱数シード

    Returns:
        {"sensitivity": {"ci_lower":..., "ci_upper":..., "se":..., "n_valid_bootstrap":...}, ...}
        有効なリサンプリングが10回未満の指標はNaNを返す(陽性/陰性どちらかが
        0件になるリサンプリングが多発する、極端に小さいサンプルの場合)。
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    rng = np.random.default_rng(random_state)

    samples: dict[str, list[float]] = {key: [] for key in BOOTSTRAP_METRIC_KEYS}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_true_bs = y_true[idx]
        if len(np.unique(y_true_bs)) < 2:
            continue
        y_pred_bs = (y_proba[idx] >= threshold).astype(int)
        metrics_bs = compute_binary_metrics(y_true_bs, y_pred_bs)
        for key in BOOTSTRAP_METRIC_KEYS:
            value = metrics_bs[key]
            if not np.isnan(value):
                samples[key].append(value)

    alpha = (1 - ci_level) / 2
    result: dict[str, dict[str, float]] = {}
    for key, values in samples.items():
        if len(values) < 10:
            result[key] = {
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "se": float("nan"),
                "n_valid_bootstrap": len(values),
            }
        else:
            arr = np.array(values)
            result[key] = {
                "ci_lower": float(np.quantile(arr, alpha)),
                "ci_upper": float(np.quantile(arr, 1 - alpha)),
                "se": float(np.std(arr, ddof=1)),
                "n_valid_bootstrap": len(values),
            }
    return result


def evaluate_with_locked_threshold(
    predictions: pd.DataFrame,
    clip_metadata: pd.DataFrame,
    clip_threshold: float,
    recording_threshold: float,
) -> dict[str, Any]:
    """ロック済みの閾値を用いて、クリップ・音源レベルの評価指標を算出する。

    utils.metrics.compute_all_metrics と異なり、Youden閾値を渡されたデータ自身から
    再計算しない。外部データを評価する際は、開発データ側で
    lock_thresholds_from_oof_predictions により決定した閾値をそのまま渡すこと。

    Args:
        predictions: 評価対象の予測 DataFrame。
            必須カラム: clip_file, patient_id, true_label, pred_proba
        clip_metadata: 評価対象データのクリップメタデータ（clip_file, audio_file, patient_id）
        clip_threshold: ロック済みのクリップレベル閾値
        recording_threshold: ロック済みの音源レベル閾値

    Returns:
        compute_all_metrics とほぼ同じキー構成の辞書（"locked_threshold" のprefixが付く
        二値指標のみが異なり、Youden閾値そのものは引数の値がそのまま記録される）。
    """
    clip_y_true = predictions["true_label"].values
    clip_y_proba = predictions["pred_proba"].values

    clip_auc = compute_auc_with_ci(clip_y_true, clip_y_proba, method="delong")
    clip_y_pred = (clip_y_proba >= clip_threshold).astype(int)
    clip_metrics = compute_binary_metrics(clip_y_true, clip_y_pred)
    clip_ci = bootstrap_binary_metrics_ci(clip_y_true, clip_y_proba, clip_threshold)

    recording_pred = aggregate_clip_to_recording(predictions, clip_metadata)
    rec_y_true = recording_pred["true_label"].values
    rec_y_proba = recording_pred["pred_proba_mean"].values

    rec_auc = compute_auc_with_ci(rec_y_true, rec_y_proba, method="delong")
    rec_y_pred = (rec_y_proba >= recording_threshold).astype(int)
    rec_metrics = compute_binary_metrics(rec_y_true, rec_y_pred)
    rec_ci = bootstrap_binary_metrics_ci(rec_y_true, rec_y_proba, recording_threshold)

    result = {
        # ===== クリップレベル =====
        "auc_clip": clip_auc["auc"],
        "auc_clip_ci_lower": clip_auc["auc_ci_lower"],
        "auc_clip_ci_upper": clip_auc["auc_ci_upper"],
        "auc_clip_se": clip_auc["auc_se"],
        "locked_clip_threshold": float(clip_threshold),
        "f1_clip": clip_metrics["f1"],
        "sensitivity_clip": clip_metrics["sensitivity"],
        "specificity_clip": clip_metrics["specificity"],
        "ppv_clip": clip_metrics["ppv"],
        "npv_clip": clip_metrics["npv"],
        "accuracy_clip": clip_metrics["accuracy"],
        "n_clips": len(predictions),
        # ===== 音源レベル(主指標) =====
        "auc_recording": rec_auc["auc"],
        "auc_recording_ci_lower": rec_auc["auc_ci_lower"],
        "auc_recording_ci_upper": rec_auc["auc_ci_upper"],
        "auc_recording_se": rec_auc["auc_se"],
        "locked_recording_threshold": float(recording_threshold),
        "f1_recording": rec_metrics["f1"],
        "sensitivity_recording": rec_metrics["sensitivity"],
        "specificity_recording": rec_metrics["specificity"],
        "ppv_recording": rec_metrics["ppv"],
        "npv_recording": rec_metrics["npv"],
        "accuracy_recording": rec_metrics["accuracy"],
        "n_recordings": len(recording_pred),
    }

    # ブートストラップ95%CI・SE(recording-levelは"_recording_ci_lower/upper/se"、
    # clip-levelは"_clip_ci_lower/upper/se"というキー名で追加する)。
    # AUCのauc_*_seと同じ「点推定値の標準誤差」という統計的な役割。
    for key in BOOTSTRAP_METRIC_KEYS:
        result[f"{key}_clip_ci_lower"] = clip_ci[key]["ci_lower"]
        result[f"{key}_clip_ci_upper"] = clip_ci[key]["ci_upper"]
        result[f"{key}_clip_se"] = clip_ci[key]["se"]
        result[f"{key}_recording_ci_lower"] = rec_ci[key]["ci_lower"]
        result[f"{key}_recording_ci_upper"] = rec_ci[key]["ci_upper"]
        result[f"{key}_recording_se"] = rec_ci[key]["se"]

    return result
