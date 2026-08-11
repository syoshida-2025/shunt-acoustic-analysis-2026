"""utils/metrics.py — 評価指標、閾値検索、音源集約、CI 算出。

主な公開関数:
    - aggregate_clip_to_recording: クリップレベル予測 → 音源レベル予測（平均化）。
    - find_youden_threshold: Youden index 最適閾値を返す。
    - compute_binary_metrics: 2 値分類指標を一括計算 (Sens, Spec, PPV, NPV, F1, Acc)。
    - compute_auc_with_ci: AUC と 95% CI を返す（DeLong 法 or bootstrap）。
    - compute_all_metrics: train_model.py から呼ばれる統合関数。

設計方針:
    - 片側クラスのみのデータでも例外を出さず NaN を返す（CV の特殊ケース対応）。
    - sklearn.metrics の関数を内部で使用。
    - すべての関数は純粋関数（副作用なし）で実装。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


# ===== 音源集約 =====

def aggregate_clip_to_recording(
    clip_predictions: pd.DataFrame,
    clip_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """クリップレベルの予測確率を音源レベルに集約する。

    同じ audio_file に属する複数クリップの予測確率を平均化し、
    1音源 = 1サンプルとして扱えるようにする。

    Args:
        clip_predictions: DataFrame
            必須カラム: clip_file, true_label, pred_proba
        clip_metadata: DataFrame
            必須カラム: clip_file, audio_file

    Returns:
        音源レベル DataFrame:
            - audio_file: str
            - true_label: int (同音源の全クリップは同じラベル)
            - pred_proba_mean: float (クリップ予測確率の平均)
            - n_clips: int (集約に使ったクリップ数)
            - patient_id: str (オプション、clip_metadata に含まれる場合)

    Raises:
        ValueError: clip_predictions に必須カラムがない、または
                   集約結果が空になる場合。
    """
    required_pred_cols = {"clip_file", "true_label", "pred_proba"}
    if not required_pred_cols.issubset(clip_predictions.columns):
        missing = required_pred_cols - set(clip_predictions.columns)
        raise ValueError(f"clip_predictions に必須カラムがありません: {missing}")

    required_meta_cols = {"clip_file", "audio_file"}
    if not required_meta_cols.issubset(clip_metadata.columns):
        missing = required_meta_cols - set(clip_metadata.columns)
        raise ValueError(f"clip_metadata に必須カラムがありません: {missing}")

    merge_cols = ["clip_file", "audio_file"]
    if "patient_id" in clip_metadata.columns:
        merge_cols.append("patient_id")

    merged = clip_predictions.merge(
        clip_metadata[merge_cols],
        on="clip_file",
        how="inner",
    )

    if len(merged) == 0:
        raise ValueError(
            "クリップ予測とメタデータの結合結果が空です。"
            "clip_file の値が一致しているか確認してください。"
        )

    agg_dict: dict[str, Any] = {
        "true_label": "first",
        "pred_proba": "mean",
        "clip_file": "count",
    }
    if "patient_id" in merged.columns:
        agg_dict["patient_id"] = "first"

    recording_pred = merged.groupby("audio_file").agg(agg_dict).reset_index()

    recording_pred = recording_pred.rename(columns={
        "pred_proba": "pred_proba_mean",
        "clip_file": "n_clips",
    })

    return recording_pred


# ===== Youden 閾値検索 =====

def find_youden_threshold(
    y_true: np.ndarray | list[int],
    y_proba: np.ndarray | list[float],
) -> float:
    """Youden index (J = sensitivity + specificity - 1) を最大化する閾値を返す。

    Args:
        y_true: 真のラベル (0 or 1)。
        y_proba: 陽性クラスの予測確率。

    Returns:
        最適閾値 (float)。片側クラスのみの場合は 0.5 を返す。
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if len(np.unique(y_true)) < 2:
        return 0.5

    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_threshold = float(thresholds[best_idx])

    # sklearn の roc_curve は先頭に無限大を返すことがある
    if np.isinf(best_threshold):
        if len(thresholds) > 1:
            best_threshold = float(thresholds[1])
        else:
            best_threshold = 0.5

    return best_threshold


# ===== AUC + 95% CI =====

def compute_auc_with_ci(
    y_true: np.ndarray | list[int],
    y_proba: np.ndarray | list[float],
    method: str = "delong",
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> dict[str, float]:
    """AUC と 95% CI を計算する。

    Args:
        y_true: 真のラベル (0 or 1)。
        y_proba: 陽性クラスの予測確率。
        method: "delong" または "bootstrap"。
        n_bootstrap: bootstrap 時のリサンプリング回数。
        ci_level: 信頼区間の水準（通常 0.95）。
        random_state: bootstrap 時の seed。

    Returns:
        辞書:
            - "auc": float
            - "auc_ci_lower": float
            - "auc_ci_upper": float
            - "auc_se": float

        片側クラスのみの場合は全て NaN。
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if len(np.unique(y_true)) < 2:
        return {
            "auc": float("nan"),
            "auc_ci_lower": float("nan"),
            "auc_ci_upper": float("nan"),
            "auc_se": float("nan"),
        }

    auc = float(roc_auc_score(y_true, y_proba))

    if method == "bootstrap":
        rng = np.random.default_rng(random_state)
        n = len(y_true)
        bootstrap_aucs = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            if len(np.unique(y_true[idx])) < 2:
                continue
            bootstrap_aucs.append(roc_auc_score(y_true[idx], y_proba[idx]))

        if len(bootstrap_aucs) < 10:
            return {
                "auc": auc,
                "auc_ci_lower": float("nan"),
                "auc_ci_upper": float("nan"),
                "auc_se": float("nan"),
            }

        bootstrap_aucs_arr = np.array(bootstrap_aucs)
        alpha = (1 - ci_level) / 2
        ci_lower = float(np.quantile(bootstrap_aucs_arr, alpha))
        ci_upper = float(np.quantile(bootstrap_aucs_arr, 1 - alpha))
        se = float(np.std(bootstrap_aucs_arr, ddof=1))

    else:  # method == "delong" (Hanley-McNeil SE 近似)
        n_pos = int(np.sum(y_true == 1))
        n_neg = int(np.sum(y_true == 0))
        q1 = auc / (2 - auc)
        q2 = 2 * auc**2 / (1 + auc)
        variance = (
            auc * (1 - auc)
            + (n_pos - 1) * (q1 - auc**2)
            + (n_neg - 1) * (q2 - auc**2)
        ) / (n_pos * n_neg)
        se = float(np.sqrt(max(0.0, variance)))
        z = 1.96
        ci_lower = float(max(0.0, auc - z * se))
        ci_upper = float(min(1.0, auc + z * se))

    return {
        "auc": auc,
        "auc_ci_lower": ci_lower,
        "auc_ci_upper": ci_upper,
        "auc_se": se,
    }


# ===== 二値分類指標 =====

def compute_binary_metrics(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
) -> dict[str, float]:
    """二値分類指標を一括計算する。

    Args:
        y_true: 真のラベル (0 or 1)。
        y_pred: 予測ラベル (0 or 1)。閾値適用後。

    Returns:
        辞書:
            - "sensitivity", "specificity", "ppv", "npv", "accuracy", "f1"
            - "n_tp", "n_fp", "n_tn", "n_fn"
            - "n_positive", "n_negative"

        分母がゼロになる場合は NaN を返す。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    n_positive = int(np.sum(y_true == 1))
    n_negative = int(np.sum(y_true == 0))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    accuracy = float(accuracy_score(y_true, y_pred))

    if not np.isnan(ppv) and not np.isnan(sensitivity) and (ppv + sensitivity) > 0:
        f1 = 2 * ppv * sensitivity / (ppv + sensitivity)
    else:
        f1 = float("nan")

    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "n_tp": tp,
        "n_fp": fp,
        "n_tn": tn,
        "n_fn": fn,
        "n_positive": n_positive,
        "n_negative": n_negative,
    }


# ===== 統合関数: train_model.py から呼ばれる =====

def compute_all_metrics(
    predictions: pd.DataFrame,
    clip_metadata: pd.DataFrame,
    config: dict,
) -> dict[str, Any]:
    """1 fold 分のテスト予測値から、全ての評価指標を計算する。

    Args:
        predictions: テスト DataLoader からの予測結果 DataFrame
            必須カラム: clip_file, patient_id, true_label, pred_proba
        clip_metadata: 全クリップメタデータ
            必須カラム: clip_file, audio_file
        config: 全 config。

    Returns:
        all_metrics.csv の 1 行分に対応する辞書。
        クリップレベル・音源レベルそれぞれの AUC / F1 / Sens / Spec / PPV / NPV / Acc を含む。
    """
    # ===== クリップレベル =====
    clip_y_true = predictions["true_label"].values
    clip_y_proba = predictions["pred_proba"].values

    clip_auc = compute_auc_with_ci(clip_y_true, clip_y_proba, method="delong")

    clip_youden_th = find_youden_threshold(clip_y_true, clip_y_proba)
    clip_y_pred_youden = (clip_y_proba >= clip_youden_th).astype(int)
    clip_metrics_youden = compute_binary_metrics(clip_y_true, clip_y_pred_youden)

    clip_y_pred_05 = (clip_y_proba >= 0.5).astype(int)
    clip_metrics_05 = compute_binary_metrics(clip_y_true, clip_y_pred_05)

    # ===== 音源レベル =====
    recording_pred = aggregate_clip_to_recording(predictions, clip_metadata)
    rec_y_true = recording_pred["true_label"].values
    rec_y_proba = recording_pred["pred_proba_mean"].values

    rec_auc = compute_auc_with_ci(rec_y_true, rec_y_proba, method="delong")

    rec_youden_th = find_youden_threshold(rec_y_true, rec_y_proba)
    rec_y_pred_youden = (rec_y_proba >= rec_youden_th).astype(int)
    rec_metrics_youden = compute_binary_metrics(rec_y_true, rec_y_pred_youden)

    rec_y_pred_05 = (rec_y_proba >= 0.5).astype(int)
    rec_metrics_05 = compute_binary_metrics(rec_y_true, rec_y_pred_05)

    return {
        # ===== クリップレベル =====
        "auc_clip": clip_auc["auc"],
        "auc_clip_ci_lower": clip_auc["auc_ci_lower"],
        "auc_clip_ci_upper": clip_auc["auc_ci_upper"],
        "auc_clip_se": clip_auc["auc_se"],
        "youden_threshold_clip": clip_youden_th,
        "f1_clip": clip_metrics_youden["f1"],
        "sensitivity_clip": clip_metrics_youden["sensitivity"],
        "specificity_clip": clip_metrics_youden["specificity"],
        "ppv_clip": clip_metrics_youden["ppv"],
        "npv_clip": clip_metrics_youden["npv"],
        "accuracy_clip": clip_metrics_youden["accuracy"],
        "f1_clip_th05": clip_metrics_05["f1"],
        "accuracy_clip_th05": clip_metrics_05["accuracy"],
        "n_test_clips": len(predictions),
        "n_test_clips_positive": clip_metrics_youden["n_positive"],
        "n_test_clips_negative": clip_metrics_youden["n_negative"],
        # ===== 音源レベル（主指標） =====
        "auc_recording": rec_auc["auc"],
        "auc_recording_ci_lower": rec_auc["auc_ci_lower"],
        "auc_recording_ci_upper": rec_auc["auc_ci_upper"],
        "auc_recording_se": rec_auc["auc_se"],
        "youden_threshold_recording": rec_youden_th,
        "f1_recording": rec_metrics_youden["f1"],
        "sensitivity_recording": rec_metrics_youden["sensitivity"],
        "specificity_recording": rec_metrics_youden["specificity"],
        "ppv_recording": rec_metrics_youden["ppv"],
        "npv_recording": rec_metrics_youden["npv"],
        "accuracy_recording": rec_metrics_youden["accuracy"],
        "f1_recording_th05": rec_metrics_05["f1"],
        "accuracy_recording_th05": rec_metrics_05["accuracy"],
        "n_test_recordings": len(recording_pred),
        "n_test_recordings_positive": rec_metrics_youden["n_positive"],
        "n_test_recordings_negative": rec_metrics_youden["n_negative"],
    }
