"""scripts/train_model.py

メイン学習スクリプト。

27 条件 × 30 分割 = 810 モデルを学習し、各分割の予測値と指標を保存する。
チェックポイント機構 (results/status.json) により、中断・再開が可能。
ハイブリッドストレージ方針により、1 条件分の画像のみをディスク上に保持する。

実行例:
    uv run python scripts/train_model.py --config config.yaml

主要な出力:
    - results/predictions/{condition_name}.csv     (各条件、追記モード)
    - results/metrics/all_metrics.csv              (全条件累積、追記モード)
    - results/loss_curves/{condition_name}/repeat{R}_fold{F}.csv
    - results/status.json                          (進捗管理)
    - results/run.log                              (実行ログ)
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.checkpoint import (
    compute_config_hash,
    is_condition_complete,
    is_split_complete,
    load_or_init_status,
    mark_condition_failed,
    mark_split_complete,
    print_status_summary,
    save_status,
)
from utils.data_split import split_train_test_by_patient
from utils.dataset import build_dataloader
from utils.metadata_io import read_clip_metadata
from utils.metrics import compute_all_metrics
from utils.trainer import derive_fold_seed, predict_on_loader, train_one_fold


# ===== 例外種別 =====

class FatalError(Exception):
    """致命的エラー：即座に学習を停止すべきエラー。"""


class TransientError(Exception):
    """軽微エラー：記録して次の分割に進むエラー。"""


FATAL_EXCEPTION_TYPES = (
    FileNotFoundError,
    KeyError,
    FatalError,
)


# ===== エントリーポイント =====

def main(config_path: str) -> int:
    """学習スクリプトのメイン関数。"""
    config = load_config(config_path)

    setup_logging(config)
    log_run_start(config_path, config)

    splits_data = load_splits(config)
    clip_metadata = read_clip_metadata(config["paths"]["clip_metadata_csv"])

    condition_names = generate_condition_names(config)
    if len(condition_names) != 27:
        print(
            f"WARNING: 条件数が 27 ではありません ({len(condition_names)})。",
            file=sys.stderr,
        )

    status = load_or_init_status(
        status_path=config["execution"]["status_json_path"],
        config_path=config_path,
        condition_names=condition_names,
        n_splits_per_condition=len(splits_data["splits"]),
    )
    print_status_summary(status)

    save_config_snapshot(config_path, config)

    n_soft_errors = 0
    max_soft_errors = config["execution"].get("max_soft_errors", 3)

    for condition_name in condition_names:
        if is_condition_complete(condition_name, status):
            print(f"[skip] 条件 {condition_name} は完了済み")
            continue

        print(f"\n{'=' * 60}")
        print(f"条件: {condition_name}")
        print(f"{'=' * 60}")

        try:
            ensure_images_generated(condition_name, config_path, config)

            for split in splits_data["splits"]:
                repeat_id = split["repeat_id"]
                fold_id = split["fold_id"]

                if is_split_complete(condition_name, repeat_id, fold_id, status):
                    continue

                split_label = f"repeat{repeat_id:02d}_fold{fold_id}"
                print(f"\n  [{condition_name}] {split_label} 学習開始...")
                t0 = time.time()

                try:
                    result = train_one_split(
                        condition_name=condition_name,
                        split=split,
                        clip_metadata=clip_metadata,
                        config=config,
                    )

                    save_predictions(result["predictions"], condition_name, config)
                    save_metrics(result["metrics"], condition_name, split, config)
                    save_loss_curve(result["history"], condition_name, split_label, config)

                    if should_save_model(condition_name, config):
                        save_model_weights(
                            result["model_state_dict"], condition_name, split_label, config_path, config
                        )

                    mark_split_complete(status, condition_name, repeat_id, fold_id)
                    save_status(status, config["execution"]["status_json_path"])

                    elapsed = time.time() - t0
                    best_val_auc = result["metrics"].get("best_val_auc", float("nan"))
                    auc_str = f"{best_val_auc:.4f}" if not pd.isna(best_val_auc) else "N/A"
                    print(
                        f"  [{condition_name}] {split_label} 完了 "
                        f"({elapsed:.1f}秒, best_val_auc={auc_str})"
                    )

                except FATAL_EXCEPTION_TYPES as e:
                    log_fatal_error(condition_name, split_label, e)
                    print(
                        f"FATAL ERROR at {condition_name}/{split_label}: {type(e).__name__}: {e}",
                        file=sys.stderr,
                    )
                    mark_condition_failed(
                        status, condition_name, f"FATAL: {type(e).__name__}: {e}"
                    )
                    save_status(status, config["execution"]["status_json_path"])
                    return 1

                except torch.cuda.OutOfMemoryError as e:  # type: ignore[attr-defined]
                    log_fatal_error(condition_name, split_label, e)
                    print(
                        f"FATAL ERROR (OOM) at {condition_name}/{split_label}: {e}",
                        file=sys.stderr,
                    )
                    mark_condition_failed(
                        status, condition_name, f"FATAL: OutOfMemoryError: {e}"
                    )
                    save_status(status, config["execution"]["status_json_path"])
                    return 1

                except Exception as e:
                    n_soft_errors += 1
                    log_soft_error(condition_name, split_label, e, config)
                    print(
                        f"SOFT ERROR at {condition_name}/{split_label}: "
                        f"{type(e).__name__}: {e} "
                        f"(soft errors so far: {n_soft_errors}/{max_soft_errors})",
                        file=sys.stderr,
                    )
                    if n_soft_errors >= max_soft_errors:
                        print(
                            f"FATAL: ソフトエラー上限 ({max_soft_errors} 件) に達しました。停止します。",
                            file=sys.stderr,
                        )
                        return 1

                finally:
                    cleanup_gpu_memory()

            # 条件完了：ハイブリッドストレージ（画像削除）
            if config["execution"].get("delete_images_after_training", True):
                delete_condition_images(condition_name, config)
                print(f"  [{condition_name}] 画像ディレクトリを削除しました")

        except FATAL_EXCEPTION_TYPES as e:
            log_fatal_error(condition_name, "condition_level", e)
            mark_condition_failed(
                status, condition_name, f"FATAL: {type(e).__name__}: {e}"
            )
            save_status(status, config["execution"]["status_json_path"])
            return 1

    print("\n" + "=" * 60)
    print("全 27 条件の学習が完了しました")
    print_status_summary(status)
    print(f"ソフトエラー総数: {n_soft_errors}")
    print("=" * 60)

    return 0


# ===== サブ関数群 =====

def load_config(config_path: str) -> dict:
    """config.yaml を読み込む。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_splits(config: dict) -> dict:
    """splits.json を読み込む。"""
    splits_path = Path(config["paths"]["cv_splits_json"])
    if not splits_path.exists():
        raise FatalError(
            f"splits.json が存在しません: {splits_path}\n"
            f"先に `uv run python scripts/prepare_cv_splits.py` を実行してください。"
        )
    with open(splits_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_condition_names(config: dict) -> list[str]:
    """config['mel_conditions'] から全 27 条件名を生成する。"""
    mel = config["mel_conditions"]
    names = []
    for n_fft in mel["n_fft"]:
        for n_mels in mel["n_mels"]:
            for fmax in mel["fmax"]:
                names.append(f"nfft{n_fft}_mel{n_mels}_fmax{fmax}")
    return names


def ensure_images_generated(condition_name: str, config_path: str, config: dict) -> None:
    """指定条件のメルスペクトログラム画像が存在しなければ subprocess で生成する。

    Raises:
        FatalError: 画像生成に失敗した場合。
    """
    images_root = Path(config["paths"]["images_root"])
    condition_dir = images_root / condition_name
    marker_file = condition_dir / ".complete"

    if marker_file.exists():
        print(f"  [{condition_name}] 画像は生成済み、スキップ")
        return

    print(f"  [{condition_name}] 画像生成を開始...")
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "generate_melspectrogram.py"),
            "--config", config_path,
            "--condition", condition_name,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise FatalError(
            f"画像生成に失敗しました ({condition_name}):\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    if not marker_file.exists():
        raise FatalError(
            f"画像生成完了後に .complete マーカーがありません: {marker_file}"
        )

    print(f"  [{condition_name}] 画像生成完了")


def delete_condition_images(condition_name: str, config: dict) -> None:
    """指定条件の画像ディレクトリを削除する。"""
    images_root = Path(config["paths"]["images_root"])
    condition_dir = images_root / condition_name
    if condition_dir.exists():
        shutil.rmtree(condition_dir)


def train_one_split(
    condition_name: str,
    split: dict,
    clip_metadata: pd.DataFrame,
    config: dict,
) -> dict[str, Any]:
    """1 つの (condition, repeat_id, fold_id) について学習と評価を実行する。

    Returns:
        以下のキーを持つ dict:
            - predictions: クリップレベル予測 DataFrame
            - metrics: 評価指標 dict (all_metrics.csv の 1 行分)
            - history: Loss 曲線 dict
            - model_state_dict: 保存条件の場合のみ含まれる
    """
    # 1. データ準備
    train_df, val_df, test_df = prepare_dataframes(split, clip_metadata, config)

    # 2. データリーク二重チェック
    verify_no_patient_overlap(train_df, val_df, test_df)

    # 3. DataLoader 構築
    train_loader = build_dataloader(train_df, condition_name, config, mode="train")
    val_loader = build_dataloader(val_df, condition_name, config, mode="val")
    test_loader = build_dataloader(test_df, condition_name, config, mode="test")

    # 4. 学習
    fold_seed = derive_fold_seed(condition_name, split["repeat_id"], split["fold_id"])
    model, history = train_one_fold(
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        seed=fold_seed,
    )

    # 5. テスト予測
    predictions_df = predict_on_loader(model, test_loader, config)

    # 6. 指標計算
    metrics = compute_all_metrics(
        predictions=predictions_df,
        clip_metadata=clip_metadata,
        config=config,
    )

    # 7. history のサマリーを metrics に追加（all_metrics.csv の後半カラム用）
    metrics["epochs_run"] = history.get("epochs_run", 0)
    metrics["best_epoch"] = history.get("best_epoch", 0)
    metrics["best_val_auc"] = history.get("best_val_auc", float("nan"))
    metrics["best_val_loss_at_best_auc"] = history.get("best_val_loss_at_best_auc", float("nan"))

    # 8. メタ情報を予測 DataFrame に付与
    predictions_df = predictions_df.copy()
    predictions_df["condition_name"] = condition_name
    predictions_df["repeat_id"] = split["repeat_id"]
    predictions_df["fold_id"] = split["fold_id"]

    result: dict[str, Any] = {
        "predictions": predictions_df,
        "metrics": metrics,
        "history": history,
    }

    if should_save_model(condition_name, config):
        result["model_state_dict"] = {k: v.cpu() for k, v in model.state_dict().items()}

    return result


def prepare_dataframes(
    split: dict,
    clip_metadata: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """splits.json の 1 エントリから train_df / val_df / test_df を構築する。

    外側分割は splits.json から取得。内側 validation split は動的に計算。
    """
    train_clips = split["train_clips"]
    test_clips = split["test_clips"]

    full_train_df = clip_metadata[clip_metadata["clip_file"].isin(train_clips)].copy()
    test_df = clip_metadata[clip_metadata["clip_file"].isin(test_clips)].copy()

    val_random_state = split["random_state"] + 1000
    train_val_result = split_train_test_by_patient(
        df=full_train_df,
        patient_id_col=config["cv"]["patient_id_col"],
        label_name="fv_value",
        threshold=config["metadata"]["threshold_fv"],
        test_size=config["cv"]["validation_split_in_train"],
        random_state=val_random_state,
        use_yield=False,
    )

    # split_train_test_by_patient は use_yield=False のとき (train_df, test_df) タプルを返す
    if isinstance(train_val_result, tuple):
        train_df, val_df = train_val_result
    else:
        # generator の場合 (use_yield=True の戻り値)
        folds = list(train_val_result)
        train_df, val_df = folds[0]

    return train_df, val_df, test_df


def verify_no_patient_overlap(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """train / val / test の patient_id 重複をチェックする。

    Raises:
        FatalError: 重複が見つかった場合。
    """
    train_patients = set(train_df["patient_id"].tolist())
    val_patients = set(val_df["patient_id"].tolist())
    test_patients = set(test_df["patient_id"].tolist())

    tv_overlap = train_patients & val_patients
    tt_overlap = train_patients & test_patients
    vt_overlap = val_patients & test_patients

    if tv_overlap or tt_overlap or vt_overlap:
        raise FatalError(
            f"データリーク検出！患者ID重複：\n"
            f"  train ∩ val:  {tv_overlap}\n"
            f"  train ∩ test: {tt_overlap}\n"
            f"  val ∩ test:   {vt_overlap}"
        )


def should_save_model(condition_name: str, config: dict) -> bool:
    """この条件のモデル重みを保存すべきかを返す。"""
    save_config = config.get("save_models", {})
    if save_config.get("save_all", False):
        return True
    explicit_list = save_config.get("explicit_conditions", [])
    return condition_name in explicit_list


def save_predictions(
    predictions_df: pd.DataFrame,
    condition_name: str,
    config: dict,
) -> None:
    """予測値を {condition_name}.csv に追記する。"""
    output_path = Path(config["paths"]["results_root"]) / "predictions" / f"{condition_name}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        predictions_df.to_csv(output_path, mode="a", header=False, index=False)
    else:
        predictions_df.to_csv(output_path, mode="w", header=True, index=False)


def save_metrics(
    metrics: dict,
    condition_name: str,
    split: dict,
    config: dict,
) -> None:
    """指標を all_metrics.csv に追記する。1 行 = 1 (condition, repeat_id, fold_id)。"""
    output_path = Path(config["paths"]["results_root"]) / "metrics" / "all_metrics.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    m = re.match(r"nfft(\d+)_mel(\d+)_fmax(\d+)", condition_name)
    if m is None:
        raise ValueError(f"condition_name のパース失敗: {condition_name}")

    row = {
        "condition_name": condition_name,
        "n_fft": int(m.group(1)),
        "n_mels": int(m.group(2)),
        "fmax": int(m.group(3)),
        "repeat_id": split["repeat_id"],
        "fold_id": split["fold_id"],
        **metrics,
    }
    row_df = pd.DataFrame([row])

    if output_path.exists():
        row_df.to_csv(output_path, mode="a", header=False, index=False)
    else:
        row_df.to_csv(output_path, mode="w", header=True, index=False)


def save_loss_curve(
    history: dict,
    condition_name: str,
    split_label: str,
    config: dict,
) -> None:
    """Loss 曲線を CSV として保存する。"""
    output_dir = Path(config["paths"]["results_root"]) / "loss_curves" / condition_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # history にはリスト値のキーとスカラー値のキーが混在するため、リスト部分のみ DataFrame 化
    list_keys = ["epoch", "train_loss", "train_auc", "val_loss", "val_auc", "lr"]
    curve_data = {k: history[k] for k in list_keys if k in history}
    df = pd.DataFrame(curve_data)
    df.to_csv(output_dir / f"{split_label}.csv", index=False)


def save_model_weights(
    state_dict: dict,
    condition_name: str,
    split_label: str,
    config_path: str,
    config: dict,
) -> None:
    """モデルの state_dict を .pth として保存する。"""
    output_dir = Path(config["paths"]["results_root"]) / "models" / condition_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split_label}.pth"

    snapshot_path = Path(config["paths"]["results_root"]) / "config_snapshot.yaml"
    config_hash = compute_config_hash(snapshot_path if snapshot_path.exists() else config_path)

    torch.save(
        {
            "model_state_dict": state_dict,
            "condition_name": condition_name,
            "split_label": split_label,
            "config_hash": config_hash,
        },
        output_path,
    )


def cleanup_gpu_memory() -> None:
    """GPU メモリと Python ヒープを明示的に解放する。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_config_snapshot(config_path: str, config: dict) -> None:
    """config.yaml のスナップショットを results/ に保存する（初回のみ）。"""
    snapshot_path = Path(config["paths"]["results_root"]) / "config_snapshot.yaml"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot_path.exists():
        shutil.copy(config_path, snapshot_path)


# ===== ログ関連 =====

def setup_logging(config: dict) -> None:
    """ログファイルのディレクトリを準備する。"""
    log_path = Path(config["paths"]["results_root"]) / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)


def log_run_start(config_path: str, config: dict) -> None:
    """実行開始情報をログに記録する。"""
    log_path = Path(config["paths"]["results_root"]) / "run.log"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"実行開始: {datetime.now().isoformat()}\n")
        f.write(f"config: {config_path}\n")
        f.write(f"config_hash: {compute_config_hash(config_path)}\n")
        f.write(f"{'=' * 60}\n")


def log_fatal_error(condition_name: str, split_label: str, error: Exception) -> None:
    """致命的エラーを stderr に出力する。"""
    print(f"[FATAL] {condition_name}/{split_label}", file=sys.stderr)
    print(f"        {type(error).__name__}: {error}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)


def log_soft_error(
    condition_name: str,
    split_label: str,
    error: Exception,
    config: dict,
) -> None:
    """軽微エラーをファイルに記録する。"""
    error_log_path = Path(config["paths"]["results_root"]) / "soft_errors.log"
    error_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(error_log_path, "a", encoding="utf-8") as f:
        f.write(f"\n{datetime.now().isoformat()} - {condition_name}/{split_label}\n")
        f.write(f"{type(error).__name__}: {error}\n")
        f.write(traceback.format_exc())


# ===== コマンドライン =====

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="メイン学習スクリプト")
    parser.add_argument("--config", default="config.yaml", help="config.yaml へのパス")
    args = parser.parse_args()

    sys.exit(main(args.config))
