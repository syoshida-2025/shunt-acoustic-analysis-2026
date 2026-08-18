"""external_validation/tune_and_develop.py

外部検証パイプライン Phase A: 開発データ(内部検証と同一データセット)を用いた
ハイパーパラメータ探索と最終モデルの構築。

処理の流れ(アーキテクチャごとに独立に実行):
    1. 選択した単一条件(config_external.yaml の selected_condition)のメルスペク
       トログラム画像を開発データに対して生成する(未生成の場合のみ)。
    2. 患者単位で3-fold(反復なし)に分割する。
    3. learning_rate x weight_decay のgrid search を行う。各HP候補について、
       3-foldそれぞれで学習し(訓練foldにはデータ拡張を適用、検証foldには適用しない)、
       平均val_aucで比較する。
    4. 最良のHPを選び、その3-foldぶんの検証予測(out-of-fold予測)をプールして
       Youden閾値を決定する(開発データのみに基づく「ロック済み閾値」)。
    5. 最良HPの3-fold分のbest_epochの中央値をエポック数として固定し、開発データ
       全例で最終モデルを1回だけ再学習する(早期停止なし)。
    6. grid_search_results.csv / selected_hp.json / locked_threshold.json /
       oof_predictions_winning_hp.csv / final_model.pth を保存する。

内部検証パイプライン(scripts/train_model.py 等)とは独立したスクリプトだが、
utils/ 以下のデータ処理・学習・評価ロジックはそのまま再利用している。
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_split import split_train_test_by_patient
from utils.dataset import build_dataloader
from utils.external_metrics import lock_thresholds_from_oof_predictions
from utils.metadata_io import read_clip_metadata
from utils.sound2image import process_audio_to_images
from utils.trainer import (
    _set_seeds,
    build_optimizer_and_scheduler,
    create_model,
    derive_fold_seed,
    predict_on_loader,
    train_one_epoch,
    train_one_fold,
)

import torch.nn as nn
from torch.amp import GradScaler


def _get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_condition_name(selected_condition: dict) -> str:
    return (
        f"nfft{selected_condition['n_fft']}"
        f"_mel{selected_condition['n_mels']}"
        f"_fmax{selected_condition['fmax']}"
    )


def ensure_dev_images(config: dict, condition_name: str) -> Path:
    """開発データについて、選択した条件のメルスペクトログラム画像を生成する(未生成の場合のみ)。"""
    images_root = Path(config["dev"]["images_root"])
    condition_dir = images_root / condition_name
    condition_dir.mkdir(parents=True, exist_ok=True)

    if (condition_dir / ".complete").exists():
        print(f"[ensure_dev_images] {condition_name} は生成済み。スキップ。")
        return condition_dir

    clip_df = read_clip_metadata(config["dev"]["clip_metadata_csv"])
    df_for_process = clip_df.copy()
    df_for_process["audio_file"] = df_for_process["clip_file"]

    sc = config["selected_condition"]
    aug = config["augmentation"]
    hop_length = int(sc["n_fft"] * config["mel_conditions"]["hop_length_ratio"])

    print(f"[ensure_dev_images] {condition_name} の画像を生成中...")
    process_audio_to_images(
        df=df_for_process,
        audio_folder=config["dev"]["clip_dir"],
        output_folder=str(condition_dir),
        label_name="label",
        threshold=0.5,
        start_time=0.0,
        clip_duration=config["audio"]["clip_duration_sec"],
        target_sr=config["audio"]["sample_rate"],
        use_mel=True,
        n_mels=sc["n_mels"],
        n_fft_direct=sc["n_fft"],
        hop_length_direct=hop_length,
        fmax=sc["fmax"],
        target_shape=tuple(config["mel_conditions"]["image_size"]),
        cmap=config["mel_conditions"]["cmap"],
        n_aug=aug["noise_n_aug"],
        apply_silence=aug["apply_silence"],
        max_silence_duration=0.0,
        apply_noise=aug["apply_noise"],
        min_snr_db=aug["noise_snr_db_min"],
        max_snr_db=aug["noise_snr_db_max"],
        strength=1.0,
        use_adaptive_snr=False,
        enable_multiclip=False,
        base_seed=42,
    )
    (condition_dir / ".complete").touch()
    return condition_dir


def build_runtime_config(
    config: dict,
    architecture_key: str,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    images_root: str | None = None,
) -> dict:
    """utils/dataset.py, utils/trainer.py が期待する形の config を組み立てる。

    config_external.yaml のセクション構成(dev/external/architectures/hp_search)は
    内部検証の config.yaml とは異なるため、既存ユーティリティ関数(build_dataloader,
    train_one_fold 等)にそのまま渡せる形へ変換する。

    images_root を省略すると開発データ用(config["dev"]["images_root"])になる。
    外部データを評価する場合は、呼び出し側で config["external"]["images_root"] を
    明示的に渡すこと(渡し忘れると開発データ側の画像フォルダを見に行ってしまう)。
    """
    training = dict(config["training"])
    if learning_rate is not None:
        training["learning_rate"] = learning_rate
    if weight_decay is not None:
        training["weight_decay"] = weight_decay

    return {
        "paths": {"images_root": images_root if images_root is not None else config["dev"]["images_root"]},
        "training": training,
        "mel_conditions": {"image_size": config["mel_conditions"]["image_size"]},
        "augmentation": {"brightness_jitter": config["augmentation"]["brightness_jitter"]},
        "normalization": config["normalization"],
        "model": {"main": config["architectures"][architecture_key]},
    }


def run_hp_search(
    config: dict,
    architecture_key: str,
    condition_name: str,
    clip_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """LR x weight_decay のgrid searchを3-fold CV(反復なし)で実施する。

    Returns:
        (grid_results_df, winning_trial):
            grid_results_df: 全HP候補 x 全foldの結果一覧(論文報告用)。
            winning_trial: 最良HPの詳細(lr, wd, mean_val_auc, folds=[{fold_id,
                best_epoch, val_auc, val_predictions(DataFrame)}, ...])。
    """
    hp = config["hp_search"]
    lr_grid = hp["learning_rate"]
    wd_grid = hp["weight_decay"]
    n_folds = hp["n_folds"]

    folds = list(
        split_train_test_by_patient(
            df=clip_df,
            patient_id_col=hp["patient_id_col"],
            label_name="fv_value",
            threshold=config["metadata"]["threshold_fv"],
            test_size=1.0 / n_folds,
            random_state=hp["random_state"],
            use_yield=True,
        )
    )
    if len(folds) != n_folds:
        raise RuntimeError(
            f"{n_folds}-foldを期待しましたが{len(folds)}foldが生成されました。"
        )

    grid_rows: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []

    for lr, wd in itertools.product(lr_grid, wd_grid):
        fold_results = []
        for fold_id, (train_df, val_df) in enumerate(folds, start=1):
            runtime_config = build_runtime_config(config, architecture_key, lr, wd)

            train_loader = build_dataloader(train_df, condition_name, runtime_config, mode="train")
            val_loader = build_dataloader(val_df, condition_name, runtime_config, mode="val")

            fold_seed = derive_fold_seed(architecture_key, condition_name, lr, wd, fold_id)
            model, history = train_one_fold(train_loader, val_loader, runtime_config, seed=fold_seed)
            val_predictions = predict_on_loader(model, val_loader, runtime_config)

            fold_results.append({
                "fold_id": fold_id,
                "best_epoch": history["best_epoch"],
                "val_auc": history["best_val_auc"],
                "val_predictions": val_predictions,
            })

            grid_rows.append({
                "architecture": architecture_key,
                "learning_rate": lr,
                "weight_decay": wd,
                "fold_id": fold_id,
                "best_epoch": history["best_epoch"],
                "val_auc": history["best_val_auc"],
            })

            print(
                f"  [{architecture_key}] lr={lr:.1e} wd={wd:.1e} fold={fold_id}/{n_folds} "
                f"val_auc={history['best_val_auc']:.4f} (best_epoch={history['best_epoch']})"
            )

        mean_val_auc = float(np.mean([r["val_auc"] for r in fold_results]))
        trials.append({
            "learning_rate": lr,
            "weight_decay": wd,
            "mean_val_auc": mean_val_auc,
            "folds": fold_results,
        })

    winning_trial = max(trials, key=lambda t: t["mean_val_auc"])
    print(
        f"[{architecture_key}] 採用HP: lr={winning_trial['learning_rate']:.1e}, "
        f"wd={winning_trial['weight_decay']:.1e} (mean_val_auc={winning_trial['mean_val_auc']:.4f})"
    )

    return pd.DataFrame(grid_rows), winning_trial


def train_fixed_epochs(
    train_loader,
    runtime_config: dict,
    n_epochs: int,
    seed: int = 42,
) -> torch.nn.Module:
    """検証セットなしで、指定エポック数だけ学習する(最終モデルの全データ再学習用)。

    HP探索フェーズの early stopping とは異なり、ここでは開発データ全例を訓練に使うため
    検証データが存在しない。エポック数は run_hp_search で選ばれたHPの3-fold分の
    best_epoch の中央値をあらかじめ決定し、固定値として渡す。

    seed: モデル重み初期化・Dropout・オンザフライ拡張・cuDNN演算選択を固定する
        シード(呼び出し元で architecture_key/condition_name から導出)。
    """
    _set_seeds(seed)

    train_config = runtime_config["training"]
    device = _get_device()
    use_amp = train_config.get("mixed_precision", True) and device.type == "cuda"

    model = create_model(runtime_config["model"]["main"], device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, runtime_config)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=use_amp)

    for epoch in range(1, n_epochs + 1):
        train_loss, train_auc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )
        scheduler.step()
        print(f"  [final retrain] epoch {epoch}/{n_epochs}: train_loss={train_loss:.4f} train_auc={train_auc:.4f}")

    return model


def run_architecture(config: dict, architecture_key: str, condition_name: str, clip_df: pd.DataFrame) -> None:
    output_dir = Path(config["results_root"]) / architecture_key / condition_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== 1. HP探索(3-fold CV, 反復なし) =====
    grid_results_df, winning_trial = run_hp_search(config, architecture_key, condition_name, clip_df)
    grid_results_df.to_csv(output_dir / "grid_search_results.csv", index=False)

    selected_hp = {
        "architecture": architecture_key,
        "condition_name": condition_name,
        "learning_rate": winning_trial["learning_rate"],
        "weight_decay": winning_trial["weight_decay"],
        "mean_val_auc": winning_trial["mean_val_auc"],
        "per_fold_best_epoch": [f["best_epoch"] for f in winning_trial["folds"]],
    }
    epoch_count = int(round(float(np.median(selected_hp["per_fold_best_epoch"]))))
    selected_hp["final_epoch_count"] = epoch_count
    with open(output_dir / "selected_hp.json", "w", encoding="utf-8") as f:
        json.dump(selected_hp, f, ensure_ascii=False, indent=2)

    # ===== 2. 閾値のロック(勝者HPのout-of-fold予測をプール) =====
    oof_predictions = pd.concat([f["val_predictions"] for f in winning_trial["folds"]], ignore_index=True)
    oof_predictions.to_csv(output_dir / "oof_predictions_winning_hp.csv", index=False)

    locked_thresholds = lock_thresholds_from_oof_predictions(oof_predictions, clip_df)
    with open(output_dir / "locked_threshold.json", "w", encoding="utf-8") as f:
        json.dump(locked_thresholds, f, ensure_ascii=False, indent=2)
    print(f"[{architecture_key}] ロック済み閾値: {locked_thresholds}")

    # ===== 3. 最終モデル: 開発データ全例で固定エポック数だけ再学習 =====
    runtime_config = build_runtime_config(
        config, architecture_key,
        learning_rate=selected_hp["learning_rate"],
        weight_decay=selected_hp["weight_decay"],
    )
    full_train_loader = build_dataloader(clip_df, condition_name, runtime_config, mode="train")
    final_seed = derive_fold_seed(architecture_key, condition_name, "final")
    final_model = train_fixed_epochs(full_train_loader, runtime_config, epoch_count, seed=final_seed)

    torch.save(
        {
            "model_state_dict": {k: v.cpu() for k, v in final_model.state_dict().items()},
            "architecture": architecture_key,
            "condition_name": condition_name,
            "selected_hp": selected_hp,
        },
        output_dir / "final_model.pth",
    )
    print(f"[{architecture_key}] 最終モデルを保存しました: {output_dir / 'final_model.pth'}")


def main(config_path: str, architecture: str | None) -> int:
    config = load_config(config_path)
    condition_name = build_condition_name(config["selected_condition"])

    Path(config["results_root"]).mkdir(parents=True, exist_ok=True)
    snapshot_path = Path(config["results_root"]) / "config_external_snapshot.yaml"
    if not snapshot_path.exists():
        import shutil
        shutil.copy(config_path, snapshot_path)

    ensure_dev_images(config, condition_name)
    clip_df = read_clip_metadata(config["dev"]["clip_metadata_csv"])

    architectures = [architecture] if architecture else list(config["architectures"].keys())
    for arch_key in architectures:
        print(f"\n{'=' * 60}\nアーキテクチャ: {arch_key}\n{'=' * 60}")
        run_architecture(config, arch_key, condition_name, clip_df)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="外部検証Phase A: HP探索と最終モデル構築")
    parser.add_argument("--config", default="config_external.yaml")
    parser.add_argument(
        "--architecture",
        default=None,
        help="config_external.yaml の architectures キーの1つ。省略時は全アーキテクチャを実行。",
    )
    args = parser.parse_args()
    sys.exit(main(args.config, args.architecture))
