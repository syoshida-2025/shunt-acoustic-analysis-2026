"""utils/trainer.py — PyTorch 学習ループ。

主な責務:
    - 1 fold 分の学習を実行（複数エポック、Early Stopping 付き）
    - Mixed Precision Training (AMP) で高速化
    - Loss/AUC 曲線データを生成・返却
    - val_auc ベースの Early Stopping
    - テストデータでの推論

主な公開関数:
    - train_one_fold: 1 fold 分の学習を実行し、モデルと履歴を返す。
    - predict_on_loader: 学習済みモデルでテストデータを推論。

EarlyStopping クラス:
    val_auc が改善しなくなったら停止判定。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader


# ===== Early Stopping =====

@dataclass
class EarlyStopping:
    """val_auc ベースの Early Stopping。

    val_auc が patience エポック連続で改善しなかったら停止フラグを立てる。

    Attributes:
        patience: 改善判定の許容エポック数（通常 5）。
        min_delta: 改善とみなす最小増分（通常 0.0）。
        best_metric: 過去最高の val_auc。
        best_epoch: best_metric を達成したエポック番号（1始まり）。
        counter: 連続で改善しなかったエポック数。
        should_stop: True なら学習を停止すべき。
        best_state_dict: best_metric 時点のモデル重み（CPU に保持）。
    """

    patience: int = 5
    min_delta: float = 0.0
    best_metric: float = -float("inf")
    best_epoch: int = 0
    counter: int = 0
    should_stop: bool = False
    best_state_dict: dict | None = field(default=None, repr=False)

    def step(
        self,
        current_metric: float,
        current_epoch: int,
        model: torch.nn.Module,
    ) -> bool:
        """1 エポック終了時に呼ぶ。

        Args:
            current_metric: 現在の val_auc。
            current_epoch: 現在のエポック番号（1始まり）。
            model: 現在のモデル（best 時に state_dict を保存する用）。

        Returns:
            True なら improved（counter リセット）、False なら not improved。
        """
        if current_metric > self.best_metric + self.min_delta:
            self.best_metric = current_metric
            self.best_epoch = current_epoch
            self.counter = 0
            # CPU にコピーして保存（GPUメモリ節約）
            self.best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


# ===== モデル構築 =====

def create_model(model_config: dict, device: torch.device) -> torch.nn.Module:
    """timm からモデルを作成し、device に転送する。

    Args:
        model_config: config["model"]["main"] または config["model"]["sub"]。
        device: torch.device。

    Returns:
        timm モデル。
    """
    model = timm.create_model(
        model_name=model_config["name"],
        pretrained=model_config["pretrained"],
        num_classes=model_config["num_classes"],
    )
    model = model.to(device)
    return model


# ===== Optimizer / Scheduler =====

def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    config: dict,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """AdamW と CosineAnnealingLR を構築する。"""
    train_config = config["training"]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
        betas=tuple(train_config["betas"]),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_config["scheduler_t_max"],
    )

    return optimizer, scheduler


# ===== 1 エポック分の学習 =====

def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    """1 エポック分の学習。AMP を使用。

    Returns:
        (epoch_loss, epoch_auc): エポック平均 loss と AUC。
    """
    model.train()
    total_loss = 0.0
    n_samples = 0
    all_probs: list[float] = []
    all_labels: list[int] = []

    for batch in train_loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp:
            with autocast(device_type=device.type):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size

        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    epoch_loss = total_loss / max(1, n_samples)
    try:
        epoch_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        epoch_auc = float("nan")

    return epoch_loss, epoch_auc


# ===== Validation 評価 =====

def validate_one_epoch(
    model: torch.nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    """1 エポック終了時の validation 評価。

    Returns:
        (val_loss, val_auc): validation 平均 loss と AUC。
    """
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_probs: list[float] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            if use_amp:
                with autocast(device_type=device.type):
                    logits = model(images)
                    loss = criterion(logits, labels)
            else:
                logits = model(images)
                loss = criterion(logits, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    val_loss = total_loss / max(1, n_samples)
    try:
        val_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        val_auc = float("nan")

    return val_loss, val_auc


# ===== Main 学習関数 =====

def train_one_fold(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
) -> tuple[torch.nn.Module, dict]:
    """1 fold 分の学習を実行する。

    フロー:
        1. モデル、optimizer、scheduler、scaler、criterion を構築。
        2. エポックループ:
           - train_one_epoch
           - validate_one_epoch
           - history に記録
           - early_stopping.step()
           - early_stopping.should_stop なら break
        3. best_state_dict をモデルに復元
        4. (model, history) を返す

    Args:
        train_loader: 訓練 DataLoader。
        val_loader: 検証 DataLoader。
        config: 全 config。

    Returns:
        (model, history):
            - model: best_val_auc を達成した時点のモデル（重み復元済み）
            - history: dict（エポック単位のログ）
    """
    train_config = config["training"]
    device = _get_device(config)
    use_amp = train_config.get("mixed_precision", True) and device.type == "cuda"

    model = create_model(config["model"]["main"], device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=use_amp)
    early_stopping = EarlyStopping(
        patience=train_config["early_stopping_patience"],
        min_delta=0.0,
    )

    history: dict[str, Any] = {
        "epoch": [],
        "train_loss": [],
        "train_auc": [],
        "val_loss": [],
        "val_auc": [],
        "lr": [],
    }
    best_val_loss_at_best_auc = float("nan")

    max_epochs = train_config["max_epochs"]
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        train_loss, train_auc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        val_loss, val_auc = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_auc"].append(train_auc)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["lr"].append(current_lr)

        improved = early_stopping.step(val_auc, epoch, model)
        if improved:
            best_val_loss_at_best_auc = val_loss

        elapsed = time.time() - t0
        patience_info = (
            "★" if improved
            else f"(patience {early_stopping.counter}/{early_stopping.patience})"
        )
        print(
            f"  Epoch {epoch:2d}/{max_epochs}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"val_auc={val_auc:.4f} {patience_info} "
            f"[{elapsed:.1f}s]"
        )

        if early_stopping.should_stop:
            print(
                f"  Early stopping triggered. Best epoch: {early_stopping.best_epoch} "
                f"(val_auc={early_stopping.best_metric:.4f})"
            )
            break

    # Best モデル復元
    if early_stopping.best_state_dict is not None:
        model.load_state_dict({
            k: v.to(device) for k, v in early_stopping.best_state_dict.items()
        })

    history["epochs_run"] = len(history["epoch"])
    history["best_epoch"] = early_stopping.best_epoch
    history["best_val_auc"] = early_stopping.best_metric
    history["best_val_loss_at_best_auc"] = best_val_loss_at_best_auc

    return model, history


# ===== テスト推論 =====

def predict_on_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    config: dict,
) -> pd.DataFrame:
    """学習済みモデルで指定 DataLoader 上を推論する。

    Args:
        model: 学習済みモデル（best_val_auc 時点の重みが復元済み）。
        loader: 推論対象の DataLoader（通常テスト用）。
        config: 全 config。

    Returns:
        DataFrame:
            - clip_file: str
            - patient_id: str
            - true_label: int
            - pred_proba: float (positive クラスの予測確率)
    """
    device = _get_device(config)
    use_amp = config["training"].get("mixed_precision", True) and device.type == "cuda"

    model.eval()

    all_clip_files: list[str] = []
    all_patient_ids: list[str] = []
    all_labels: list[int] = []
    all_probs: list[float] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)

            if use_amp:
                with autocast(device_type=device.type):
                    logits = model(images)
            else:
                logits = model(images)

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(batch["label"].numpy().tolist())
            all_clip_files.extend(batch["clip_file"])
            all_patient_ids.extend(batch["patient_id"])

    return pd.DataFrame({
        "clip_file": all_clip_files,
        "patient_id": all_patient_ids,
        "true_label": all_labels,
        "pred_proba": all_probs,
    })


# ===== ヘルパー =====

def _get_device(config: dict) -> torch.device:
    """利用可能な device を返す。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _set_seeds(seed: int = 42) -> None:
    """再現性のための seed 設定。"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
