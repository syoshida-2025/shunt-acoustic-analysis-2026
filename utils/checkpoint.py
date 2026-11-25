"""utils/checkpoint.py — status.json の読み書きと再開判定。

このモジュールは status.json のすべての操作を集約する。
他のモジュール（train_model.py）から直接 status.json を編集せず、
このモジュール経由で行うことで一貫性を担保する。
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


# ===== 公開定数 =====

STATUS_VERSION = "1.0"
VALID_STATUSES = ("pending", "in_progress", "completed", "failed")


# ===== ハッシュ計算 =====

def compute_config_hash(config_path: str | Path) -> str:
    """config.yaml の内容から SHA-256 ハッシュの先頭16文字を計算する。

    Args:
        config_path: config.yaml へのパス。

    Returns:
        16文字のハッシュ文字列。
    """
    config_path = Path(config_path)
    with open(config_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


# ===== 初期化と読み込み =====

def load_or_init_status(
    status_path: str | Path,
    config_path: str | Path,
    condition_names: list[str],
    n_splits_per_condition: int = 30,
) -> dict:
    """status.json を読み込むか、新規初期化する。

    動作:
        - status_path が存在しない場合: 新規初期化して空の status を返す。
        - 存在する場合: 読み込んで返す。
          ただし config_hash が変わっていた場合は WARNING を出す。

    Args:
        status_path: results/status.json へのパス。
        config_path: config.yaml へのパス（ハッシュ計算用）。
        condition_names: 全 27 条件の名前リスト。
        n_splits_per_condition: 各条件あたりの分割数（通常30）。

    Returns:
        status 辞書。
    """
    status_path = Path(status_path)
    config_hash = compute_config_hash(config_path)

    if not status_path.exists():
        status = _init_empty_status(
            config_hash=config_hash,
            condition_names=condition_names,
            n_splits_per_condition=n_splits_per_condition,
        )
        return status

    with open(status_path, "r", encoding="utf-8") as f:
        status = json.load(f)

    # config_hash の整合性チェック
    if status.get("config_hash") != config_hash:
        print(
            f"WARNING: config.yaml が変更されています。\n"
            f"  保存時のハッシュ: {status.get('config_hash')}\n"
            f"  現在のハッシュ:   {config_hash}\n"
            f"  再開を続けますが、config 変更が結果に影響する可能性があります。",
            file=sys.stderr,
        )

    return status


def _init_empty_status(
    config_hash: str,
    condition_names: list[str],
    n_splits_per_condition: int,
) -> dict:
    """空の status 辞書を作成する（内部関数）。"""
    now = datetime.now().isoformat(timespec="seconds")
    conditions = {}
    for name in condition_names:
        conditions[name] = {
            "status": "pending",
            "splits_completed": [],
            "n_splits_total": n_splits_per_condition,
            "n_splits_completed": 0,
        }

    return {
        "version": STATUS_VERSION,
        "started_at": now,
        "last_updated": now,
        "config_hash": config_hash,
        "conditions": conditions,
        "summary": {
            "n_conditions_total": len(condition_names),
            "n_conditions_completed": 0,
            "n_conditions_in_progress": 0,
            "n_conditions_pending": len(condition_names),
        },
    }


# ===== 進捗確認 =====

def is_condition_complete(condition_name: str, status: dict) -> bool:
    """指定条件の全 30 分割が完了しているかを判定する。

    Args:
        condition_name: 例 "nfft256_mel64_fmax1000"
        status: load_or_init_status の戻り値。

    Returns:
        True なら完了済み。
    """
    cond_status = status["conditions"].get(condition_name)
    if cond_status is None:
        return False
    return cond_status["status"] == "completed"


def is_split_complete(
    condition_name: str,
    repeat_id: int,
    fold_id: int,
    status: dict,
) -> bool:
    """指定分割が完了しているかを判定する。

    Args:
        condition_name: 条件名。
        repeat_id: 1〜10。
        fold_id: 1〜3。
        status: status 辞書。

    Returns:
        True なら完了済み。
    """
    cond_status = status["conditions"].get(condition_name)
    if cond_status is None:
        return False
    for split in cond_status["splits_completed"]:
        if split["repeat_id"] == repeat_id and split["fold_id"] == fold_id:
            return True
    return False


# ===== 進捗更新 =====

def mark_split_complete(
    status: dict,
    condition_name: str,
    repeat_id: int,
    fold_id: int,
) -> None:
    """指定分割を完了としてマークする（in-place 更新）。

    呼び出し後は `save_status` で永続化する必要がある。

    Args:
        status: status 辞書（in-place 更新される）。
        condition_name: 条件名。
        repeat_id: 1〜10。
        fold_id: 1〜3。
    """
    cond_status = status["conditions"][condition_name]

    # 重複チェック
    for existing in cond_status["splits_completed"]:
        if existing["repeat_id"] == repeat_id and existing["fold_id"] == fold_id:
            return  # 既にマーク済み

    cond_status["splits_completed"].append({
        "repeat_id": repeat_id,
        "fold_id": fold_id,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    })
    cond_status["n_splits_completed"] = len(cond_status["splits_completed"])

    # 全分割完了なら条件レベルも完了に
    if cond_status["n_splits_completed"] >= cond_status["n_splits_total"]:
        mark_condition_complete(status, condition_name)
    elif cond_status["status"] == "pending":
        cond_status["status"] = "in_progress"
        cond_status["started_at"] = datetime.now().isoformat(timespec="seconds")

    _update_summary(status)
    status["last_updated"] = datetime.now().isoformat(timespec="seconds")


def mark_condition_complete(status: dict, condition_name: str) -> None:
    """指定条件を完了としてマークする（in-place 更新）。

    通常は `mark_split_complete` 経由で自動的に呼ばれるが、
    手動で完了扱いにしたい場合に直接呼ぶこともできる。
    """
    cond_status = status["conditions"][condition_name]
    cond_status["status"] = "completed"
    cond_status["completed_at"] = datetime.now().isoformat(timespec="seconds")
    _update_summary(status)
    status["last_updated"] = datetime.now().isoformat(timespec="seconds")


def mark_condition_failed(
    status: dict,
    condition_name: str,
    error_message: str,
) -> None:
    """指定条件を失敗としてマークする（in-place 更新）。

    Args:
        status: status 辞書。
        condition_name: 条件名。
        error_message: エラーメッセージ（記録用）。
    """
    cond_status = status["conditions"][condition_name]
    cond_status["status"] = "failed"
    cond_status["failed_at"] = datetime.now().isoformat(timespec="seconds")
    cond_status["error_message"] = error_message
    _update_summary(status)
    status["last_updated"] = datetime.now().isoformat(timespec="seconds")


def _update_summary(status: dict) -> None:
    """status['summary'] を再計算する（内部関数）。"""
    n_completed = sum(
        1 for c in status["conditions"].values() if c["status"] == "completed"
    )
    n_in_progress = sum(
        1 for c in status["conditions"].values() if c["status"] == "in_progress"
    )
    n_pending = sum(
        1 for c in status["conditions"].values() if c["status"] == "pending"
    )
    n_failed = sum(
        1 for c in status["conditions"].values() if c["status"] == "failed"
    )
    status["summary"] = {
        "n_conditions_total": len(status["conditions"]),
        "n_conditions_completed": n_completed,
        "n_conditions_in_progress": n_in_progress,
        "n_conditions_pending": n_pending,
        "n_conditions_failed": n_failed,
    }


# ===== 永続化 =====

def save_status(status: dict, status_path: str | Path) -> None:
    """status を JSON ファイルに保存する。

    安全のため、一時ファイルに書いてから rename する atomic-write を実装。

    Args:
        status: status 辞書。
        status_path: 保存先パス。
    """
    status_path = Path(status_path)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = status_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)
    tmp_path.replace(status_path)  # atomic rename


# ===== サマリー表示 =====

def print_status_summary(status: dict) -> None:
    """status の現在の進捗をコンソール出力。"""
    s = status["summary"]
    print("=" * 60)
    print("進捗サマリー")
    print("=" * 60)
    print(f"  全 {s['n_conditions_total']} 条件中")
    print(f"    完了: {s['n_conditions_completed']}")
    print(f"    実行中: {s['n_conditions_in_progress']}")
    print(f"    未着手: {s['n_conditions_pending']}")
    if s.get("n_conditions_failed", 0) > 0:
        print(f"    失敗: {s['n_conditions_failed']}")
    print(f"  最終更新: {status['last_updated']}")
    print("=" * 60)
