"""tests/test_checkpoint.py — utils/checkpoint.py のスモークテスト。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.checkpoint import (
    compute_config_hash,
    is_condition_complete,
    is_split_complete,
    load_or_init_status,
    mark_condition_complete,
    mark_condition_failed,
    mark_split_complete,
    save_status,
)


# ===== load_or_init_status =====

def test_load_or_init_creates_new():
    """status.json が存在しない場合に新規作成される。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        status_path = Path(tmpdir) / "status.json"
        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text("dummy: true\n")

        condition_names = ["nfft256_mel64_fmax1000", "nfft256_mel64_fmax1500"]
        status = load_or_init_status(status_path, config_path, condition_names, 30)

        assert len(status["conditions"]) == 2
        assert all(c["status"] == "pending" for c in status["conditions"].values())
        assert status["summary"]["n_conditions_pending"] == 2
        assert status["summary"]["n_conditions_completed"] == 0


def test_load_or_init_reads_existing():
    """status.json が存在する場合に読み込まれる。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        status_path = Path(tmpdir) / "status.json"
        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text("dummy: true\n")

        condition_names = ["nfft256_mel64_fmax1000"]
        status = load_or_init_status(status_path, config_path, condition_names, 2)
        save_status(status, status_path)

        # 再読み込み
        status2 = load_or_init_status(status_path, config_path, condition_names, 2)
        assert status2["conditions"] == status["conditions"]


# ===== is_condition_complete / is_split_complete =====

def test_is_condition_complete_pending():
    """pending 状態の条件は完了でない。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text("dummy: true\n")
        status = load_or_init_status(
            Path(tmpdir) / "s.json", config_path, ["cond_a"], 2
        )
        assert not is_condition_complete("cond_a", status)


def test_is_split_complete_false_initially():
    """初期状態では全分割が未完了。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text("dummy: true\n")
        status = load_or_init_status(
            Path(tmpdir) / "s.json", config_path, ["cond_a"], 2
        )
        assert not is_split_complete("cond_a", 1, 1, status)


# ===== mark_split_complete =====

def test_mark_split_complete_updates_count():
    """mark_split_complete で n_splits_completed が増える。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text("dummy: true\n")
        status = load_or_init_status(
            Path(tmpdir) / "s.json", config_path, ["cond_a"], 2
        )

        mark_split_complete(status, "cond_a", 1, 1)
        assert status["conditions"]["cond_a"]["n_splits_completed"] == 1
        assert status["conditions"]["cond_a"]["status"] == "in_progress"


def test_mark_split_complete_all_completes_condition():
    """全 n_splits_total を完了させると条件が completed になる。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text("dummy: true\n")
        status = load_or_init_status(
            Path(tmpdir) / "s.json", config_path, ["cond_a"], 2
        )

        mark_split_complete(status, "cond_a", 1, 1)
        mark_split_complete(status, "cond_a", 1, 2)

        assert status["conditions"]["cond_a"]["status"] == "completed"
        assert status["summary"]["n_conditions_completed"] == 1


def test_mark_split_complete_idempotent():
    """同じ分割を二重にマークしても n_splits_completed は増えない。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text("dummy: true\n")
        status = load_or_init_status(
            Path(tmpdir) / "s.json", config_path, ["cond_a"], 3
        )

        mark_split_complete(status, "cond_a", 1, 1)
        mark_split_complete(status, "cond_a", 1, 1)  # 重複
        assert status["conditions"]["cond_a"]["n_splits_completed"] == 1


# ===== mark_condition_failed =====

def test_mark_condition_failed():
    """failed マークが正しくセットされる。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text("dummy: true\n")
        status = load_or_init_status(
            Path(tmpdir) / "s.json", config_path, ["cond_a"], 2
        )

        mark_condition_failed(status, "cond_a", "テストエラー")
        assert status["conditions"]["cond_a"]["status"] == "failed"
        assert status["conditions"]["cond_a"]["error_message"] == "テストエラー"
        assert status["summary"].get("n_conditions_failed", 0) == 1


# ===== save_status / atomic write =====

def test_atomic_write():
    """save_status が atomic write で正しく保存される。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        status_path = Path(tmpdir) / "status.json"
        status = {"version": "1.0", "test_key": "test_value"}
        save_status(status, status_path)

        assert status_path.exists()
        with open(status_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == status

        # .tmp ファイルが残っていないことを確認
        assert not (status_path.with_suffix(".json.tmp")).exists()


# ===== compute_config_hash =====

def test_compute_config_hash_stable():
    """同じファイルは常に同じハッシュを返す。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text("key: value\n")

        h1 = compute_config_hash(config_path)
        h2 = compute_config_hash(config_path)
        assert h1 == h2
        assert len(h1) == 16


def test_compute_config_hash_changes_on_edit():
    """ファイル内容が変わるとハッシュも変わる。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        config_path.write_text("key: value1\n")
        h1 = compute_config_hash(config_path)

        config_path.write_text("key: value2\n")
        h2 = compute_config_hash(config_path)

        assert h1 != h2
