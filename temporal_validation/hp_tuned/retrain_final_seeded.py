"""
temporal_validation/hp_tuned/retrain_final_seeded.py

Temporal validation の最終モデルを、乱数シードを固定（seed 42）して学習し直す。

tune_and_develop.py が results/{architecture}/{condition}/ に保存した
selected_hp.json（HP・エポック数）と locked_threshold.json をそのまま使い、
HP 探索はやり直さずに最終モデルだけを学習する。学習の中身は tune_and_develop.py の
関数をそのまま呼ぶので同一で、違いはシード固定のみ。

出力（results/{architecture}/{condition}/）:
    final_model.pth    … シード固定で学習した重み（tune_and_develop.py の最終モデルを上書き）
    retrain_info.json  … シード・所要時間など

実行（このフォルダで、tune_and_develop.py の後に）:
    uv run retrain_final_seeded.py --force
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from tune_and_develop import (
    build_condition_name,
    build_runtime_config,
    ensure_dev_images,
    load_config,
    train_fixed_epochs,
)
from utils.dataset import build_dataloader  # noqa: E402
from utils.metadata_io import read_clip_metadata  # noqa: E402
from utils.trainer import _set_seeds  # noqa: E402

SEED = 42


def set_deterministic(seed: int) -> None:
    """重み初期化・ColorJitter・DataLoader の乱数と cuDNN の演算選択を固定する。"""
    _set_seeds(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_architecture(config: dict, architecture_key: str, condition_name: str, clip_df, force: bool) -> None:
    out_dir = Path(config["results_root"]) / architecture_key / condition_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / "final_model.pth").exists() and not force:
        print(f"[{architecture_key}] final_model.pth が既にあるのでスキップ（上書きは --force）")
        return

    with open(out_dir / "selected_hp.json", encoding="utf-8") as f:
        selected_hp = json.load(f)

    print(
        f"[{architecture_key}] tune_and_develop.py で選んだ HP を使用: lr={selected_hp['learning_rate']:.1e}, "
        f"wd={selected_hp['weight_decay']:.1e}, epochs={selected_hp['final_epoch_count']}, seed={SEED}"
    )

    set_deterministic(SEED)
    runtime_config = build_runtime_config(
        config, architecture_key,
        learning_rate=selected_hp["learning_rate"],
        weight_decay=selected_hp["weight_decay"],
    )
    full_train_loader = build_dataloader(clip_df, condition_name, runtime_config, mode="train")

    t0 = time.time()
    final_model = train_fixed_epochs(full_train_loader, runtime_config, selected_hp["final_epoch_count"])
    elapsed = time.time() - t0

    torch.save(
        {
            "model_state_dict": {k: v.cpu() for k, v in final_model.state_dict().items()},
            "architecture": architecture_key,
            "condition_name": condition_name,
            "selected_hp": selected_hp,
            "seed": SEED,
        },
        out_dir / "final_model.pth",
    )
    with open(out_dir / "retrain_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": SEED,
                "cudnn_deterministic": True,
                "n_train_images": len(full_train_loader.dataset),
                "epochs": selected_hp["final_epoch_count"],
                "train_seconds": round(elapsed, 1),
                "hp_source": "selected_hp.json (tune_and_develop.py)",
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"[{architecture_key}] 保存: {out_dir / 'final_model.pth'}（{elapsed / 60:.1f} 分）")


def main(config_path: str, architecture: str | None, force: bool) -> int:
    config = load_config(config_path)
    condition_name = build_condition_name(config["selected_condition"])

    Path(config["results_root"]).mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, Path(config["results_root"]) / "config_external_snapshot.yaml")

    ensure_dev_images(config, condition_name)
    clip_df = read_clip_metadata(config["dev"]["clip_metadata_csv"])

    architectures = [architecture] if architecture else list(config["architectures"].keys())
    for arch_key in architectures:
        print(f"\n{'=' * 60}\nアーキテクチャ: {arch_key}\n{'=' * 60}")
        run_architecture(config, arch_key, condition_name, clip_df, force)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="HP調整あり temporal validation: 最終モデルのシード固定再学習")
    parser.add_argument("--config", default="config_external.yaml")
    parser.add_argument("--architecture", default=None)
    parser.add_argument("--force", action="store_true", help="既存の final_model.pth を上書きする")
    args = parser.parse_args()
    sys.exit(main(args.config, args.architecture, args.force))
