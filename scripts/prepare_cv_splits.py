"""scripts/prepare_cv_splits.py

3-fold × 10 repeat = 30 個のCV分割を事前計算する。

StratifiedGroupKFold で患者ID単位の分割を生成し、
splits.json と split_summary.csv に保存する。

このスクリプトはプロジェクト全体で1回だけ実行する。
変更したい場合（n_repeats, n_folds, random_state_base を変えた場合）は再実行する。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import read_clip_metadata, split_train_test_by_patient


def main(config_path: str) -> int:
    """エントリーポイント。"""
    config = load_config(config_path)

    # 入力
    try:
        clip_df = read_clip_metadata(config["paths"]["clip_metadata_csv"])
    except FileNotFoundError as e:
        print(f"ERROR: {e}\n先に generate_metadata.py を実行してください。", file=sys.stderr)
        return 1

    n_folds = config["cv"]["n_folds"]
    n_repeats = config["cv"]["n_repeats"]
    random_state_base = config["cv"]["random_state_base"]
    patient_id_col = config["cv"]["patient_id_col"]
    label_name = "fv_value"  # split_train_test_by_patient が要求するカラム名
    threshold_fv = config["metadata"]["threshold_fv"]
    test_size = config["cv"]["test_size"]

    # 出力先
    splits_json_path = Path(config["paths"]["cv_splits_json"])
    splits_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_csv_path = splits_json_path.parent / "split_summary.csv"

    # 全分割を生成
    all_splits: list[dict] = []
    summary_rows: list[dict] = []
    print(f"[prepare_cv_splits] {n_repeats} repeats × {n_folds} folds = "
          f"{n_repeats * n_folds} 分割を生成中...")

    for repeat_id in range(1, n_repeats + 1):
        random_state = random_state_base + repeat_id - 1
        folds = list(split_train_test_by_patient(
            df=clip_df,
            patient_id_col=patient_id_col,
            label_name=label_name,
            threshold=threshold_fv,
            test_size=test_size,
            random_state=random_state,
            use_yield=True,
        ))

        if len(folds) != n_folds:
            print(
                f"WARNING: repeat {repeat_id} で {n_folds} fold を期待しましたが "
                f"{len(folds)} fold が生成されました（test_size 設定確認）",
                file=sys.stderr,
            )

        for fold_id, (train_df, test_df) in enumerate(folds, start=1):
            # 患者重複チェック
            train_patients = set(train_df[patient_id_col].tolist())
            test_patients = set(test_df[patient_id_col].tolist())
            overlap = train_patients & test_patients
            if overlap:
                print(
                    f"ERROR: repeat {repeat_id} fold {fold_id} で患者重複あり: {overlap}",
                    file=sys.stderr,
                )
                return 1

            # 分割エントリを生成
            entry = {
                "repeat_id": repeat_id,
                "fold_id": fold_id,
                "random_state": random_state,
                "train_patients": sorted(train_patients),
                "test_patients": sorted(test_patients),
                "train_clips": sorted(train_df["clip_file"].tolist()),
                "test_clips": sorted(test_df["clip_file"].tolist()),
                "n_train_patients": len(train_patients),
                "n_test_patients": len(test_patients),
                "n_train_clips": len(train_df),
                "n_test_clips": len(test_df),
                "n_train_positive": int((train_df["label"] == 1).sum()),
                "n_train_negative": int((train_df["label"] == 0).sum()),
                "n_test_positive": int((test_df["label"] == 1).sum()),
                "n_test_negative": int((test_df["label"] == 0).sum()),
                "overlap_check": "passed",
            }
            all_splits.append(entry)

            summary_rows.append({
                "repeat_id": repeat_id,
                "fold_id": fold_id,
                "random_state": random_state,
                "n_train_patients": entry["n_train_patients"],
                "n_test_patients": entry["n_test_patients"],
                "n_train_clips": entry["n_train_clips"],
                "n_test_clips": entry["n_test_clips"],
                "train_pn_ratio": entry["n_train_positive"] / max(1, entry["n_train_negative"]),
                "test_pn_ratio": entry["n_test_positive"] / max(1, entry["n_test_negative"]),
            })

    # splits.json 書き出し
    splits_data = {
        "metadata": {
            "n_repeats": n_repeats,
            "n_folds": n_folds,
            "random_state_base": random_state_base,
            "patient_id_col": patient_id_col,
            "label_col": "label",
            "fv_threshold": threshold_fv,
            "test_size": test_size,
            "n_total_clips": len(clip_df),
            "n_total_patients": clip_df[patient_id_col].nunique(),
            "generated_at": datetime.now().isoformat(),
        },
        "splits": all_splits,
    }
    with open(splits_json_path, "w", encoding="utf-8") as f:
        json.dump(splits_data, f, ensure_ascii=False, indent=2)
    print(f"[prepare_cv_splits] {splits_json_path} に {len(all_splits)} 分割を保存しました")

    # split_summary.csv 書き出し
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8")
    print(f"[prepare_cv_splits] {summary_csv_path} にサマリーを保存しました")

    # コンソールサマリー
    print("\n" + "=" * 60)
    print("CV分割サマリー")
    print("=" * 60)
    print(f"  Repeat数: {n_repeats}")
    print(f"  Fold数/repeat: {n_folds}")
    print(f"  総分割数: {len(all_splits)}")
    print(f"  患者ID重複チェック: 全分割でパス")
    print(f"  訓練患者数（平均）: {summary_df['n_train_patients'].mean():.1f}")
    print(f"  テスト患者数（平均）: {summary_df['n_test_patients'].mean():.1f}")
    print(f"  訓練クリップ数（平均）: {summary_df['n_train_clips'].mean():.1f}")
    print(f"  テストクリップ数（平均）: {summary_df['n_test_clips'].mean():.1f}")
    print("=" * 60 + "\n")

    return 0


def load_config(config_path: str) -> dict:
    """config.yaml を読み込んで dict として返す。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CV分割事前計算スクリプト")
    parser.add_argument("--config", default="config.yaml", help="config.yaml へのパス")
    args = parser.parse_args()

    sys.exit(main(args.config))
