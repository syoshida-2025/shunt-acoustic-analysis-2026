"""scripts/generate_metadata.py

データ準備の第一段階：data_v3.xlsx と生WAVファイルから、
クリップレベルのメタデータ（clip_metadata.csv）と
録音レベルの拡張メタデータ（recording_metadata.csv）を生成する。

このスクリプトはプロジェクト全体で1回だけ実行する。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import librosa
from tqdm import tqdm

# utils モジュールのパスを追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    read_recording_metadata,
    write_recording_metadata,
    write_clip_metadata,
    validate_audio_files_exist,
    write_missing_data_report,
)


def main(config_path: str) -> int:
    """エントリーポイント。

    Args:
        config_path: config.yaml へのパス。

    Returns:
        終了コード（0: 成功、1: エラー）。
    """
    # 1. config 読み込み
    config = load_config(config_path)

    # 2. data_v3.xlsx 読み込み（metadata_io 内で検証実施）
    try:
        df = read_recording_metadata(
            xlsx_path=config["paths"]["metadata_xlsx"],
            sheet_name=config["metadata"]["sheet_name"],
            threshold_fv=config["metadata"]["threshold_fv"],
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"[generate_metadata] {len(df)} 件の録音を読み込みました。"
          f"ユニーク患者数: {df['patient_id'].nunique()}")

    # 3. 欠損値ポリシー適用
    try:
        df, missing_report = apply_missing_value_policy(
            df=df,
            threshold_fv=config["metadata"]["threshold_fv"],
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # 4. ファイル存在チェック
    print(f"[generate_metadata] {config['paths']['audio_dir']} とのクロスチェック中...")
    try:
        missing_files, extra_files = validate_audio_files_exist(
            df=df,
            audio_dir=config["paths"]["audio_dir"],
        )
    except NotADirectoryError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if missing_files:
        print(
            f"ERROR: {len(missing_files)} 件のWAVファイルが見つかりません: "
            f"{missing_files[:5]}{'...' if len(missing_files) > 5 else ''}",
            file=sys.stderr,
        )
        return 1

    if extra_files:
        print(
            f"WARNING: {len(extra_files)} 件の余分なWAVファイルがあります（メタデータに記載なし）: "
            f"{extra_files[:5]}{'...' if len(extra_files) > 5 else ''}",
            file=sys.stderr,
        )
        missing_report["extra_files_on_disk"] = extra_files
    else:
        print(f"[generate_metadata]   ✓ メタデータの全ファイルがディスク上に存在します。")

    # 5. WAVファイルから派生カラムを計算
    df = compute_derived_columns(
        df=df,
        audio_dir=config["paths"]["audio_dir"],
        clipping_threshold=config["audio"]["clipping_threshold"],
        clipping_max_rate=config["audio"]["clipping_max_rate"],
        clip_duration_sec=config["audio"]["clip_duration_sec"],
        skip_initial_sec=config["audio"]["skip_initial_sec"],
    )

    # 6. 録音メタデータ書き出し
    recording_csv_path = Path(config["paths"]["clip_metadata_csv"]).parent / "recording_metadata.csv"
    write_recording_metadata(df, recording_csv_path)
    print(f"[generate_metadata] 録音メタデータを書き出しました: {recording_csv_path}")

    # 7. クリップメタデータを生成
    clip_df = build_clip_metadata(
        recording_df=df,
        clip_duration_sec=config["audio"]["clip_duration_sec"],
        skip_initial_sec=config["audio"]["skip_initial_sec"],
    )

    # 8. クリップメタデータ書き出し
    write_clip_metadata(clip_df, config["paths"]["clip_metadata_csv"])
    print(
        f"[generate_metadata] クリップメタデータを書き出しました: "
        f"{config['paths']['clip_metadata_csv']} ({len(clip_df)} クリップ)"
    )

    # 9. 欠損データレポート出力
    report_path = Path(config["paths"]["results_root"]) / "missing_data_report.txt"
    write_missing_data_report(
        report=missing_report,
        output_path=report_path,
    )
    print(f"[generate_metadata] 欠損データレポートを出力しました: {report_path}")

    # 10. クラス分布サマリー
    print_class_distribution_summary(df, clip_df, threshold_fv=config["metadata"]["threshold_fv"])

    return 0


def load_config(config_path: str) -> dict:
    """config.yaml を読み込んで dict として返す。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_missing_value_policy(
    df: pd.DataFrame,
    threshold_fv: float,
) -> tuple[pd.DataFrame, dict]:
    """欠損値ポリシーを適用し、レポート用のサマリーを返す。

    ポリシー:
        audio_file 欠損 → ERROR（例外）
        patient_id 欠損 → audio_file 先頭3桁で自動補完
        POD 欠損 → WARNING、NaN保持
        fv_value 欠損 → ERROR（例外）
        RI 欠損 → NaN保持
        label 欠損 → fv_value から自動計算
        sesson_id 欠損 → NaN保持

    Returns:
        (補完済みDataFrame, レポート用dict)

    Raises:
        ValueError: 必須カラム（audio_file, fv_value）に欠損がある場合。
    """
    report: dict = {
        "policy_applied": {},
        "errors": [],
        "warnings": [],
    }

    # audio_file 欠損チェック
    if df["audio_file"].isnull().any():
        missing_indices = df[df["audio_file"].isnull()].index.tolist()
        msg = f"audio_file 欠損: {len(missing_indices)} 件"
        report["errors"].append(msg)
        raise ValueError(msg)

    # fv_value 欠損チェック
    fv_missing = df["fv_value"].isnull()
    if fv_missing.any():
        missing_files = df.loc[fv_missing, "audio_file"].tolist()
        msg = f"fv_value 欠損: {len(missing_files)} 件 ({missing_files[:5]}...)"
        report["errors"].append(msg)
        raise ValueError(msg)

    # patient_id 欠損 → 自動補完
    pid_missing = df["patient_id"].isnull()
    if pid_missing.any():
        n_filled = int(pid_missing.sum())
        df = df.copy()
        df.loc[pid_missing, "patient_id"] = df.loc[pid_missing, "audio_file"].str[:3]
        msg = f"patient_id 欠損: {n_filled} 件を audio_file 先頭3桁で自動補完"
        print(f"INFO: {msg}")
        report["policy_applied"]["patient_id_filled"] = n_filled

    # label 欠損 → 自動計算
    label_missing = df["label"].isnull()
    if label_missing.any():
        n_calc = int(label_missing.sum())
        df = df.copy() if not pid_missing.any() else df
        df.loc[label_missing, "label"] = (
            df.loc[label_missing, "fv_value"] < threshold_fv
        ).astype(int)
        msg = f"label 欠損: {n_calc} 件を fv_value から自動計算"
        print(f"INFO: {msg}")
        report["policy_applied"]["label_calculated"] = n_calc

    # POD 欠損 → WARNING
    pod_missing = df["POD"].isnull()
    if pod_missing.any():
        n_pod = int(pod_missing.sum())
        files = df.loc[pod_missing, "audio_file"].tolist()
        msg = f"POD 欠損: {n_pod} 件（NaN保持） {files[:5]}..."
        print(f"WARNING: {msg}", file=sys.stderr)
        report["warnings"].append(msg)
        report["policy_applied"]["pod_missing"] = n_pod

    # RI 欠損 → 本研究で未使用
    ri_missing = df["RI"].isnull()
    if ri_missing.any():
        n_ri = int(ri_missing.sum())
        report["policy_applied"]["ri_missing"] = n_ri

    # sesson_id 欠損 → 将来用
    session_missing = df["sesson_id"].isnull()
    if session_missing.any():
        n_session = int(session_missing.sum())
        report["policy_applied"]["sesson_id_missing"] = n_session

    # label を int 型に統一
    df = df.copy()
    df["label"] = df["label"].astype(int)

    return df, report


def compute_derived_columns(
    df: pd.DataFrame,
    audio_dir: str | Path,
    clipping_threshold: float,
    clipping_max_rate: float,
    clip_duration_sec: float,
    skip_initial_sec: float,
) -> pd.DataFrame:
    """各WAVファイルから派生カラムを計算する。

    派生カラム:
        recording_duration_sec: 録音長（秒）
        clipping_rate: |sample| >= clipping_threshold のサンプル比率
        excluded: clipping_rate > clipping_max_rate なら True
        exclusion_reason: 除外理由（"high_clipping", "too_short", "" のいずれか）
        n_clips: 切り出し可能なクリップ数

    Args:
        df: 録音レベルのDataFrame。
        audio_dir: WAVファイルが置かれているディレクトリ。
        clipping_threshold: クリッピング判定の閾値（通常 0.99）。
        clipping_max_rate: 除外する最大クリッピング率（通常 0.01）。
        clip_duration_sec: クリップの長さ（秒、通常 5.0）。
        skip_initial_sec: 冒頭スキップ秒数（通常 1.0）。

    Returns:
        派生カラムが追加されたDataFrame。
    """
    audio_dir = Path(audio_dir)
    derived: dict[str, list] = {
        "recording_duration_sec": [],
        "clipping_rate": [],
        "excluded": [],
        "exclusion_reason": [],
        "n_clips": [],
    }

    for audio_file in tqdm(df["audio_file"], desc="WAVファイル解析"):
        wav_path = audio_dir / audio_file
        try:
            wave, sr = librosa.load(wav_path, sr=None, mono=True)
        except Exception as e:
            derived["recording_duration_sec"].append(np.nan)
            derived["clipping_rate"].append(np.nan)
            derived["excluded"].append(True)
            derived["exclusion_reason"].append(f"load_error: {type(e).__name__}")
            derived["n_clips"].append(0)
            continue

        duration_sec = len(wave) / sr
        clipping_rate = float(np.mean(np.abs(wave) >= clipping_threshold))

        excluded = False
        reason = ""
        if clipping_rate > clipping_max_rate:
            excluded = True
            reason = "high_clipping"
        elif duration_sec < skip_initial_sec + clip_duration_sec:
            excluded = True
            reason = "too_short"

        if excluded:
            n_clips = 0
        else:
            usable_duration = duration_sec - skip_initial_sec
            n_clips = int(np.floor(usable_duration / clip_duration_sec))

        derived["recording_duration_sec"].append(duration_sec)
        derived["clipping_rate"].append(clipping_rate)
        derived["excluded"].append(excluded)
        derived["exclusion_reason"].append(reason)
        derived["n_clips"].append(n_clips)

    df = df.copy()
    for col, vals in derived.items():
        df[col] = vals

    return df


def build_clip_metadata(
    recording_df: pd.DataFrame,
    clip_duration_sec: float,
    skip_initial_sec: float,
) -> pd.DataFrame:
    """録音メタデータからクリップレベルのメタデータを生成する。

    除外された録音はスキップ。各録音から n_clips 行を生成する。

    Args:
        recording_df: 派生カラムが計算済みの録音DataFrame。
        clip_duration_sec: クリップの長さ（秒）。
        skip_initial_sec: 冒頭スキップ秒数。

    Returns:
        クリップレベルのDataFrame。
    """
    clip_rows = []
    for _, rec in recording_df.iterrows():
        if rec["excluded"]:
            continue
        audio_basename = Path(rec["audio_file"]).stem
        for clip_idx in range(int(rec["n_clips"])):
            start_sec = skip_initial_sec + clip_idx * clip_duration_sec
            clip_rows.append({
                "clip_file": f"{audio_basename}_clip{clip_idx}.wav",
                "audio_file": rec["audio_file"],
                "patient_id": rec["patient_id"],
                "clip_idx": clip_idx,
                "start_sec": start_sec,
                "fv_value": rec["fv_value"],
                "label": int(rec["label"]),
                "POD": rec["POD"] if pd.notnull(rec["POD"]) else np.nan,
            })
    return pd.DataFrame(clip_rows)


def print_class_distribution_summary(
    recording_df: pd.DataFrame,
    clip_df: pd.DataFrame,
    threshold_fv: float,
) -> None:
    """録音レベルとクリップレベルのクラス分布をコンソール出力。"""
    print("\n" + "=" * 60)
    print("クラス分布サマリー")
    print("=" * 60)

    included = recording_df[~recording_df["excluded"]]
    print(f"\n[録音レベル] FV閾値 = {threshold_fv} mL/min")
    print(f"  全録音数: {len(recording_df)}")
    print(f"  除外: {int(recording_df['excluded'].sum())} 件")
    print(f"  解析対象録音数: {len(included)}")
    print(f"  ユニーク患者数: {included['patient_id'].nunique()}")
    print(f"  Positive (FV<{threshold_fv}): {int((included['label'] == 1).sum())}")
    print(f"  Negative (FV>={threshold_fv}): {int((included['label'] == 0).sum())}")
    n_neg = (included["label"] == 0).sum()
    n_pos = (included["label"] == 1).sum()
    if n_neg > 0:
        ratio = n_pos / n_neg
        print(f"  P/N比: {ratio:.3f}")

    print(f"\n[クリップレベル]")
    print(f"  全クリップ数: {len(clip_df)}")
    if len(clip_df) > 0:
        n_pos_clip = int((clip_df["label"] == 1).sum())
        n_neg_clip = int((clip_df["label"] == 0).sum())
        print(f"  Positive: {n_pos_clip}")
        print(f"  Negative: {n_neg_clip}")
        if n_neg_clip > 0:
            ratio = n_pos_clip / n_neg_clip
            print(f"  P/N比: {ratio:.3f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="メタデータ生成スクリプト")
    parser.add_argument("--config", default="config.yaml", help="config.yaml へのパス")
    args = parser.parse_args()

    sys.exit(main(args.config))
