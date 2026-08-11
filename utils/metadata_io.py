"""utils/metadata_io.py — shunt_melspec_v2 のメタデータI/Oヘルパー。

data_v3.xlsx と clip_metadata.csv に対するすべてのI/O処理をカプセル化し、
下流スクリプトをクリーンに保つ。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd


# ===== スキーマ定義 =====

REQUIRED_RECORDING_COLUMNS = [
    "audio_file",     # str: ファイル名, e.g. "A0101.wav"
    "patient_id",     # str: audio_fileの先頭3文字
    "sesson_id",      # nullable, 将来利用予定
    "POD",            # int: 術後日数
    "fv_value",       # float: mL/min
    "RI",             # float: 未使用、保持
    "label",          # int: 1 if fv_value < threshold else 0
]

DERIVED_RECORDING_COLUMNS = [
    "recording_duration_sec",   # float: WAVから取得
    "clipping_rate",            # float: |x| >= 0.99 のサンプル比率
    "excluded",                 # bool: clipping_rate > threshold の場合 True
    "exclusion_reason",         # str: 除外理由 "high_clipping", "" など
    "n_clips",                  # int: 切り出し可能なクリップ数
]

CLIP_METADATA_COLUMNS = [
    "clip_file",      # str: e.g. "A0101_clip0.wav"
    "audio_file",     # str: 元録音, e.g. "A0101.wav"
    "patient_id",     # str
    "clip_idx",       # int: 0, 1, 2, ...
    "start_sec",      # float: 録音内での開始時間
    "fv_value",       # float: 録音から継承
    "label",          # int: 録音から継承
    "POD",            # int: 録音から継承
]


# ===== 読み込み =====

def read_recording_metadata(
    xlsx_path: str | Path,
    sheet_name: str = "dataset",
    threshold_fv: float = 400.0,
) -> pd.DataFrame:
    """
    data_v3.xlsx から録音レベルのメタデータを読み込み、検証する。

    検証ステップ:
    1. REQUIRED_RECORDING_COLUMNS がすべて存在すること。
    2. audio_file 列に重複・NaN がないこと。
    3. patient_id が audio_file[:3] と一致すること（不一致は WARNING）。
    4. label が (fv_value < threshold_fv).astype(int) と一致すること（不一致は WARNING）。
    5. sesson_id は nullable、そのまま保持。

    Parameters:
    -----------
    xlsx_path : str | Path
        data_v3.xlsx へのパス
    sheet_name : str
        Excelシート名
    threshold_fv : float
        FV値の陽性クラス閾値

    Returns:
    --------
    pd.DataFrame
        元のカラムを持つDataFrame。バリデーション警告は stderr に出力。

    Raises:
    -------
    FileNotFoundError: xlsx_path が存在しない場合。
    ValueError: 必須カラムが欠損、audio_file に重複がある、null がある場合。
    """
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"メタデータファイルが見つかりません: {xlsx_path}")

    print(f"[metadata_io] {xlsx_path} を読み込み中 (sheet={sheet_name})...")
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    # ステップ1: 必須カラムチェック
    ensure_columns(df, REQUIRED_RECORDING_COLUMNS, context="data_v3.xlsx")

    # ステップ2: audio_file の整合性チェック
    if df["audio_file"].isnull().any():
        raise ValueError("audio_file カラムに null 値が含まれています。")
    duplicates = df[df["audio_file"].duplicated()]["audio_file"].tolist()
    if duplicates:
        raise ValueError(f"audio_file に重複エントリがあります: {duplicates}")

    print(f"[metadata_io] {len(df)} 件の録音を読み込みました。患者数: {df['patient_id'].nunique()}")

    # ステップ3: patient_id の整合性チェック（WARNING のみ）
    mismatches = validate_patient_id_consistency(df)
    if mismatches:
        print(
            f"WARNING: patient_id != audio_file[:3] の行が {len(mismatches)} 件あります: "
            f"{mismatches[:5]}{'...' if len(mismatches) > 5 else ''}",
            file=sys.stderr,
        )
    else:
        print("[metadata_io] patient_id 整合性チェック: ✓ (不一致なし)")

    # ステップ4: label の整合性チェック（WARNING のみ）
    label_mismatches = validate_label_consistency(df, threshold_fv)
    if label_mismatches:
        print(
            f"WARNING: label != (fv_value < {threshold_fv}) の行が "
            f"{len(label_mismatches)} 件あります: {label_mismatches[:5]}...",
            file=sys.stderr,
        )
    else:
        print(f"[metadata_io] label 整合性チェック (threshold={threshold_fv}): ✓ (不一致なし)")

    return df


def write_recording_metadata(
    df: pd.DataFrame,
    csv_path: str | Path,
) -> None:
    """
    録音レベルのメタデータ（元のカラム + 派生カラム）をCSVに書き出す。

    Parameters:
    -----------
    df : pd.DataFrame
        書き出すDataFrame
    csv_path : str | Path
        出力先CSVパス
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")


def read_clip_metadata(csv_path: str | Path) -> pd.DataFrame:
    """
    clip_metadata.csv からクリップレベルのメタデータを読み込む。

    Parameters:
    -----------
    csv_path : str | Path
        clip_metadata.csv へのパス

    Returns:
    --------
    pd.DataFrame
        CLIP_METADATA_COLUMNS を持つDataFrame

    Raises:
    -------
    FileNotFoundError: csv_path が存在しない場合。
    ValueError: 必須カラムが欠損している場合。
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"クリップメタデータが見つかりません: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8")
    ensure_columns(df, CLIP_METADATA_COLUMNS, context="clip_metadata.csv")
    return df


def write_clip_metadata(df: pd.DataFrame, csv_path: str | Path) -> None:
    """
    クリップレベルのメタデータをCSVに書き出す（カラム順を固定）。

    Parameters:
    -----------
    df : pd.DataFrame
        書き出すDataFrame
    csv_path : str | Path
        出力先CSVパス
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # カラム順を CLIP_METADATA_COLUMNS に合わせる（存在するもののみ）
    ordered_cols = [c for c in CLIP_METADATA_COLUMNS if c in df.columns]
    extra_cols = [c for c in df.columns if c not in CLIP_METADATA_COLUMNS]
    df[ordered_cols + extra_cols].to_csv(csv_path, index=False, encoding="utf-8")


# ===== バリデーションヘルパー =====

def validate_patient_id_consistency(df: pd.DataFrame) -> list[str]:
    """
    patient_id == audio_file[:3] であることをすべての行で確認する。

    Returns:
    --------
    list[str]
        不一致が検出された audio_file の値のリスト（問題なければ空リスト）
    """
    mismatches = []
    for _, row in df.iterrows():
        if pd.notnull(row["audio_file"]) and pd.notnull(row["patient_id"]):
            expected = str(row["audio_file"])[:3]
            if str(row["patient_id"]) != expected:
                mismatches.append(str(row["audio_file"]))
    return mismatches


def validate_label_consistency(
    df: pd.DataFrame,
    threshold_fv: float,
) -> list[str]:
    """
    label == int(fv_value < threshold_fv) であることをすべての行で確認する。

    Returns:
    --------
    list[str]
        ラベル不一致が検出された audio_file の値のリスト（問題なければ空リスト）
    """
    mismatches = []
    for _, row in df.iterrows():
        if pd.notnull(row["fv_value"]) and pd.notnull(row["label"]):
            expected_label = int(float(row["fv_value"]) < threshold_fv)
            if int(row["label"]) != expected_label:
                mismatches.append(str(row["audio_file"]))
    return mismatches


def validate_audio_files_exist(
    df: pd.DataFrame,
    audio_dir: str | Path,
) -> tuple[list[str], list[str]]:
    """
    df['audio_file'] とディスク上の実際のファイルをクロスチェックする。

    Parameters:
    -----------
    df : pd.DataFrame
        audio_file カラムを持つDataFrame
    audio_dir : str | Path
        WAVファイルが置かれているディレクトリ

    Returns:
    --------
    tuple[list[str], list[str]]
        (missing_in_disk, extra_on_disk):
        - missing_in_disk: dfに記載されているがディスクにないファイル名
        - extra_on_disk: ディスクにあるがdfに記載されていないファイル名
        両方ともアルファベット順にソート済み。

    Raises:
    -------
    NotADirectoryError: audio_dir がディレクトリでない場合。
    """
    audio_dir = Path(audio_dir)
    if not audio_dir.is_dir():
        raise NotADirectoryError(f"audio_dir がディレクトリではありません: {audio_dir}")

    files_in_df = set(df["audio_file"].tolist())
    files_on_disk = {p.name for p in audio_dir.iterdir() if p.suffix.lower() == ".wav"}

    missing = sorted(files_in_df - files_on_disk)
    extra = sorted(files_on_disk - files_in_df)
    return missing, extra


# ===== スキーマユーティリティ =====

def ensure_columns(
    df: pd.DataFrame,
    required: list[str],
    context: str = "",
) -> None:
    """
    必須カラムがすべて存在するかチェックする。欠損がなければ何もしない。

    Parameters:
    -----------
    df : pd.DataFrame
        チェック対象のDataFrame
    required : list[str]
        必須カラム名のリスト
    context : str
        エラーメッセージに含めるコンテキスト情報

    Raises:
    -------
    ValueError: 欠損カラムが1つ以上ある場合。
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        msg = f"必須カラムが欠損しています: {missing}"
        if context:
            msg = f"[{context}] {msg}"
        raise ValueError(msg)


def write_missing_data_report(report: dict, output_path: str | Path) -> None:
    """
    欠損データレポートをテキストファイルとして出力する。

    generate_metadata.py から呼ばれる。
    report は apply_missing_value_policy が返す dict で、以下のキーを持つ:
        - "policy_applied": dict (各処理の件数)
        - "errors": list[str]
        - "warnings": list[str]
        - "extra_files_on_disk": list[str] (オプション)

    Parameters:
    -----------
    report : dict
        欠損データポリシー適用結果
    output_path : str | Path
        出力先テキストファイルパス
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# 欠損データレポート", ""]
    lines.append("## 適用された欠損値ポリシー")
    for k, v in report.get("policy_applied", {}).items():
        lines.append(f"- {k}: {v} 件")
    if not report.get("policy_applied"):
        lines.append("- なし")
    lines.append("")

    lines.append("## エラー（処理が停止した場合）")
    for e in report.get("errors", []):
        lines.append(f"- {e}")
    if not report.get("errors"):
        lines.append("- なし")
    lines.append("")

    lines.append("## 警告（処理は継続）")
    for w in report.get("warnings", []):
        lines.append(f"- {w}")
    if not report.get("warnings"):
        lines.append("- なし")

    if report.get("extra_files_on_disk"):
        lines.append("")
        lines.append("## ディスク上にあるがメタデータに記載のないファイル")
        for f in report["extra_files_on_disk"][:20]:
            lines.append(f"- {f}")
        if len(report["extra_files_on_disk"]) > 20:
            lines.append(f"  ... 他 {len(report['extra_files_on_disk']) - 20} 件")

    output_path.write_text("\n".join(lines), encoding="utf-8")
