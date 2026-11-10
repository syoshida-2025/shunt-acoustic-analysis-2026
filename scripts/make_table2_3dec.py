"""scripts/make_table2_3dec.py

v4 追加（2026-10-27）: 内部検証 27 条件の mean ± SD 表（論文 Table 2）を小数第3位で作る。

2026-10-24 に作った D:\\...\\make_table_3dec.py と同じ集計で、入力を v2 の学習結果
（results/metrics/all_metrics_27conditions_aug100.csv）に直接向けたもの。
SD は 30 分割（10 repeat x 3 fold）の標本標準偏差（ddof=1）。

実行（プロジェクトのルートで）:
    uv run scripts/make_table2_3dec.py [小数桁数(既定3)]
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "metrics" / "all_metrics_27conditions_aug100.csv"
DECIMALS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
OUT_DIR = ROOT / "results" / "tables"
OUT = OUT_DIR / f"table2_27conditions_{DECIMALS}dec"

METRICS = {
    "auc_recording": "mean AUC",
    "f1_recording": "mean F1",
    "sensitivity_recording": "mean Sens",
    "specificity_recording": "mean Spec",
    "accuracy_recording": "mean Accuracy",
}

df = pd.read_csv(SRC)
g = df.groupby(["fmax", "n_fft", "n_mels"])[list(METRICS)]
mean, sd = g.mean(), g.std(ddof=1)

table = pd.DataFrame(index=mean.index)
for col, label in METRICS.items():
    table[label] = [f"{m:.{DECIMALS}f} ± {s:.{DECIMALS}f}" for m, s in zip(mean[col], sd[col])]
table = table.reset_index()
table.insert(0, "No. condition", range(1, len(table) + 1))
table["n"] = g.size().values

OUT_DIR.mkdir(parents=True, exist_ok=True)
table.to_csv(OUT.with_suffix(".csv"), index=False, encoding="utf-8-sig")
table.to_excel(OUT.with_suffix(".xlsx"), index=False)
print(table.to_string(index=False))
