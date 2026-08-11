"""analyze_metrics.py

results/metrics/all_metrics*.csv の内容を素早く確認するためのデバッグ用スクリプト。
条件別・因子別のAUC集計をコンソールに出力するだけで、ファイルの保存は行わない。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# CSVを読み込み（入力ファイル名は実行時の統合済みCSVに合わせて変更すること）
df = pd.read_csv('results/metrics/all_metrics_27conditions_aug100.csv')

print("=" * 80)
print("メトリクス概要")
print("=" * 80)
print(f"総行数: {len(df)}")
print(f"条件数: {df['condition_name'].nunique()}")
print(f"各条件の分割数: {df.groupby('condition_name').size().unique()}")
print()

# 主指標: auc_recording（音源レベル）で条件ごとに集計
auc_summary = df.groupby('condition_name')['auc_recording'].agg([
    ('mean', 'mean'),
    ('std', 'std'),
    ('min', 'min'),
    ('max', 'max'),
    ('n', 'count')
]).round(4).sort_values('mean', ascending=False)

print("=" * 80)
print("条件別 AUC_Recording (音源レベル主指標) の統計")
print("=" * 80)
print(auc_summary)
print()

# トップ5とボトム5
print("TOP 5:")
print(auc_summary.head())
print("\nBOTTOM 5:")
print(auc_summary.tail())
print()

# 3つの因子（n_fft, n_mels, fmax）の効果を確認
print("=" * 80)
print("因子別の効果分析")
print("=" * 80)
for factor in ['n_fft', 'n_mels', 'fmax']:
    factor_effect = df.groupby(factor)['auc_recording'].agg(['mean', 'std', 'count']).round(4)
    print(f"\n{factor}の効果:")
    print(factor_effect)

