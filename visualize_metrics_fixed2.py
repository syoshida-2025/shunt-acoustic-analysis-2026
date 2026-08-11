"""visualize_metrics_fixed2.py

全27条件のAUCランキング図（results/all_conditions_ranking.png）とコンソール
サマリーを出力する。visualize_metrics_fixed.py の同グラフ部分の別実装であり、
機能は重複している（将来的に一本化を検討）。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('results/metrics/all_metrics_27conditions_aug100.csv')

# 図3: 全条件の分布
fig, ax = plt.subplots(figsize=(14, 6))
auc_summary = df.groupby('condition_name')['auc_recording'].mean().sort_values(ascending=False)
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(auc_summary)))
ax.barh(range(len(auc_summary)), auc_summary.values, color=colors)
ax.set_yticks(range(len(auc_summary)))
ax.set_yticklabels(auc_summary.index, fontsize=8)
ax.set_xlabel('Mean AUC (Recording)', fontsize=11)
ax.set_title('All 27 Conditions - Mean AUC Ranking', fontsize=12, weight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('results/all_conditions_ranking.png', dpi=100, bbox_inches='tight')
print("Graph saved: results/all_conditions_ranking.png")
plt.close()

# 統計テーブル
auc_stats = df.groupby('condition_name')['auc_recording'].agg(['mean', 'std', 'min', 'max']).sort_values('mean', ascending=False)

print("\n" + "="*80)
print("RESULT SUMMARY - AUC (Recording)")
print("="*80)
print(f"\nBest condition: {auc_stats.index[0]}")
print(f"  Mean AUC: {auc_stats.iloc[0]['mean']:.4f} ± {auc_stats.iloc[0]['std']:.4f}")
print(f"  Range: {auc_stats.iloc[0]['min']:.4f} - {auc_stats.iloc[0]['max']:.4f}")

print(f"\nWorst condition: {auc_stats.index[-1]}")
print(f"  Mean AUC: {auc_stats.iloc[-1]['mean']:.4f} ± {auc_stats.iloc[-1]['std']:.4f}")
print(f"  Range: {auc_stats.iloc[-1]['min']:.4f} - {auc_stats.iloc[-1]['max']:.4f}")

print(f"\nPerformance gap: {(auc_stats.iloc[0]['mean'] - auc_stats.iloc[-1]['mean'])*100:.2f}%")
print(f"Overall mean AUC across all conditions: {df['auc_recording'].mean():.4f}")
print(f"Overall std: {df['auc_recording'].std():.4f}")

print("\n" + "="*80)
print("TOP 10 CONDITIONS")
print("="*80)
print(auc_stats.head(10).round(4))
