"""visualize_metrics_fixed.py

Top10条件の一覧表（results/top10_conditions.png）と、全27条件のAUCランキング図
（results/all_conditions_ranking.png）を生成する。visualize_metrics.py と入力は
共通だが出力形式が異なる。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results/metrics/all_metrics_27conditions_aug100.csv')

# 統計テーブル
auc_summary = df.groupby('condition_name')['auc_recording'].agg(['mean', 'std', 'min', 'max']).sort_values('mean', ascending=False)

fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')

# トップ10
top10 = auc_summary.head(10).reset_index()
top10.columns = ['Condition', 'Mean AUC', 'Std Dev', 'Min', 'Max']
top10['Rank'] = range(1, 11)
top10 = top10[['Rank', 'Condition', 'Mean AUC', 'Std Dev', 'Min', 'Max']]

table = ax.table(cellText=top10.round(4).values, 
                colLabels=top10.columns, 
                cellLoc='center', 
                loc='center',
                colWidths=[0.08, 0.25, 0.12, 0.12, 0.12, 0.12])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

for i in range(len(top10.columns)):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

plt.title('Top 10 Conditions by AUC (Recording)\nEach row represents 30 CV splits (10-repeat x 3-fold)', 
          fontsize=13, pad=20, weight='bold')
plt.savefig('results/top10_conditions.png', dpi=100, bbox_inches='tight')
print("Graph saved: results/top10_conditions.png")
plt.close()

# 図3: 全条件の分布
fig, ax = plt.subplots(figsize=(14, 6))
auc_values = df.groupby('condition_name')['auc_recording'].mean().sort_values(ascending=False)
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(auc_values)))
ax.barh(range(len(auc_values)), auc_values.values, color=colors)
ax.set_yticks(range(len(auc_values)))
ax.set_yticklabels(auc_values.index, fontsize=8)
ax.set_xlabel('Mean AUC (Recording)', fontsize=11)
ax.set_title('All 27 Conditions - Mean AUC Ranking', fontsize=12, weight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('results/all_conditions_ranking.png', dpi=100, bbox_inches='tight')
print("Graph saved: results/all_conditions_ranking.png")
plt.close()

print("\n" + "="*80)
print("RESULT SUMMARY - AUC (Recording)")
print("="*80)
print(f"\nBest condition: {auc_summary.index[0]}")
print(f"  Mean AUC: {auc_summary.iloc[0]['mean']:.4f} ± {auc_summary.iloc[0]['std']:.4f}")
print(f"  Range: {auc_summary.iloc[0]['min']:.4f} - {auc_summary.iloc[0]['max']:.4f}")

print(f"\nWorst condition: {auc_summary.index[-1]}")
print(f"  Mean AUC: {auc_summary.iloc[-1]['mean']:.4f} ± {auc_summary.iloc[-1]['std']:.4f}")
print(f"  Range: {auc_summary.iloc[-1]['min']:.4f} - {auc_summary.iloc[-1]['max']:.4f}")

print(f"\nPerformance gap: {(auc_summary.iloc[0]['mean'] - auc_summary.iloc[-1]['mean'])*100:.2f}%")
print(f"Overall mean AUC across all conditions: {df['auc_recording'].mean():.4f}")
print(f"Overall std: {df['auc_recording'].std():.4f}")
