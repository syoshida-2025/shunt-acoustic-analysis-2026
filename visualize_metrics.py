"""visualize_metrics.py

条件別AUCの平均+-SD、因子別主効果、箱ひげ図、交互作用プロットの4パネル図
（results/auc_analysis.png）と、Top10/Bottom10条件の一覧表（results/auc_summary_table.png）
を生成する。visualize_metrics_fixed.py / _fixed2.py と入力は共通だが出力形式が異なる。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap


def format_condition_label(condition_name: str) -> str:
    """条件名を短く読みやすい形式に整える。"""
    if not isinstance(condition_name, str):
        return str(condition_name)

    parts = condition_name.split('_')
    values = {}
    for part in parts:
        if part.startswith('nfft'):
            values['nfft'] = part[4:]
        elif part.startswith('mel'):
            values['mel'] = part[3:]
        elif part.startswith('fmax'):
            values['fmax'] = part[4:]

    if values:
        return f"nfft={values.get('nfft', '?')}, mel={values.get('mel', '?')}, fmax={values.get('fmax', '?')}"
    return condition_name


df = pd.read_csv('results/metrics/all_metrics_27conditions_aug100.csv')

# 英語フォント設定（表示崩れを防ぐため）
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# 図1: 条件別AUCの平均 + 標準偏差
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# グラフ1: 条件別 AUC (recording) - 棒グラフ
auc_summary = (
    df.groupby('condition_name')['auc_recording']
    .agg(['mean', 'std', 'min', 'max', 'count'])
    .sort_values('mean', ascending=False)
)
ax = axes[0, 0]
x_pos = np.arange(len(auc_summary))
condition_labels = [textwrap.fill(format_condition_label(name), width=18) for name in auc_summary.index]
ax.bar(x_pos, auc_summary['mean'], yerr=auc_summary['std'], capsize=5, alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels([])
for i, label in enumerate(condition_labels):
    ax.text(i, -0.02, label, rotation=35, ha='right', va='top', fontsize=7, clip_on=False)
ax.set_ylabel('AUC (Recording)', fontsize=10)
ax.set_title('Condition-wise AUC (Recording) - Mean +/- Std', fontsize=12)
ax.grid(axis='y', alpha=0.3)
ax.margins(x=0.01)
ax.set_ylim(0.65, 0.90)
fig.subplots_adjust(bottom=0.32, top=0.95, left=0.06, right=0.98)

# グラフ2: 因子別 AUC の平均
ax = axes[0, 1]
factor_effects = []
for factor in ['n_fft', 'n_mels', 'fmax']:
    factor_effect = df.groupby(factor)['auc_recording'].mean()
    for val in factor_effect.index:
        factor_effects.append({'Factor': factor, 'Value': str(val), 'AUC': factor_effect[val]})
factor_df = pd.DataFrame(factor_effects)
sns.barplot(data=factor_df, x='Factor', y='AUC', hue='Value', ax=ax, palette='Set2')
ax.set_ylabel('Mean AUC (Recording)', fontsize=10)
ax.set_title('Factor Main Effects', fontsize=12)
ax.set_ylim([0.7, 0.8])

# グラフ3: 箱ひげ図（因子別）
ax = axes[1, 0]
data_for_box = []
for factor in ['n_fft', 'n_mels', 'fmax']:
    for val in df[factor].unique():
        auc_vals = df[df[factor] == val]['auc_recording'].values
        for auc in auc_vals:
            data_for_box.append({'Factor': factor, 'Value': str(val), 'AUC': auc})
box_df = pd.DataFrame(data_for_box)
sns.boxplot(data=box_df, x='Factor', y='AUC', hue='Value', ax=ax, palette='Set2')
ax.set_ylabel('AUC (Recording)', fontsize=10)
ax.set_title('Distribution by Factor', fontsize=12)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# グラフ4: 相互作用プロット（n_mels vs fmax, n_fft=2048の場合）
ax = axes[1, 1]
n_fft_val = 2048
subset = df[df['n_fft'] == n_fft_val].groupby(['n_mels', 'fmax'])['auc_recording'].mean().reset_index()
for n_mels in sorted(subset['n_mels'].unique()):
    data = subset[subset['n_mels'] == n_mels].sort_values('fmax')
    ax.plot(data['fmax'], data['auc_recording'], marker='o', label=f'n_mels={n_mels}')
ax.set_xlabel('fmax', fontsize=10)
ax.set_ylabel('Mean AUC (Recording)', fontsize=10)
ax.set_title(f'Interaction Plot: n_mels vs fmax (n_fft={n_fft_val})', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

plt.savefig('results/auc_analysis.png', dpi=150, bbox_inches='tight', pad_inches=0.3)
print("Graph saved: results/auc_analysis.png")
plt.close()

# 図2: 統計サマリーテーブル
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')

# トップ10とボトム10
summary_table = pd.concat([
    auc_summary.head(10).reset_index().assign(Rank='Top 10').head(10),
    auc_summary.tail(10).reset_index().assign(Rank='Bottom 10').head(10)
])
summary_table = summary_table[['Rank', 'condition_name', 'mean', 'std', 'min', 'max']]
summary_table.columns = ['Rank', 'Condition', 'Mean AUC', 'Std Dev', 'Min', 'Max']
summary_table['Condition'] = summary_table['Condition'].apply(format_condition_label)
summary_table['Condition'] = summary_table['Condition'].apply(lambda x: textwrap.fill(x, width=22))

# Ensure the table is visibly readable with explicit English text.
ax.set_title('Top 10 and Bottom 10 Conditions by AUC (Recording)', fontsize=14, pad=20)
ax.text(0.5, 0.97, 'English condition labels are shown below', transform=ax.transAxes,
        ha='center', va='top', fontsize=9, color='#444444')

numeric_columns = ['Mean AUC', 'Std Dev', 'Min', 'Max']
summary_table[numeric_columns] = summary_table[numeric_columns].round(4)

cell_values = summary_table.astype({'Rank': str, 'Condition': str, 'Mean AUC': str, 'Std Dev': str, 'Min': str, 'Max': str}).values

table = ax.table(cellText=cell_values, 
                colLabels=summary_table.columns, 
                cellLoc='center', 
                loc='center',
                colWidths=[0.08, 0.28, 0.12, 0.12, 0.12, 0.12])
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.35)

# ヘッダーを強調
for i in range(len(summary_table.columns)):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

plt.title('Top 10 and Bottom 10 Conditions by AUC (Recording)', fontsize=14, pad=20)
fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02)
plt.savefig('results/auc_summary_table.png', dpi=150, bbox_inches='tight', pad_inches=0.25)
print("Graph saved: results/auc_summary_table.png")
plt.close()

print("\n" + "="*80)
print("RESULT SUMMARY")
print("="*80)
print(f"\nBest condition: {auc_summary.index[0]}")
print(f"  Mean AUC: {auc_summary.iloc[0]['mean']:.4f} ± {auc_summary.iloc[0]['std']:.4f}")
print(f"\nWorst condition: {auc_summary.index[-1]}")
print(f"  Mean AUC: {auc_summary.iloc[-1]['mean']:.4f} ± {auc_summary.iloc[-1]['std']:.4f}")
print(f"\nPerformance gap: {(auc_summary.iloc[0]['mean'] - auc_summary.iloc[-1]['mean'])*100:.2f}%")

