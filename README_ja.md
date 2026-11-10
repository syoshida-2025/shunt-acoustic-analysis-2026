# shunt-acoustic-analysis-2026（日本語版）

[English](README.md)

透析シャント音（AVF の聴診音）を深層学習で分類する際の、メルスペクトログラム生成パラメータの組み合わせを
比較し、最良条件を temporal validation で検証した研究のコードです。

- **内部検証**：メルスペクトログラム 27 条件（`n_fft` × `n_mels` × `fmax` = 3 × 3 × 3）について、
  ResNet200d を患者単位の 10 回反復 3 分割交差検証（1 条件あたり 30 分割）で評価し、
  交差検証分割をブロック因子とした分散分析と Tukey HSD 法で比較します。
- **Temporal validation**：選択した条件（`n_fft=512`, `n_mels=256`, `fmax=1000`）について、
  開発データでハイパーパラメータを調整し、開発データ全例で最終モデルを学習したうえで、
  開発データで固定した閾値を用いて時間的に独立したコホートに 1 回だけ適用します
  （ResNet200d と EfficientNetV2-S）。ResNet200d については Grad-CAM も計算します。

陽性クラス：上腕動脈血流量（FV）< 400 mL/min。

> **データについて**：患者のプライバシー保護のため、音声データと臨床データは本リポジトリに含まれていません。
> 公開しているのは、コード、設定ファイル、患者単位の交差検証分割の定義（`cv_splits/`、匿名化した連番 ID）のみです。

---

## 1. 環境構築

必要なもの：Python 3.10〜3.12、[uv](https://docs.astral.sh/uv/)、NVIDIA GPU
（ロックファイルは CUDA 12.8 版の PyTorch を指定しています）。

```bash
git clone https://github.com/syoshida-2025/shunt-acoustic-analysis-2026.git
cd shunt-acoustic-analysis-2026
uv sync
```

以下のコマンドはすべて `uv run` で実行します（`uv sync` で作成したプロジェクト環境が使われます）。

## 2. 想定するデータ配置（リポジトリには含まれません）

```
data/
├── data_v3.xlsx          # シート "dataset": audio_file, patient_id, POD, fv_value, RI など
├── all_sounds/           # 元の WAV ファイル（4 kHz、16 bit、モノラル）
└── all_clips/            # 5 秒クリップ（create_clipped_audio.py で生成）
metadata/                 # generate_metadata.py / create_clipped_audio.py で生成
temporal_validation/data/
├── external_data_v3.xlsx # temporal validation cohort（同じ形式）
├── all_sounds/
├── all_clips/
└── metadata/
```

パスは `config.yaml`（内部検証）と `temporal_validation/hp_tuned/config_external.yaml`
（temporal validation）で設定します。

## 3. ファイル構成

```
├── config.yaml                 # 内部検証の設定（27 条件、データ拡張、交差検証、学習）
├── cv_splits/                  # 事前に作成した患者単位の交差検証分割（10 反復 × 3 分割）
├── scripts/
│   ├── generate_metadata.py        # xlsx → 録音・クリップのメタデータ（5 秒クリップ、冒頭 1 秒除外、重なりなし、1 録音あたり最大 2）
│   ├── create_clipped_audio.py     # 5 秒クリップの WAV を書き出す
│   ├── prepare_cv_splits.py        # 患者単位の反復交差検証分割
│   ├── generate_melspectrogram.py  # 1 条件分のメルスペクトログラム画像
│   ├── train_model.py              # 全条件 × 全分割の学習・評価（中断後に再開可能）
│   ├── analyze_statistics.py       # 分散分析（split_id をブロック因子）+ Tukey HSD
│   ├── make_table2_3dec.py         # Table 2（条件ごとの平均 ± SD）
│   ├── visualize_metrics.py        # AUC の集計図
│   └── generate_27condition*_examples.py  # 27 条件の画像例
├── utils/                      # データ読み込み、データ拡張、学習、評価指標、解析
├── tests/                      # 単体テスト（pytest）
└── temporal_validation/hp_tuned/
    ├── config_external.yaml            # 選択条件、HP 探索範囲、パス
    ├── config_external_metadata.yaml   # temporal validation cohort のメタデータ・クリップ生成用
    ├── tune_and_develop.py             # 開発データでの HP 探索（3 分割交差検証）と閾値の固定
    ├── retrain_final_seeded.py         # 開発データ全例での最終モデル学習（seed 42）
    ├── evaluate_external_noisy.py      # temporal validation cohort への 1 回適用（主解析）
    ├── evaluate_external.py            # temporal validation cohort への 1 回適用（補足）
    ├── analyze_external.py             # Table 4（95% CI、Brier score、較正）、ROC 曲線・較正図
    ├── plot_external_roc.py            # ROC 曲線
    ├── gradcam_resnet.py               # ResNet200d の Grad-CAM（gradcam_common.py を使用）
    └── run_a1.sh                       # 再学習 → 適用 → 解析 → Grad-CAM を順に実行
```

## 4. 内部検証（27 条件）

### 4.1 データ準備と交差検証分割

```bash
uv run scripts/generate_metadata.py --config config.yaml
uv run scripts/create_clipped_audio.py --config config.yaml
uv run scripts/prepare_cv_splits.py --config config.yaml
```

本リポジトリの `cv_splits/splits.json` は、本研究で実際に用いた分割の定義です。

### 4.2 学習と評価

```bash
uv run scripts/train_model.py --config config.yaml
```

- 27 条件 × 30 分割 = 810 モデル。画像は 1 条件ごとに生成 → 使用 → 削除します
  （`execution.delete_images_after_training`）。
- 進捗は `results/status.json` に保存され、中断しても続きから再開できます。
- 出力：`results/metrics/all_metrics.csv`、`results/predictions/{condition}.csv`、`results/loss_curves/`。

### 4.3 複数回に分けて実行した結果の統合

本研究では 27 条件を複数回に分けて実行しました。別々の実行で得られた評価指標の CSV を統合します
（`condition_name + repeat_id + fold_id` で重複を除外）。この手順はコードを標準入力で渡すため、
`uv run python -` で実行する必要があります。

```bash
uv run python - <<'PY'
from pathlib import Path
from utils.analysis import combine_metrics_csvs

combined = combine_metrics_csvs(
    [
        Path('results_run1/metrics/all_metrics.csv'),
        Path('results_run2/metrics/all_metrics.csv'),
    ],
    output_path='results/metrics/all_metrics_27conditions_aug100.csv',
)
print(f'rows={len(combined)} conditions={combined["condition_name"].nunique()}')
PY
```

### 4.4 統計解析・表・図

```bash
uv run scripts/analyze_statistics.py --input results/metrics/all_metrics_27conditions_aug100.csv --output_dir results/statistics
uv run scripts/make_table2_3dec.py
uv run scripts/visualize_metrics.py
```

`analyze_statistics.py` は既定で `auc_recording ~ C(n_fft) + C(n_mels) + C(fmax) + C(split_id)` に
すべての 2 因子交互作用を加えたモデルを当てはめます（Type II 平方和）。30 分割は全条件で共通のため、
`split_id` をブロック因子としています。

## 5. Temporal validation

`temporal_validation/hp_tuned/` で実行します。

```bash
cd temporal_validation/hp_tuned

# temporal validation cohort のメタデータとクリップ
uv run ../../scripts/generate_metadata.py --config config_external_metadata.yaml
uv run ../../scripts/create_clipped_audio.py --config config_external_metadata.yaml

# 開発データでの HP 探索、エポック数と閾値の固定
uv run tune_and_develop.py

# 最終モデル、temporal validation cohort への 1 回適用、解析、Grad-CAM
bash run_a1.sh
```

`run_a1.sh` は次の順に実行します：`retrain_final_seeded.py`、`evaluate_external_noisy.py`、
`evaluate_external.py`、`analyze_external.py --images noisy|clean`、
`gradcam_resnet.py --images noisy|clean`。

設計上の要点：

- ハイパーパラメータ（学習率 × 重み減衰のグリッド）、エポック数（HP 探索時の best epoch の中央値）、
  音源単位の閾値は開発データで決めて固定し、temporal validation cohort では閾値を再計算しません。
- Temporal validation の結果は点推定値と 95% 信頼区間で示します。

## 6. テスト

```bash
uv run pytest tests/ -v
```

## 7. 再現性に関する注意

- 内部検証の学習では乱数シードを固定していないため、再実行すると数値はわずかに変わります。
  Temporal validation の最終モデルは seed 42 で学習しています。
- 解析対象：153 音源（長さ不足の 1 音源を除外）、282 クリップ、61 患者（陽性 49 音源・陰性 104 音源）。
  Temporal validation cohort：15 患者、38 音源、75 クリップ。
- コード内のコメントは主に日本語です。
