# shunt_melspec_v3

シャント音メルスペクトログラムの実験パイプラインです。
本リポジトリは `n_fft × n_mels × fmax` の27条件を比較し、患者レベルの反復交差検証と統計解析(ANOVA / Tukey HSD / 線形混合効果モデル)によって、分類性能に影響するメルスペクトログラム設定を評価します。

`config.yaml` は既定で27条件フル実行(`n_fft` 3水準 × `n_mels` 3水準 × `fmax` 3水準)の設定になっています。実行時間の都合で条件を分けて実行したい場合は「5. 複数回に分けて実行した場合の結果統合」を参照してください。

このリポジトリのパイプラインは2つあります:
- **本体(このREADME)**: 27条件のメルスペクトログラム設定を比較する内部検証パイプライン(反復CV、ハイパーパラメータ固定)
- **[`external_validation/`](external_validation/README.md)**: 内部検証で選ばれた単一条件について、ハイパーパラメータを調整したモデルを開発データで構築し、外部hold-outデータに1回だけ適用する外部検証パイプライン(独立したconfig・スクリプト、詳細はリンク先のREADME参照)

---

## 1. 共有対象と構成

### リポジトリに含まれるもの

- `scripts/` : 実行パイプライン
- `utils/` : データ処理・モデル学習・評価のロジック
- `tests/` : 機能検証・回帰テスト
- `cv_splits/` : 再現性のある CV 分割定義(`splits.json` / `split_summary.csv`)
- `config.yaml`, `pyproject.toml`, `uv.lock`, `README.md`
- `visualize_metrics.py`, `visualize_metrics_fixed.py`, `visualize_metrics_fixed2.py`
- `analyze_metrics.py`, `check_dataset.py`

### リポジトリに含まれないもの(`.gitignore` 参照)

- `data/`(元音声 WAV、`data_v3.xlsx` などの臨床メタデータ)
- `metadata/*.csv`(patient_id は匿名化済みだが、fv_value・POD・RI 等の臨床測定値を含む)
- `results/predictions/`(clip_file・patient_id・true_label が紐づく患者単位の予測結果)
- `results/loss_curves/`, `results/status.json`, `results/run.log`(容量が大きい・再現に必須ではない実行ログ)
- `results/config_snapshot.yaml`(実行時の `config.yaml` の複製。特定バッチの内部記録であり、`config.yaml` 本体があれば十分なため公開リポジトリには含めない)
- `.venv/`, `__pycache__/`, `.pytest_cache/` などの環境依存ファイル
- 研究メモ(方法・結果セクションの草稿、実験ノート等)は本リポジトリの対象外としています

`cv_splits/splits.json` は患者ID(匿名化済みの通し番号)と fold 割り当てを含みますが、臨床測定値そのものは含まないためリポジトリに含めています。

### データスキーマについて

元データ(`data/data_v3.xlsx`)自体は共有しませんが、必要なカラム構成は `config.yaml` の `metadata.required_columns` に定義されています:

| カラム名 | 型 | 説明 |
|---|---|---|
| `audio_file` | str | WAVファイル名(例: `A0101.wav`) |
| `patient_id` | str | 患者ID(匿名化済み、audio_file 先頭3文字から自動補完可) |
| `sesson_id` | nullable | 将来利用予定、現状未使用 |
| `POD` | int | 術後日数 |
| `fv_value` | float | 血流量(mL/min) |
| `RI` | float | 抵抗係数(本研究では未使用) |
| `label` | int | `1` if `fv_value < threshold_fv` else `0`(閾値は `metadata.threshold_fv`、既定 400 mL/min) |

欠損値の扱いは [scripts/generate_metadata.py](scripts/generate_metadata.py) の `apply_missing_value_policy()` を参照してください。同じスキーマの xlsx を用意すれば、以下の手順でパイプラインを再現できます。

---

## 2. ディレクトリ構成

```
shunt_melspec_v3/
├── config.yaml
├── pyproject.toml
├── scripts/
│   ├── generate_metadata.py
│   ├── create_clipped_audio.py
│   ├── prepare_cv_splits.py
│   ├── generate_melspectrogram.py
│   ├── train_model.py
│   ├── analyze_statistics.py
│   └── analyze_interactions.py
├── utils/
│   ├── augmentation.py
│   ├── checkpoint.py
│   ├── data_split.py
│   ├── dataset.py
│   ├── df_from_folder.py
│   ├── metadata_io.py
│   ├── metrics.py
│   ├── sound2image.py
│   ├── trainer.py
│   └── analysis.py
├── tests/
│   ├── test_checkpoint.py
│   ├── test_combine_metrics.py
│   ├── test_data_layer.py
│   ├── test_interaction_analysis.py
│   └── test_metrics.py
├── cv_splits/
│   ├── splits.json
│   └── split_summary.csv
├── visualize_metrics.py
├── visualize_metrics_fixed.py
├── visualize_metrics_fixed2.py
├── analyze_metrics.py
├── check_dataset.py
└── README.md
```

---

## 3. 準備

### 前提条件

- Python 3.10–3.12
- `uv` がインストール済み
- NVIDIA GPU(CUDA 対応)推奨
- 元データは別途用意(`data/data_v3.xlsx`, `data/all_sounds/` を「1. データスキーマについて」の形式で配置)

### 依存パッケージのインストール

```bash
cd shunt_melspec_v3
uv sync
```

`pyproject.toml` は CUDA 12.8 向けの PyTorch インデックスを明示的に指定しています(RTX 5060 等 Blackwell 世代 GPU 向け)。異なる CUDA バージョンの環境では `[tool.uv.sources]` の調整が必要です。

---

## 4. 実行手順

### Phase 1: データ準備と CV 分割

```bash
uv run python scripts/generate_metadata.py --config config.yaml
uv run python scripts/create_clipped_audio.py --config config.yaml
uv run python scripts/prepare_cv_splits.py --config config.yaml
```

`cv_splits/splits.json` と `cv_splits/split_summary.csv` が生成されます(既にリポジトリに含まれる分割を使う場合はこの手順は不要です)。

### Phase 2: モデル学習

```bash
uv run python scripts/train_model.py --config config.yaml
```

- 既定の `config.yaml` は27条件 × 30分割 = 810モデルを学習する設定です
- `results/status.json` によって中断後の再開が可能です
- `execution.delete_images_after_training: true` の場合、1条件の学習が終わるごとにメルスペクトログラム画像を削除します(ディスク容量の節約)

---

## 5. 複数回に分けて実行した場合の結果統合

条件数が多く1回で学習しきれない場合、条件のサブセットごとに `config.yaml` の `mel_conditions` を変えて複数回に分けて実行し、後から結果を統合できます。

```bash
uv run python - <<'PY'
from pathlib import Path
from utils.analysis import combine_metrics_csvs

combined = combine_metrics_csvs(
    [
        Path('results_run1/metrics/all_metrics.csv'),
        Path('results_run2/metrics/all_metrics.csv'),
    ],
    output_path='results/metrics/all_metrics_combined.csv',
)
print(f'rows={len(combined)} conditions={combined["condition_name"].nunique()}')
PY
```

`combine_metrics_csvs` は `condition_name + repeat_id + fold_id` をキーに重複を排除し、**後に渡したファイルの値を優先**します(`drop_duplicates(keep="last")`)。

> **注意(実運用で起きた事故の教訓)**: 学習条件(特にデータ拡張倍率などの `config.yaml` 設定)を変えて再学習した際、統合スクリプトに渡すファイルの組み合わせを誤ると、意図した新しい結果ではなく古い結果が生き残ります。統合後は必ず、`condition_name / repeat_id / fold_id` をキーに元の各ファイルと突き合わせて、期待した実行結果が反映されているかを検証してください(単純に行数・条件数が一致するだけでは、どちらのバッチの値が採用されたかは分かりません)。また、統合対象の各バッチが同一の `config.yaml`(特に `augmentation` 設定)で学習されたことを `results_run*/config_snapshot.yaml` 同士を diff して確認することを推奨します。

その後、統合済みファイルに対して解析・可視化を実行します。

```bash
uv run python scripts/analyze_statistics.py --input results/metrics/all_metrics_combined.csv --output_dir results/statistics
uv run python scripts/analyze_interactions.py --input results/metrics/all_metrics_combined.csv --output_dir results/interaction_analysis
```

`visualize_metrics*.py` は入力パスをスクリプト内でハードコードしています(既定: `results/metrics/all_metrics_27conditions_aug100.csv`)。統合ファイル名を変えた場合は、各スクリプト冒頭の `pd.read_csv(...)` を書き換えてください。

---

## 6. 統計解析スクリプト

### scripts/analyze_statistics.py

- `n_fft`, `n_mels`, `fmax` の3要因に加え、交差検証分割(`repeat_id × fold_id`、既定30水準)を**ブロック因子**としてモデルに含めた分散分析(Type II ANOVA)を実行します。同一の CV 分割が全条件で共有されているため、これをブロック因子として明示的に扱わないと疑似反復により検定が楽観的になります(`--no_block` で従来型の単純ANOVAに切替可能ですが非推奨)。
- 既定では全ての2要因交互作用(`n_fft:n_mels`, `n_fft:fmax`, `n_mels:fmax`)も含めた完全な因子モデルを実行します(`--no_interactions` で主効果のみに変更可能)。
- 有意な主効果について、ブロック効果を除去した調整後AUC(`auc_recording_block_adj`)に基づき Tukey HSD による多重比較を行います。
- 結果は `--output_dir`(既定 `results/statistics`)に `anova_table.csv`, `condition_means.csv`, `tukey_{factor}.csv`, `summary.txt` として保存されます。

### scripts/analyze_interactions.py

- 交差検証分割をランダム切片とする線形混合効果モデル(固定効果: `n_fft`, `n_mels`, `fmax`, `n_fft:n_mels`, `n_mels:fmax`)による感度分析を行います。
- `n_fft×n_mels`, `n_fft×fmax`, `n_mels×fmax` それぞれの平均AUCヒートマップを出力します。

### visualize_metrics.py / _fixed.py / _fixed2.py

3スクリプトとも同じ入力(条件別AUC)から異なる形式の図表を生成します(機能が重複しているため、将来的に統合予定)。

- `visualize_metrics.py`: 条件別AUC・因子別主効果・箱ひげ図・交互作用プロットの4パネル図、および Top10/Bottom10 テーブル
- `visualize_metrics_fixed.py`: Top10テーブルと全条件ランキング図
- `visualize_metrics_fixed2.py`: 全条件ランキング図(`_fixed.py` とほぼ同内容の別実装)

### analyze_metrics.py / check_dataset.py

- `analyze_metrics.py`: `results/metrics/all_metrics.csv` の簡易サマリー出力(デバッグ用)
- `check_dataset.py`: `metadata/recording_metadata.csv` のクラス分布確認(デバッグ用)

---

## 7. テスト

```bash
uv run python -m pytest tests/ -v
```

---

## 8. 設定概要(`config.yaml`)

| セクション | 項目 | 説明 |
|---|---|---|
| `mel_conditions` | `n_fft`, `n_mels`, `fmax` | 比較する27条件の水準 |
| `mel_conditions` | `hop_length_ratio` | ホップ長 = `n_fft × hop_length_ratio`(固定 0.25) |
| `augmentation` | `noise_n_aug` | ノイズ拡張倍率(訓練データのみ) |
| `cv` | `n_folds`, `n_repeats` | 外側CVの分割数・反復回数(既定 3×10=30分割) |
| `cv` | `validation_split_in_train` | 訓練foldの中でモデル選択用に切り出す検証集合の割合 |
| `training` | `learning_rate`, `max_epochs`, `mixed_precision` 等 | 学習ハイパーパラメータ(全条件で固定) |
| `evaluation` | `threshold_method`, `aggregation` | 閾値決定方法・クリップ→音源集約方法 |
| `execution` | `delete_images_after_training` | 画像削除の有無(ハイブリッドストレージ運用) |

> `mel_conditions.htk`, `norm`, `power`, `amplitude_to_db_top_db` はドキュメント目的の記載であり、[utils/sound2image.py](utils/sound2image.py) の実装は librosa のデフォルト値(Slaney正規化・非HTK・power=2・top_db=80dB)に依存しています。値を変更しても実際の計算には反映されないため注意してください。

---

## 9. 出力ファイル

```
results/
├── status.json              # 進捗管理(git管理外)
├── config_snapshot.yaml     # 実行時のconfig.yamlのコピー(git管理外、複数バッチを統合する際の確認用)
├── run.log                  # 実行ログ(git管理外)
├── predictions/{condition_name}.csv   # クリップ単位の予測(git管理外、患者ラベル付き)
├── metrics/all_metrics.csv            # 条件×分割ごとの評価指標(1行 = 1 condition × 1 repeat × 1 fold)
├── loss_curves/{condition_name}/repeat{R}_fold{F}.csv   # (git管理外)
├── statistics/               # analyze_statistics.py の出力
└── interaction_analysis/     # analyze_interactions.py の出力
```

### `metrics/all_metrics.csv` の主なカラム

- `condition_name`, `n_fft`, `n_mels`, `fmax`, `repeat_id`, `fold_id`
- `auc_recording`, `auc_recording_ci_lower/upper`(DeLong近似CI)
- `f1_recording`, `sensitivity_recording`, `specificity_recording`, `ppv_recording`, `npv_recording`, `accuracy_recording`
- `auc_clip` 等のクリップレベル指標
- `best_val_auc`, `best_epoch`, `epochs_run`

主指標は `auc_recording`(クリップレベル予測を音源単位に平均集約したAUC、閾値非依存)です。`*_recording` の二値指標(F1・感度・特異度等)は、各テスト分割のデータ自体からYouden indexで事後決定した閾値に基づく参考値である点に注意してください。

---

## 10. 再現性に関する注意点

- `cv_splits/splits.json` は同じ CV 分割を再現するために重要です。乱数シードは `cv.random_state_base`(既定42)を起点に、`repeat_id` ごとに +1、検証分割はテスト分割のシード + 1000 で決定論的に導出されます。
- `results/status.json` を削除すると全条件が再実行されます。
- 学習時にモデル重みは既定で保存されません(`save_models.save_all: false`)。事後的な閾値再計算や推論のやり直しはできないため、必要な場合は学習前に設定を変更してください。
