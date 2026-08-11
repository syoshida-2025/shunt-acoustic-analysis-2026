# external_validation/

内部検証パイプライン(リポジトリ直下の `config.yaml` / `scripts/`、27条件 x 反復CV、
ハイパーパラメータ固定)とは別の目的を持つ、独立したパイプラインです。

**内部検証**: 27条件のメルスペクトログラム設定を比較するため、学習ハイパーパラメータを
全条件で固定し、患者レベルの反復交差検証(10repeat x 3fold = 30分割)で評価する。

**外部検証(本フォルダ)**: 内部検証で選ばれた単一条件について、今度はハイパー
パラメータをきちんと調整したモデルを開発データ(内部検証と同一データセット)で
構築し、未知の外部hold-outデータに1回だけ適用して汎化性能を確認する。

両者は目的(パラメータ比較 vs 汎化性能の確認)も統計設計(反復CV vs 開発/外部の
1回きりの分離)も異なるため、意図的にコード・configを分離しています。
`utils/` 以下のデータ処理・学習・評価ロジックは共通で再利用しています。

---

## 設計上の要点

### 1. ハイパーパラメータ探索

- 探索対象: `learning_rate` x `weight_decay` のgrid search(既定 5 x 3 = 15通り)
- 評価方法: 患者単位の3-fold CV(**反復なし**、1回のみ)。各HP候補について3-fold
  それぞれで学習し、平均val_aucで比較する
- アーキテクチャ(ResNet200d, EfficientNetV2-S)ごとに**独立して**探索する(HPを共有しない)
- `batch_size` は探索対象外(32に固定、内部検証と同一)
- データ拡張(ノイズ100倍・輝度ジッタ)は探索対象外、内部検証と同一設定に固定し、
  訓練foldにのみ適用する(検証fold・外部データには適用しない)

3-foldを採用した理由: 単一の train/val 分割(1回きり)では、検証データが少ない
（本データセット規模では約12〜15例）ため、AUC推定のばらつきが大きく、HP比較が
ノイズに左右されるリスクがある。3-foldで全開発データを検証に使い切ることで、
比較の安定性を確保している。

### 2. 最終モデル

- 選ばれたHPの3-fold分の `best_epoch` の中央値を、最終モデルのエポック数として固定する
- 開発データ**全例**(検証データを分離しない)で、そのエポック数だけ再学習する
  (早期停止は行わない)
- 3-foldのモデルをアンサンブルする方式は採用していない(単一モデルの方がシンプルで
  説明しやすく、全データを学習に使い切れるため)

### 3. 閾値のロック(内部検証の既知の課題への対応)

内部検証パイプラインは、Youden閾値をテスト分割自身のデータから事後決定しており、
これはAUC(主指標)には影響しないが、二次指標(F1・感度・特異度等)を楽観的にする
可能性がある。

外部検証では、この閾値を**開発データ側だけ**で決定してロックし、外部データには
一切変更せずそのまま適用する。具体的には、選ばれたHPで学習した3-foldモデルそれぞれの
「学習に使わなかった1foldに対する予測」をプールしたもの(全開発患者について、
学習に使われていないモデルからの予測が1つずつ揃うout-of-fold予測)から
Youden閾値を決定する(`utils/external_metrics.py:lock_thresholds_from_oof_predictions`)。

外部データ側では、この閾値をそのまま適用するだけで、外部データから閾値を
再計算することは絶対に行わない(`utils/external_metrics.py:evaluate_with_locked_threshold`)。

### 4. モデル重みの保存

内部検証は再学習が前提のため重みを保存しないが、外部検証は「1つのロックされた
モデルを外部データに適用する」ことが目的のため、最終モデルの重みを保存する
(`results/{architecture}/{condition_name}/final_model.pth`)。

### 5. Phase Bの評価範囲(意図的に簡略化している点)

対象条件をsingle(1条件)に絞っているため、外部データ上で複数条件を比較する
ような複雑な統計検定(条件間のpaired bootstrap比較等)は行っていない。
recording-level AUC(95%CI, DeLong法)と、ロック済み閾値による二次指標を
記述的に報告するだけの、意図的にシンプルな設計にしている。

---

## 実行手順

### 前提: 外部データの準備

まず、以下の場所にファイルを配置する(`external_validation/` を基準としたパス):

```
external_validation/
└── data_external/
    ├── external_data_v3.xlsx    # メタデータ(トップレベル README.md の「データスキーマについて」と同一スキーマ)
    └── all_sounds/               # WAVファイル一式
        ├── ....wav
        └── ...
```

`all_clips/`・`metadata_external/` は次のコマンドで自動生成されるため、事前に用意する必要はない。

`scripts/generate_metadata.py` と `scripts/create_clipped_audio.py` は `config["paths"]` という
構造を前提にしており、`config_external.yaml`(dev/external に分かれた別構造)とはそのまま
互換しない。そのため、外部データのメタデータ生成専用に `config_external_metadata.yaml` を
用意している(値は `config_external.yaml` の `external` セクションと揃えてあるので、パスを
変更する場合は両方を更新すること)。

```bash
cd external_validation
uv run python ../scripts/generate_metadata.py --config config_external_metadata.yaml
uv run python ../scripts/create_clipped_audio.py --config config_external_metadata.yaml
```

実行後、`external_validation/metadata_external/clip_metadata.csv` と
`external_validation/data_external/all_clips/` が生成され、`config_external.yaml` の
`external` セクションが期待するパスと一致する。

### Phase A: 開発データでのHP探索・最終モデル構築

```bash
cd external_validation
uv run python tune_and_develop.py --config config_external.yaml
# 特定のアーキテクチャだけ実行する場合:
uv run python tune_and_develop.py --config config_external.yaml --architecture resnet200d
```

出力(`results/{architecture}/{condition_name}/`):
- `grid_search_results.csv` — 全HP候補 x 全foldの結果(論文の表・ヒートマップ用)
- `selected_hp.json` — 採用したHP、平均val_auc、最終エポック数
- `oof_predictions_winning_hp.csv` — 採用HPのout-of-fold予測(閾値ロックの根拠)
- `locked_threshold.json` — ロック済みのクリップ/音源レベル閾値
- `final_model.pth` — 開発データ全例で再学習した最終モデルの重み

### Phase B: 外部データでの評価(1回きり)

```bash
uv run python evaluate_external.py --config config_external.yaml
```

出力(`results/{architecture}/{condition_name}/external/`):
- `external_predictions.csv` — クリップ単位の予測確率(ROC曲線等は事後的にこれから作図可能)
- `external_metrics.json` — recording-level AUC(+95%CI)、ロック済み閾値によるF1・感度・特異度等
- `external_vs_dev_summary.txt` — 開発フェーズ(3-fold探索)と外部データの性能比較サマリー

実行前に、開発データと外部データの患者IDに重複がないことを自動チェックする
(重複があれば例外を送出して停止する)。

---

## 内部検証との関係

- 前処理(音声読み込み・メルスペクトログラム変換・画像正規化)のロジックは
  `utils/sound2image.py` 等を完全に共有しており、内部検証と外部検証で異なる処理は
  行っていない
- `config_external.yaml` の `selected_condition` は、内部検証のANOVA/Tukey結果に基づき
  選定すること(既定値は内部検証で最良だった条件)
- ここで得られる外部データのAUCは、内部検証の反復CV AUC(30分割平均)とは異なる
  推定である(開発データ全体で学習した単一モデルを、1回だけ独立データに適用した結果)。
  一般に外部データでは内部CVより性能が低下する("generalization gap")ことが多いが、
  それ自体が汎化性能の重要な情報である。
