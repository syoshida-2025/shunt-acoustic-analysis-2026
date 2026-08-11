"""check_dataset.py

metadata/*.csv と cv_splits/splits.json を突き合わせて、クラス分布・患者数・
CV分割のサマリーをコンソールに出力するデバッグ用スクリプト。
"""

import pandas as pd
import json

# メタデータ確認
metadata = pd.read_csv('metadata/recording_metadata.csv')
print("="*80)
print("DATASET SUMMARY")
print("="*80)
print(f"\nRecordings (音源):")
print(f"  Total: {len(metadata)}")
print(f"  Positive (label=1): {(metadata['label']==1).sum()}")
print(f"  Negative (label=0): {(metadata['label']==0).sum()}")
print(f"  Ratio: {(metadata['label']==1).sum()}/{len(metadata)} = {(metadata['label']==1).sum()/len(metadata):.2%}")

# クリップメタデータ確認
clip_metadata = pd.read_csv('metadata/clip_metadata.csv')
print(f"\nClips (5秒クリップ):")
print(f"  Total: {len(clip_metadata)}")
print(f"  Positive: {(clip_metadata['label']==1).sum()}")
print(f"  Negative: {(clip_metadata['label']==0).sum()}")

# 患者数確認
print(f"\nPatients (患者):")
print(f"  Total unique patients: {metadata['patient_id'].nunique()}")
patient_dist = metadata.groupby('label')['patient_id'].nunique()
print(f"  Positive patients: {patient_dist.get(1, 0)}")
print(f"  Negative patients: {patient_dist.get(0, 0)}")

# クロスバリデーション分割確認
# splits.json は {"metadata": {...}, "splits": [{repeat_id, fold_id, train_patients,
# test_patients, ...}, ...]} というフラットな構造（prepare_cv_splits.py 参照）。
# 訓練fold内の検証分割（val）は splits.json には保存されておらず、
# train_model.py が学習実行時に動的に計算する。
with open('cv_splits/splits.json', 'r') as f:
    splits_data = json.load(f)
splits = splits_data['splits']
n_repeats = splits_data['metadata']['n_repeats']
n_folds = splits_data['metadata']['n_folds']
print(f"\nCross-Validation:")
print(f"  Repeats: {n_repeats}")
print(f"  Folds per repeat: {n_folds}")
print(f"  Total splits: {len(splits)}")

# 最初の分割を確認（repeat_id=1, fold_id=1）
first_split = splits[0]
print(f"\nFirst split (repeat {first_split['repeat_id']}, fold {first_split['fold_id']}):")
print(f"  Train patients: {first_split['n_train_patients']}")
print(f"  Test patients: {first_split['n_test_patients']}")

