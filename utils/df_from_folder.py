"""utils/df_from_folder.py

positive/negative フォルダ構成から DataFrame を作る補助関数。
現在のパイプライン（clip_metadata.csv ベース）からは呼ばれておらず、
本リポジトリのスクリプト群では未使用。過去のフォルダ構成を前提にした
データ確認用のレガシーユーティリティとして残っている。
"""

from pathlib import Path
import pandas as pd
from pprint import pprint

def create_df(database: str | Path, folder: str, expected_labels: dict[str, int] | None = None) -> pd.DataFrame:
    """
    フォルダ構成からpandas DataFrameを作成する関数

    Args:
        database (str | Path): 基本パス（各フォルダが含まれるディレクトリ）
        folder (str): train, val, or test
        expected_labels (dict[str, int] | None): クラス名とラベルのマッピング（オプション）

    Returns:
        pd.DataFrame: image列とlabel列を持つDataFrame
    """

    dir_path = Path(database, folder)

    label_file = Path(database, "labels.txt")

    if expected_labels is not None:
        print(f"\n指定されたラベルマッピングを使用します")
        folder_labels = expected_labels
    elif label_file.exists():
        print(f"\n{label_file.as_posix()}をもとにファルダ名とラベルのマッピングを行います")

        folder_labels = {}

        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(":", 1)
                    folder_labels[key.strip()] = int(value.strip())
    else:
        print("\n新規にファルダ名とラベルのマッピングを行います")

        from torchvision.datasets import ImageFolder
        # Check if dir_path exists, otherwise ImageFolder might fail,
        # but create_df is expected to fail if path doesn't exist?
        # Assuming dir_path exists if we reach here, or ImageFolder will raise error.
        temp_dataset = ImageFolder(dir_path)

        # フォルダ名とラベルのマッピング
        folder_labels = {name: idx for idx, name in enumerate(temp_dataset.classes)}

        with open(label_file, "w", encoding="utf-8") as f:
            for key, value in folder_labels.items():
                f.write(f"{key}: {value}\n")
        print(f"結果は{label_file.as_posix()}に保存しました")

    print()
    pprint(folder_labels, width=1, sort_dicts=False)

    # データを格納するリスト
    data = []

    # 各フォルダを処理
    for folder_name, label in folder_labels.items():
        folder_path = dir_path / folder_name

        if not folder_path.exists():
             continue

        image_files = folder_path.iterdir()

        for image_path in image_files:
            # フルパスを取得
            full_path = image_path.resolve()

            data.append({
                'image': full_path.as_posix(),
                'label': label
            })

    # DataFrameを作成
    df = pd.DataFrame(data)

    # 結果を表示
    print(f"\n{dir_path.as_posix()}から作成されたDataFrame")
    print(df)

    # ラベルの分布を確認
    if not df.empty:
        print("\nラベルの分布:")
        print(df['label'].value_counts().sort_index())
    else:
        print("\nデータが見つかりませんでした。")

    return df
