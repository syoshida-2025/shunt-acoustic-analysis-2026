"""utils/data_split.py

患者ID単位（クリップ単位ではない）でのデータ分割ロジック。
StratifiedGroupKFold により、陽性/陰性比を保ちつつ同一患者のデータが
train/test 両方に分散しないように分割する。
scripts/prepare_cv_splits.py（外側CV）と scripts/train_model.py（内側validation分割）
の両方から呼び出される。
"""

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def print_ratio(
    df: pd.DataFrame,
    label_name: str,
    threshold: float
) -> None:
    """
    データフレーム内の陽性/陰性データの比率を表示

    Parameters:
    -----------
    df : pd.DataFrame
        対象のデータフレーム
    label_name : str
        ラベル列名
    threshold : float
        陽性/陰性の閾値
    """
    positive_data = df[df[label_name] >= threshold]
    negative_data = df[df[label_name] < threshold]
    print(
        f"total: {len(df)} "
        f"positive: {len(positive_data)} "
        f"negative: {len(negative_data)} "
        f"p/n ratio: {(df[label_name] >= threshold).mean():.2f}"
    )


def _split_train_test_generator(
    df: pd.DataFrame,
    patient_id_col: str,
    label_name: str,
    threshold: float,
    test_size: float = 0.2,
    random_state: int = 42,
    use_yield_mode: bool = False
):
    """
    内部ジェネレータ関数
    """
    # 入力検証
    if not 0 < test_size < 1:
        raise ValueError(f"test_sizeは0から1の間である必要があります（現在: {test_size}）")

    if patient_id_col not in df.columns:
        raise ValueError(f"列 '{patient_id_col}' がデータフレームに存在しません")

    if label_name not in df.columns:
        raise ValueError(f"列 '{label_name}' がデータフレームに存在しません")

    if len(df) == 0:
        raise ValueError("データフレームが空です")

    n_patients = df[patient_id_col].nunique()
    if n_patients < 2:
        raise ValueError(f"患者数が不十分です（{n_patients}人）。最低2人必要です")

    # 統計情報の表示（初回のみ）
    print("全データ:")
    print(f"総データ数: {len(df)}, 患者数: {n_patients}")
    print_ratio(df=df, label_name=label_name, threshold=threshold)
    print()

    # 層別化のためのラベルを作成（各サンプルが陽性か陰性か）
    stratify_labels = (df[label_name] >= threshold).astype(int)

    # StratifiedGroupKFoldを使用して患者単位で層別化分割
    # test_size > 0.5 の場合は訓練とテストを入れ替える
    if test_size > 0.5:
        # テストの方が大きい場合: 訓練用の比率で分割し、結果を入れ替える
        target_size = 1 - test_size
        swap_train_test = True
    else:
        # 通常の場合: テスト用の比率で分割
        target_size = test_size
        swap_train_test = False

    # n_splitsを計算（最小値は2）
    n_splits = max(2, int(1 / target_size))

    # 各分割を処理
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(df, y=stratify_labels, groups=df[patient_id_col]), start=1):
        # test_size > 0.5 の場合は訓練とテストを入れ替え
        if swap_train_test:
            train_idx, test_idx = test_idx, train_idx

        # データフレームを分割
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        # 各分割の統計情報
        if use_yield_mode:
            print(f"=== Fold {fold_idx}/{n_splits} ===")
        
        # 訓練データの統計情報
        print("訓練データ:")
        print(f"訓練データ数: {len(train_df)}, 患者数: {train_df[patient_id_col].nunique()}")
        print_ratio(df=train_df, label_name=label_name, threshold=threshold)
        print()

        # テストデータの統計情報
        print("テストデータ:")
        print(f"テストデータ数: {len(test_df)}, 患者数: {test_df[patient_id_col].nunique()}")
        print_ratio(df=test_df, label_name=label_name, threshold=threshold)
        print()

        # 患者の重複チェック
        train_patients = set(train_df[patient_id_col].unique())
        test_patients = set(test_df[patient_id_col].unique())
        overlap = train_patients & test_patients

        if overlap:
            print(f"⚠ 警告: 以下の患者が訓練・テストの両方に存在します: {overlap}")
        else:
            print(f"✓ 患者の重複なし（訓練とテストで完全に分離）")
        print()

        yield train_df, test_df

def split_train_test_by_patient(
    df: pd.DataFrame,
    patient_id_col: str,
    label_name: str,
    threshold: float,
    test_size: float = 0.2,
    random_state: int = 42,
    use_yield: bool = False
):
    """
    患者IDごとにデータを訓練セットとテストセットに層別化分割

    同一患者のデータが訓練とテストに分散しないようにグループ化し、
    かつ陽性/陰性の比率を可能な限り保つように層別化します。

    Parameters:
    -----------
    df : pd.DataFrame
        分割するデータフレーム
    patient_id_col : str
        患者IDの列名
    label_name : str
        ラベル列名
    threshold : float
        陽性/陰性の閾値
    test_size : float
        テストセットの割合（デフォルト: 0.2）
        0.5より大きい値も指定可能（テストセットの方が大きくなる）
    random_state : int
        乱数シード（デフォルト: 42）
    use_yield : bool
        Trueの場合、全ての分割をyieldで返すジェネレータとして動作
        Falseの場合、最初の分割のみをtupleで返す（デフォルト: False）

    Returns (use_yield=False):
    ---------------------------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)

    Yields (use_yield=True):
    ------------------------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)

    Raises:
    -------
    ValueError
        必要な列が存在しない、データが空、患者数が不十分、
        またはtest_sizeが範囲外の場合

    Notes:
    ------
    test_sizeで指定された割合に最も近くなるようにn_splitsを計算するため、
    実際のテストデータの割合は厳密に一致せず、約 1/n_splits となります。
    (例: test_size=0.3 の場合、n_splits=3となり、テストデータの割合は約33.3%になります)

    use_yield=Falseの場合は最初の分割のみを返します。
    use_yield=Trueの場合はn_splits個の異なる分割を順番にyieldします。

    Examples:
    ---------
    # 単一の分割を取得（デフォルト）
    >>> df = pd.read_excel('data.xlsx')
    >>> train_df, test_df = split_train_test_by_patient(
    ...     df=df,
    ...     patient_id_col="Pt No",
    ...     label_name="FV",
    ...     threshold=400,
    ...     test_size=0.2
    ... )

    # 全ての分割を取得（ジェネレータとして使用）
    >>> for fold_idx, (train_df, test_df) in enumerate(split_train_test_by_patient(
    ...     df=df,
    ...     patient_id_col="Pt No",
    ...     label_name="FV",
    ...     threshold=400,
    ...     test_size=0.2,
    ...     use_yield=True
    ... ), start=1):
    ...     print(f"Fold {fold_idx}")
    """
    gen = _split_train_test_generator(
        df=df,
        patient_id_col=patient_id_col,
        label_name=label_name,
        threshold=threshold,
        test_size=test_size,
        random_state=random_state,
        use_yield_mode=use_yield
    )

    if use_yield:
        return gen
    else:
        # 最初の分割のみを返して終了
        try:
            return next(gen)
        except StopIteration:
            # ここに来ることは通常ないはずだが念のため
            raise ValueError("データの分割に失敗しました")
