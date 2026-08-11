"""utils/augmentation.py

音声波形レベルのデータ拡張（訓練データのみに適用）。
無音挿入（insert_random_silence）とSNR指定のノイズ付加（add_noise_by_snr）を提供する。
utils/sound2image.py の process_audio_to_images から呼び出される。
"""

import numpy as np


def insert_random_silence(
    wave_data: np.ndarray,
    sr: int,
    max_silence_duration: float,
    random_seed: int | None = None
) -> np.ndarray:
    """
    音声データにランダムな位置で無音を挿入

    Parameters:
    -----------
    wave_data : np.ndarray
        音声データ
    sr : int
        サンプリングレート
    max_silence_duration : float
        挿入する無音の最大長（秒）
    random_seed : int | None
        乱数シード（再現性のため）

    Returns:
    --------
    np.ndarray
        無音が挿入された音声データ
    """
    # 乱数ジェネレータを初期化
    rng = np.random.default_rng(random_seed)

    # 無音の長さをランダムに決定
    silence_duration = rng.uniform(0, max_silence_duration)

    # 無音の長さをサンプル数に変換
    silence_samples = int(silence_duration * sr)

    # 無音を挿入できる範囲を計算（元のデータ長を超えないように）
    max_start_pos = max(0, len(wave_data) - silence_samples)

    if max_start_pos <= 0:
        # 無音が元データより長い場合は全て無音にする
        return np.zeros_like(wave_data)

    # ランダムな位置を選択
    start_pos = rng.integers(0, max_start_pos, endpoint=True)
    end_pos = start_pos + silence_samples

    # データをコピーして無音を挿入
    augmented_data = wave_data.copy()
    augmented_data[start_pos:end_pos] = 0.0

    return augmented_data

def add_noise_by_snr(
    wave_data: np.ndarray,
    min_snr_db: float = 10.0,
    max_snr_db: float = 30.0,
    strength: float = 1.0,
    random_seed: int | None = None,
    snr_db: float | None = None,
) -> np.ndarray:
    """
    指定されたSNR（信号対雑音比）に基づいてノイズを追加する

    Parameters:
    -----------
    wave_data : np.ndarray
        元の音声データ
    min_snr_db : float
        目標とする最小SNR（デシベル）。数値が低いほどノイズが強くなる。
    max_snr_db : float
        目標とする最大SNR（デシベル）。
    snr_db : float | None
        固定のSNR値を指定する場合はこちらを使用します。
    strength : float
        ノイズの強度（デフォルト: 1.0）
    random_seed : int | None
        乱数シード

    Returns:
    --------
    np.ndarray
        ノイズが追加された音声データ
    """
    if snr_db is not None:
        min_snr_db = snr_db
        max_snr_db = snr_db

    # 乱数ジェネレータを初期化
    rng = np.random.default_rng(random_seed)
    
    # 信号のパワー（平均二乗）を計算
    signal_power = np.mean(wave_data**2)
    if signal_power == 0:
        return wave_data # 無音の場合は何もせず返す

    # 指定された範囲から目標SNRをランダムに選択
    target_snr_db = rng.uniform(min_snr_db, max_snr_db)
    
    # SNRをデシベルからリニアスケールに変換
    snr_linear = 10 ** (target_snr_db / 10.0)
    
    # 必要なノイズのパワーを計算
    noise_power = signal_power / snr_linear
    
    # ガウシアンホワイトノイズを生成
    # ノイズの標準偏差はパワーの平方根
    noise = rng.normal(0, np.sqrt(noise_power), len(wave_data))

    # 元のデータにノイズを追加（強度を適用）
    noisy_data = wave_data + noise * strength

    return noisy_data.astype(wave_data.dtype, copy=False)
