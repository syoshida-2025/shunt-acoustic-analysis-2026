"""utils/sound2image.py

音声波形をメルスペクトログラム画像（PNG化前のuint8配列）に変換する中核ロジック。
wave2image が STFT・メル変換・対数振幅変換・画像正規化を行い、
process_audio_to_images がデータ拡張と組み合わせてフォルダへの一括書き出しを行う。
scripts/generate_melspectrogram.py から呼び出される。
"""

import librosa
import numpy as np
import cv2
import warnings
from matplotlib import colormaps
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pandas as pd
from .augmentation import insert_random_silence, add_noise_by_snr

warnings.filterwarnings('ignore', category=UserWarning, module='librosa')


def get_adaptive_snr_range(wave_data: np.ndarray) -> tuple[float, float]:
    """
    音声データの音量レベルに基づいて適応的なSNR範囲を返す

    小さい音にはより強いノイズ（低いSNR）を、
    大きい音には弱めのノイズ（高いSNR）を適用することで、
    すべての音に適切なデータ拡張を行う。

    Parameters:
    -----------
    wave_data : np.ndarray
        音声波形データ

    Returns:
    --------
    tuple[float, float]
        (min_snr_db, max_snr_db) の範囲
    """
    rms = np.sqrt(np.mean(wave_data**2))
    signal_db = 20 * np.log10(rms + 1e-10)

    if signal_db < -45:
        return (15.0, 30.0)
    elif signal_db < -35:
        return (20.0, 35.0)
    else:
        return (25.0, 45.0)


def clip_audio(
    file_path: str,
    start_time: float,
    duration: float,
    target_sr: int | None = None
) -> tuple[np.ndarray, int]:
    """
    WAVファイルから指定時間範囲の音声を切り出す

    Parameters:
    -----------
    file_path : str
        入力WAVファイルのパス
    start_time : float
        開始時間（秒）
    duration : float
        切り取る音声の長さ（秒）
    target_sr : int | None
        目標サンプリングレート（Noneの場合は元のレートを維持）

    Returns:
    --------
    tuple[np.ndarray, int]
        (音声波形データ, サンプリングレート)
    """
    if duration <= 0:
        raise ValueError("durationは0より大きくする必要があります。")
    try:
        y, sr = librosa.load(
            file_path,
            sr=target_sr,
            offset=start_time,
            duration=duration
        )
    except FileNotFoundError:
        raise IOError(f"ファイル '{file_path}' が見つかりません。")
    except Exception as e:
        raise IOError(f"ファイル '{file_path}' の読み込み中にエラーが発生しました。") from e

    target_length = int(duration * sr)
    actual_length = len(y)
    if actual_length < target_length:
        padding_size = target_length - actual_length
        y = np.pad(y, (0, padding_size), mode='constant')
    return y, sr


def _spectrogram_to_image(
    spec_db: np.ndarray,
    target_shape: tuple[int, int],
    cmap: str | None
) -> np.ndarray:
    """
    デシベルスケールのスペクトログラムを正規化・リサイズして画像配列に変換する

    Parameters:
    -----------
    spec_db : np.ndarray
        デシベルスケールのスペクトログラム
    target_shape : tuple[int, int]
        出力画像のサイズ (高さ, 幅)
    cmap : str | None
        カラーマップ名（Noneの場合はグレースケール）

    Returns:
    --------
    np.ndarray
        uint8型の3チャンネル画像配列 (H, W, 3)
    """
    db_min, db_max = np.min(spec_db), np.max(spec_db)
    if db_max == db_min:
        normalized_db = np.zeros_like(spec_db, dtype=np.float32)
    else:
        normalized_db = (spec_db - db_min) / (db_max - db_min)

    resized_db = cv2.resize(normalized_db, (target_shape[1], target_shape[0]))

    if cmap is None:
        gray_img = (resized_db * 255).astype(np.uint8)
        return np.stack([gray_img] * 3, axis=-1)
    else:
        colormap = colormaps.get_cmap(cmap)
        colored_img = colormap(resized_db)
        return (colored_img[:, :, :3] * 255).astype(np.uint8)


def wave2image(
    wave_data: np.ndarray,
    sr: int,
    n_mels: int = 128,
    # レートベースモード（既存、後方互換性保持）
    n_fft_rate: float = 0.025,
    hop_length_rate: float = 0.01,
    min_n_fft: int = 512,
    # 直接モード（NEW: 両方 not None の場合に優先される）
    n_fft_direct: int | None = None,
    hop_length_direct: int | None = None,
    # その他のパラメータ（変更なし）
    fmax: int | None = None,
    target_shape: tuple[int, int] = (128, 128),
    cmap: str | None = None
) -> np.ndarray:
    """
    音声波形データをメルスペクトログラム画像に変換する

    モード選択:
    - n_fft_direct と hop_length_direct が両方 not None の場合:
        直接モードを使用（本研究で使用）。
    - それ以外: レートベースモード（レガシー）を使用。

    Parameters:
    -----------
    wave_data : np.ndarray
        音声波形データ
    sr : int
        サンプリングレート
    n_mels : int
        メルフィルタバンクの数
    n_fft_rate : float
        ウィンドウ長レート（レートベースモード用）
    hop_length_rate : float
        ホップ長レート（レートベースモード用）
    min_n_fft : int
        FFTサイズの最小値（レートベースモード用）
    n_fft_direct : int | None
        直接指定するFFTサイズ（直接モード用）
    hop_length_direct : int | None
        直接指定するホップ長（直接モード用）
    fmax : int | None
        最大周波数 (Hz)
    target_shape : tuple[int, int]
        出力画像サイズ (高さ, 幅)
    cmap : str | None
        カラーマップ名

    Returns:
    --------
    np.ndarray
        uint8型の3チャンネル画像配列 (H, W, 3)
    """
    # 直接モードの引数検証
    if (n_fft_direct is None) != (hop_length_direct is None):
        raise ValueError(
            "n_fft_direct と hop_length_direct は両方指定するか、両方 None にしてください。"
        )

    if n_fft_direct is not None:
        if n_fft_direct <= 0 or hop_length_direct <= 0:
            raise ValueError(
                f"n_fft_direct と hop_length_direct は正の整数である必要があります "
                f"(got {n_fft_direct}, {hop_length_direct})."
            )
        if hop_length_direct > n_fft_direct:
            warnings.warn(
                f"hop_length_direct ({hop_length_direct}) > n_fft_direct ({n_fft_direct}); "
                f"これは珍しい設定で情報損失が発生する可能性があります。",
                UserWarning,
            )

    # ナイキスト周波数チェック（モードに依存しない）
    nyquist = sr / 2.0
    if fmax is not None and fmax > nyquist:
        raise ValueError(
            f"fmax={fmax} はナイキスト周波数 ({nyquist} Hz, sr={sr}) を超えています。"
        )

    # n_fft と hop_length の決定
    if n_fft_direct is not None and hop_length_direct is not None:
        # 直接モード
        n_fft = n_fft_direct
        hop_length = hop_length_direct
        win_length = n_fft  # 本研究ではウィンドウ長 = n_fft
    else:
        # レートモード（レガシー）
        win_length = int(n_fft_rate * sr)
        hop_length = int(hop_length_rate * sr)
        n_fft = 1 << (win_length - 1).bit_length()
        n_fft = max(n_fft, min_n_fft)

    # fmax の有効値を決定
    if fmax is None:
        effective_fmax = min(8000, nyquist)
    else:
        effective_fmax = min(fmax, nyquist)

    stft = librosa.stft(
        wave_data,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft
    )
    mel_spectrogram = librosa.feature.melspectrogram(
        S=np.abs(stft)**2,
        sr=sr,
        n_mels=n_mels,
        fmax=effective_fmax,
        n_fft=n_fft,
        hop_length=hop_length
    )
    mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return _spectrogram_to_image(mel_db, target_shape, cmap)


def wave2image_fullspec(
    wave_data: np.ndarray,
    sr: int,
    n_fft_rate: float = 0.025,
    hop_length_rate: float = 0.01,
    fmax: int | None = None,
    min_n_fft: int = 512,
    target_shape: tuple[int, int] = (128, 128),
    cmap: str | None = None
) -> np.ndarray:
    """
    音声波形データをフルスペクトログラム画像に変換する（既存コードから変更なし）

    Parameters:
    -----------
    wave_data : np.ndarray
        音声波形データ
    sr : int
        サンプリングレート
    n_fft_rate : float
        ウィンドウ長レート
    hop_length_rate : float
        ホップ長レート
    fmax : int | None
        最大周波数 (Hz)
    min_n_fft : int
        FFTサイズの最小値
    target_shape : tuple[int, int]
        出力画像サイズ
    cmap : str | None
        カラーマップ名

    Returns:
    --------
    np.ndarray
        uint8型の3チャンネル画像配列 (H, W, 3)
    """
    win_length = int(n_fft_rate * sr)
    hop_length = int(hop_length_rate * sr)
    n_fft = 1 << (win_length - 1).bit_length()
    n_fft = max(n_fft, min_n_fft)

    stft = librosa.stft(
        wave_data,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length
    )

    power_spec = np.abs(stft)**2

    if fmax is not None:
        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        valid_bins = np.where(freq_bins <= fmax)[0]
        if len(valid_bins) == 0:
            raise ValueError(
                f"fmax={fmax}は有効な周波数範囲外です "
                f"（最小: {freq_bins[0]:.2f} Hz, 最大: {freq_bins[-1]:.2f} Hz）"
            )
        max_bin = valid_bins[-1] + 1
        power_spec = power_spec[:max_bin, :]

    db = librosa.power_to_db(power_spec, ref=np.max)

    return _spectrogram_to_image(db, target_shape, cmap)


def process_audio_to_images(
    df: pd.DataFrame,
    audio_folder: str,
    output_folder: str,
    label_name: str,
    threshold: float,
    start_time: float = 0.0,
    clip_duration: float = 3.0,
    target_sr: int | None = None,
    use_mel: bool = True,
    n_mels: int = 128,
    # レートベース（レガシー）
    n_fft_rate: float = 0.025,
    hop_length_rate: float = 0.01,
    min_n_fft: int = 512,
    # 直接モード（NEW）
    n_fft_direct: int | None = None,
    hop_length_direct: int | None = None,
    # その他のパラメータ（変更なし）
    fmax: int | None = None,
    target_shape: tuple[int, int] = (128, 128),
    cmap: str | None = None,
    n_aug: int = 1,
    apply_silence: bool = False,
    max_silence_duration: float = 0.0,
    apply_noise: bool = False,
    min_snr_db: float = 30.0,
    max_snr_db: float = 50.0,
    strength: float = 1.0,
    use_adaptive_snr: bool = False,
    enable_multiclip: bool = False,
    clip_step: float = 3.0,
    max_clip: int = 3,
    base_seed: int | None = 42
) -> None:
    """
    音声ファイルを画像に変換して positive/negative フォルダに保存する

    直接モード（NEW）: n_fft_direct と hop_length_direct が両方 not None の場合、
    wave2image に直接渡す。レートベースパラメータは無視される。

    Parameters:
    -----------
    df : pd.DataFrame
        処理対象の音声ファイル情報を含むDataFrame（audio_file カラムが必要）
    audio_folder : str
        音声ファイルが格納されているフォルダパス
    output_folder : str
        画像の出力先フォルダパス
    label_name : str
        DataFrameのラベル列名
    threshold : float
        positive/negative を分ける閾値
    start_time : float
        音声の切り出し開始時間（秒）
    clip_duration : float
        音声の切り出し時間（秒）
    target_sr : int | None
        目標サンプリングレート
    use_mel : bool
        True でメルスペクトログラム、False でフルスペクトログラム
    n_mels : int
        メルフィルタバンクの数
    n_fft_rate : float
        ウィンドウ長レート（レートベースモード用）
    hop_length_rate : float
        ホップ長レート（レートベースモード用）
    min_n_fft : int
        FFTサイズの最小値（レートベースモード用）
    n_fft_direct : int | None
        直接指定するFFTサイズ（直接モード用）
    hop_length_direct : int | None
        直接指定するホップ長（直接モード用）
    fmax : int | None
        最大周波数 (Hz)
    target_shape : tuple[int, int]
        出力画像サイズ
    cmap : str | None
        カラーマップ名
    n_aug : int
        拡張回数（拡張なしの場合は1）
    apply_silence : bool
        True で無音挿入を適用
    max_silence_duration : float
        挿入する無音の最大長さ（秒）
    apply_noise : bool
        True でノイズ追加を適用
    min_snr_db : float
        最小SNR (dB)
    max_snr_db : float
        最大SNR (dB)
    strength : float
        ノイズ強度
    use_adaptive_snr : bool
        True で音量レベルに基づく適応的SNR
    enable_multiclip : bool
        True で音声をスライドさせながら複数切り出し
    clip_step : float
        マルチクリップのステップ時間（秒）
    max_clip : int
        マルチクリップの最大クリップ数
    base_seed : int | None
        乱数シード
    """
    if use_mel:
        spectrogram_type = f"メルスペクトログラム (n_mels: {n_mels})"
    else:
        spectrogram_type = "フルスペクトログラム"
    print(f"画像生成を開始します (種類: {spectrogram_type})")

    positive_folder = Path(output_folder) / "positive"
    negative_folder = Path(output_folder) / "negative"
    positive_folder.mkdir(parents=True, exist_ok=True)
    negative_folder.mkdir(parents=True, exist_ok=True)

    actual_n_aug = 1 if (not apply_silence and not apply_noise) else n_aug

    rng = np.random.default_rng(base_seed)
    success_count = 0
    error_count = 0

    for row in tqdm(df.itertuples(), total=len(df), desc="音声ファイルを処理中"):
        try:
            audio_file = Path(audio_folder) / row.audio_file
            if not audio_file.exists():
                print(f"⚠ ファイルが見つかりません: {audio_file}")
                error_count += 1
                continue

            if clip_duration <= 0:
                raise ValueError("clip_durationは0より大きくする必要があります。")

            if enable_multiclip:
                full_wave_data, original_sr = librosa.load(str(audio_file), sr=target_sr)
                sr = original_sr
                total_duration = len(full_wave_data) / sr

                start_times = []
                current_start = start_time
                while current_start + clip_duration <= total_duration:
                    start_times.append(current_start)
                    current_start += clip_step
                if max_clip > 0:
                    start_times = start_times[:max_clip]
            else:
                start_times = [start_time]
                full_wave_data, sr = None, None

            for clip_idx, current_start in enumerate(start_times):
                for aug_idx in range(actual_n_aug):
                    if enable_multiclip:
                        random_offset = rng.random() * start_time
                        randomized_start_time = current_start - random_offset

                        start_sample = int(randomized_start_time * sr)
                        end_sample = start_sample + int(clip_duration * sr)
                        wave_data = full_wave_data[start_sample:end_sample]

                        target_length = int(clip_duration * sr)
                        if len(wave_data) < target_length:
                            wave_data = np.pad(wave_data, (0, target_length - len(wave_data)), mode='constant')
                    else:
                        wave_data, sr = clip_audio(
                            file_path=str(audio_file),
                            start_time=current_start,
                            duration=clip_duration,
                            target_sr=target_sr
                        )

                    augmented_wave = wave_data

                    if apply_noise and use_adaptive_snr:
                        adaptive_min_snr, adaptive_max_snr = get_adaptive_snr_range(augmented_wave)

                    if apply_silence and max_silence_duration > 0.0:
                        seed = rng.integers(np.iinfo(np.int32).max)
                        augmented_wave = insert_random_silence(
                            wave_data=augmented_wave, sr=sr,
                            max_silence_duration=max_silence_duration,
                            random_seed=seed
                        )

                    if apply_noise:
                        seed = rng.integers(np.iinfo(np.int32).max)
                        if use_adaptive_snr:
                            augmented_wave = add_noise_by_snr(
                                wave_data=augmented_wave,
                                min_snr_db=adaptive_min_snr,
                                max_snr_db=adaptive_max_snr,
                                strength=strength,
                                random_seed=seed
                            )
                        else:
                            augmented_wave = add_noise_by_snr(
                                wave_data=augmented_wave,
                                min_snr_db=min_snr_db,
                                max_snr_db=max_snr_db,
                                strength=strength,
                                random_seed=seed
                            )

                    if use_mel:
                        img_array = wave2image(
                            wave_data=augmented_wave, sr=sr, n_mels=n_mels,
                            n_fft_rate=n_fft_rate, hop_length_rate=hop_length_rate,
                            min_n_fft=min_n_fft,
                            n_fft_direct=n_fft_direct,
                            hop_length_direct=hop_length_direct,
                            fmax=fmax,
                            target_shape=target_shape, cmap=cmap
                        )
                    else:
                        img_array = wave2image_fullspec(
                            wave_data=augmented_wave, sr=sr,
                            n_fft_rate=n_fft_rate, hop_length_rate=hop_length_rate,
                            fmax=fmax, min_n_fft=min_n_fft,
                            target_shape=target_shape, cmap=cmap
                        )

                    label_value = getattr(row, label_name)
                    save_folder = positive_folder if label_value >= threshold else negative_folder

                    base_filename = Path(row.audio_file).stem
                    clip_suffix = f"_clip{clip_idx}" if enable_multiclip else ""
                    aug_suffix = f"_aug{aug_idx}" if actual_n_aug > 1 else ""
                    output_filename = f"{base_filename}{clip_suffix}{aug_suffix}.png"
                    output_path = save_folder / output_filename

                    img = Image.fromarray(img_array)
                    img.save(output_path)
                    success_count += 1

        except Exception as e:
            filename = getattr(row, 'audio_file', 'unknown')
            print(f"✗ エラー: {filename} - {str(e)}")
            error_count += 1

    print(f"\n変換完了: 生成された画像 {success_count}枚, エラーが発生したファイル {error_count}件")
