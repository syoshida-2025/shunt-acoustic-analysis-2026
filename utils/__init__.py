"""shunt_melspec_v2 の再利用可能なモジュール群。"""

from .data_split import split_train_test_by_patient, print_ratio
from .augmentation import add_noise_by_snr, insert_random_silence
from .sound2image import (
    clip_audio,
    wave2image,
    process_audio_to_images,
    get_adaptive_snr_range,
)
from .df_from_folder import create_df
from .metadata_io import (
    read_recording_metadata,
    write_recording_metadata,
    read_clip_metadata,
    write_clip_metadata,
    validate_patient_id_consistency,
    validate_label_consistency,
    validate_audio_files_exist,
    ensure_columns,
    write_missing_data_report,
    REQUIRED_RECORDING_COLUMNS,
    DERIVED_RECORDING_COLUMNS,
    CLIP_METADATA_COLUMNS,
)

# Part 4 で追加（学習関連）:
# from .dataset import ShuntMelDataset, build_dataloader
# from .trainer import train_one_fold, predict_on_loader
# from .checkpoint import load_or_init_status, save_status, mark_condition_complete
# from .metrics import compute_all_metrics, find_youden_threshold, aggregate_clip_to_recording

__all__ = [
    # データ分割
    "split_train_test_by_patient",
    "print_ratio",
    # データ拡張
    "add_noise_by_snr",
    "insert_random_silence",
    # 音声→画像変換
    "clip_audio",
    "wave2image",
    "process_audio_to_images",
    "get_adaptive_snr_range",
    # フォルダからDataFrame構築
    "create_df",
    # メタデータI/O
    "read_recording_metadata",
    "write_recording_metadata",
    "read_clip_metadata",
    "write_clip_metadata",
    "validate_patient_id_consistency",
    "validate_label_consistency",
    "validate_audio_files_exist",
    "ensure_columns",
    "write_missing_data_report",
    "REQUIRED_RECORDING_COLUMNS",
    "DERIVED_RECORDING_COLUMNS",
    "CLIP_METADATA_COLUMNS",
]
