"""utils/dataset.py — PyTorch Dataset と DataLoader 構築。

メルスペクトログラム画像（PNG）を PyTorch Dataset として扱い、
ImageNet 統計での正規化と訓練時の Brightness augmentation を適用する。

主な公開関数:
    - ShuntMelDataset: クラス。1サンプル = 1クリップの画像とラベル。
    - build_dataloader: DataFrame と condition_name を受け取り、DataLoader を返す。
    - build_transform: mode に応じた transforms.Compose を構築。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ===== Dataset クラス =====

class ShuntMelDataset(Dataset):
    """シャント音メルスペクトログラム画像の PyTorch Dataset。

    1 サンプル = 1 クリップに対応する PNG 画像とラベル。

    画像ファイルパスの構築:
        {images_root}/{condition_name}/{positive|negative}/{clip_basename}.png

    注意:
        - process_audio_to_images は、素の（拡張なし）画像を1枚（サフィックスなし）
          + augmentation設定時は追加でノイズ拡張画像（_aug0, _aug1, ...）を生成する。
        - 訓練時は「素の画像 + 拡張画像すべて」をディスクから読む。
        - 検証・テスト時は素の画像（サフィックスなし）のみを使用する。
    """

    def __init__(
        self,
        df: pd.DataFrame,
        condition_name: str,
        images_root: str | Path,
        transform: transforms.Compose,
        mode: str,
        clip_metadata: pd.DataFrame | None = None,
    ):
        """
        Args:
            df: クリップレベルの DataFrame
                必須カラム: clip_file, patient_id, label
            condition_name: 例 "nfft256_mel64_fmax1000"
            images_root: 画像ルートディレクトリ（config["paths"]["images_root"]）
            transform: build_transform で構築した transforms.Compose
            mode: "train"（素の画像+拡張画像すべてを使用）、"val"/"test"（素の画像のみ使用）
            clip_metadata: 全クリップメタデータ（補助情報取得用、Noneでも動作）

        Raises:
            FileNotFoundError: condition_dir が見つからない場合。
        """
        self.df = df.reset_index(drop=True).copy()
        self.condition_name = condition_name
        self.images_root = Path(images_root)
        self.transform = transform
        self.mode = mode

        # 画像ディレクトリ確認
        self.condition_dir = self.images_root / condition_name
        if not self.condition_dir.exists():
            raise FileNotFoundError(
                f"画像ディレクトリが見つかりません: {self.condition_dir}\n"
                f"先に generate_melspectrogram.py を実行してください。"
            )

        # 各サンプルの画像パスを事前計算（高速化）
        self.image_paths = self._build_image_paths()

    def _build_image_paths(self) -> list[Path]:
        """各サンプルの画像パスを構築する。

        訓練時: 素の画像（サフィックスなし）+ _aug0 〜 _augN すべてを使用
        検証・テスト時: 素の画像（サフィックスなし）のみを使用

        Returns:
            各サンプルの画像パスリスト。
        """
        paths: list[Path] = []
        for _, row in self.df.iterrows():
            label_dir = "positive" if row["label"] == 1 else "negative"
            clip_basename = Path(row["clip_file"]).stem
            label_folder = self.condition_dir / label_dir
            plain_path = label_folder / f"{clip_basename}.png"

            if self.mode == "train":
                # 素の画像 + 拡張済み画像すべてを使用
                aug_files = sorted(label_folder.glob(f"{clip_basename}_aug*.png"))
                train_files = ([plain_path] if plain_path.exists() else []) + aug_files
                if not train_files:
                    raise FileNotFoundError(
                        f"画像が見つかりません: {clip_basename} (in {label_dir})"
                    )
                paths.extend(train_files)
            else:
                # val/test: 素の画像を1枚だけ
                # （_aug0 へのフォールバックは、process_audio_to_images が
                #   素の画像を必ず生成するため通常は発火しない安全策として残す）
                aug0_path = label_folder / f"{clip_basename}_aug0.png"
                if plain_path.exists():
                    paths.append(plain_path)
                elif aug0_path.exists():
                    paths.append(aug0_path)
                else:
                    raise FileNotFoundError(
                        f"画像が見つかりません: {clip_basename} (in {label_dir})"
                    )

        # 訓練時はラベル・clip_file 情報も拡張倍数分に展開する
        if self.mode == "train":
            self._expand_df_for_train(paths)

        return paths

    def _expand_df_for_train(self, image_paths: list[Path]) -> None:
        """訓練時、画像パスの数に合わせて self.df を展開する。

        各クリップに対応する画像（素の画像1枚 + 拡張画像N枚）ごとに DataFrame の行を複製する。
        """
        expanded_rows = []
        for path in image_paths:
            basename = path.stem  # "A0101_clip0_aug3"
            if "_aug" in basename:
                clip_basename = basename.rsplit("_aug", 1)[0]  # "A0101_clip0"
            else:
                clip_basename = basename

            clip_file = f"{clip_basename}.wav"
            match = self.df[self.df["clip_file"] == clip_file]
            if len(match) == 0:
                raise ValueError(f"DataFrame に対応行がありません: {clip_file}")
            expanded_rows.append(match.iloc[0])

        self.df = pd.DataFrame(expanded_rows).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """1 サンプルを取得する。

        Returns:
            以下のキーを持つ辞書:
                - "image": Tensor, shape (3, H, W), normalized
                - "label": int (0 or 1)
                - "clip_file": str (元のクリップファイル名、推論時の集約用)
                - "patient_id": str
        """
        row = self.df.iloc[idx]
        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        return {
            "image": image_tensor,
            "label": int(row["label"]),
            "clip_file": row["clip_file"],
            "patient_id": row["patient_id"],
        }


# ===== Transform 構築 =====

def build_transform(mode: str, config: dict) -> transforms.Compose:
    """訓練・検証・テスト用の transforms を構築する。

    変換パイプライン:
        1. ColorJitter (Brightness): 訓練時のみ
        2. Resize: image_size (224, 224)
        3. ToTensor: PIL [0, 255] uint8 → Tensor [0, 1] float32
        4. Normalize: ImageNet 統計でZ-score正規化

    Args:
        mode: "train", "val", "test"
        config: 全 config。

    Returns:
        transforms.Compose。
    """
    norm_config = config["normalization"]
    image_size = tuple(config["mel_conditions"]["image_size"])
    brightness = config["augmentation"]["brightness_jitter"]

    transform_list: list = []

    # 1. Brightness augmentation (訓練時のみ)
    if mode == "train" and brightness > 0:
        transform_list.append(transforms.ColorJitter(brightness=brightness))

    # 2. Resize
    transform_list.append(transforms.Resize(image_size))

    # 3. ToTensor
    transform_list.append(transforms.ToTensor())

    # 4. ImageNet 正規化
    if norm_config["imagenet_normalization"]:
        transform_list.append(transforms.Normalize(
            mean=norm_config["imagenet_mean"],
            std=norm_config["imagenet_std"],
        ))

    return transforms.Compose(transform_list)


# ===== DataLoader 構築 =====

def build_dataloader(
    df: pd.DataFrame,
    condition_name: str,
    config: dict,
    mode: str,
) -> DataLoader:
    """指定 mode の DataLoader を構築する。

    DataLoader 設定:
        - 訓練時: shuffle=True, drop_last=False
        - 検証・テスト時: shuffle=False, drop_last=False
        - pin_memory=True (GPU 転送高速化)
        - num_workers: config から取得

    Args:
        df: クリップレベル DataFrame。
        condition_name: 例 "nfft256_mel64_fmax1000"。
        config: 全 config。
        mode: "train", "val", "test"。

    Returns:
        DataLoader インスタンス。

    Raises:
        ValueError: mode が無効な場合。
    """
    if mode not in ("train", "val", "test"):
        raise ValueError(f"mode は 'train', 'val', 'test' のいずれかである必要があります。got '{mode}'")

    transform = build_transform(mode, config)
    dataset = ShuntMelDataset(
        df=df,
        condition_name=condition_name,
        images_root=config["paths"]["images_root"],
        transform=transform,
        mode=mode,
    )

    train_config = config["training"]
    is_train = (mode == "train")

    # 再現性のための generator
    generator = torch.Generator()
    generator.manual_seed(42)

    return DataLoader(
        dataset=dataset,
        batch_size=train_config["batch_size"],
        shuffle=is_train,
        num_workers=train_config["num_workers"],
        pin_memory=train_config.get("pin_memory", True),
        drop_last=False,
        generator=generator if is_train else None,
        persistent_workers=(train_config["num_workers"] > 0),
    )
