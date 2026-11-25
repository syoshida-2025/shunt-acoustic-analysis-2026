from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


def combine_metrics_csvs(
    input_paths: Sequence[Path | str],
    output_path: Path | str | None = None,
    split_key: Sequence[str] = ("condition_name", "repeat_id", "fold_id"),
) -> pd.DataFrame:
    """Combine multiple metrics CSV files into one DataFrame.

    If the same split key appears in multiple files, the later file overwrites the earlier one.
    This is useful when you want to merge previously run results with newly run results while
    keeping one row per split/condition combination.
    """

    frames: list[pd.DataFrame] = []
    for raw_path in input_paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Metrics file not found: {path}")
        frame = pd.read_csv(path)
        frames.append(frame)

    if not frames:
        raise ValueError("No input metrics CSV files were provided.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=list(split_key), keep="last")

    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path, index=False)

    return combined
