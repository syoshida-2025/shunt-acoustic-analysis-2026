from pathlib import Path

import pandas as pd

from utils.analysis import combine_metrics_csvs


def test_combine_metrics_csvs_deduplicates_by_split_key(tmp_path: Path) -> None:
    first = pd.DataFrame(
        [
            {"condition_name": "cond_a", "repeat_id": 0, "fold_id": 0, "auc_recording": 0.8},
            {"condition_name": "cond_b", "repeat_id": 0, "fold_id": 0, "auc_recording": 0.7},
        ]
    )
    second = pd.DataFrame(
        [
            {"condition_name": "cond_b", "repeat_id": 0, "fold_id": 0, "auc_recording": 0.9},
            {"condition_name": "cond_c", "repeat_id": 0, "fold_id": 0, "auc_recording": 0.6},
        ]
    )

    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    output_path = tmp_path / "combined.csv"
    first.to_csv(first_path, index=False)
    second.to_csv(second_path, index=False)

    combined = combine_metrics_csvs([first_path, second_path], output_path=output_path)

    assert len(combined) == 3
    assert combined.loc[combined["condition_name"] == "cond_b", "auc_recording"].iloc[0] == 0.9
    assert output_path.exists()
