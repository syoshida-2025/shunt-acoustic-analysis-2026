"""utils/analysis.py

複数回に分けて実行した学習結果（複数の all_metrics.csv）を1つに統合する
combine_metrics_csvs、および n_fft x n_mels 交互作用を条件レベル平均で
簡易評価する evaluate_nfft_mels_interaction を提供する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


def evaluate_nfft_mels_interaction(
    df: pd.DataFrame,
    response_col: str = "auc_recording",
    control_factor: str = "fmax",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the n_fft × n_mels interaction using condition-level means.

    The analysis aggregates the repeated CV results to one row per condition,
    then fits an ANOVA-style linear model with main effects and the interaction
    term while including the control factor fmax.
    """

    condition_level = (
        df.groupby(["condition_name", "n_fft", "n_mels", control_factor], as_index=False)[response_col]
        .mean()
        .rename(columns={response_col: "mean_auc_recording"})
    )

    reduced_formula = "mean_auc_recording ~ C(n_fft) + C(n_mels) + C(fmax)"
    full_formula = reduced_formula + " + C(n_fft):C(n_mels)"

    reduced_model = smf.ols(formula=reduced_formula, data=condition_level).fit()
    full_model = smf.ols(formula=full_formula, data=condition_level).fit()

    anova_results = anova_lm(reduced_model, full_model)
    anova_results = anova_results.rename(index={" resid": "Residual"})

    # Add a simple effect-size proxy based on the F-test.
    if not anova_results.empty and "F" in anova_results.columns:
        anova_results["partial_eta_squared"] = (
            anova_results["F"] / (anova_results["F"] + anova_results["df_resid"])
        )

    return condition_level, anova_results


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
