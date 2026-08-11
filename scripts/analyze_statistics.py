"""scripts/analyze_statistics.py

results/metrics/all_metrics.csv（27条件 x 30 CV分割の評価指標）に対して、
n_fft・n_mels・fmax を要因とする分散分析（Type II ANOVA、CV分割をブロック因子と
して含む）と Tukey HSD 多重比較を実行し、results/statistics/ に保存する。
"""

import argparse
import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ANOVA and Tukey HSD on results/metrics/all_metrics.csv"
    )
    parser.add_argument(
        "--input",
        default="results/metrics/all_metrics.csv",
        help="Path to metrics CSV file.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/statistics",
        help="Directory to save ANOVA/Tukey output.",
    )
    parser.add_argument(
        "--formula",
        default="auto",
        help=(
            "Statsmodels formula for ANOVA. 'auto' builds "
            "auc_recording ~ C(n_fft) + C(n_mels) + C(fmax) + C(split_id) "
            "from the active factors, where split_id (repeat_id x fold_id) is "
            "included as a blocking factor because the same 30 CV splits are "
            "reused across all conditions and are therefore not independent "
            "observations."
        ),
    )
    parser.add_argument(
        "--no_block",
        action="store_true",
        help=(
            "Disable the split_id blocking factor and run a plain factorial "
            "ANOVA instead. Not recommended: the 30 CV splits are shared "
            "across conditions, so treating all rows as independent inflates "
            "the F-statistics / understates p-values (pseudoreplication)."
        ),
    )
    parser.add_argument(
        "--no_interactions",
        action="store_true",
        help=(
            "Disable the two-way interaction terms (C(a):C(b) for every pair "
            "of active factors) and fit main effects only. The 'auto' formula "
            "includes all pairwise interactions by default because n_mels:fmax "
            "was found to be a significant interaction in this dataset."
        ),
    )
    return parser.parse_args()


def make_output_dir(path: str) -> Path:
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def compute_partial_eta_squared(anova_df: pd.DataFrame) -> pd.Series:
    residual_ss = anova_df.loc["Residual", "sum_sq"]
    partial_eta = anova_df["sum_sq"] / (anova_df["sum_sq"] + residual_ss)
    partial_eta.loc["Residual"] = np.nan
    return partial_eta


def run_anova(df: pd.DataFrame, formula: str) -> pd.DataFrame:
    model = smf.ols(formula=formula, data=df).fit()
    anova_results = anova_lm(model, typ=2)
    anova_results["partial_eta_squared"] = compute_partial_eta_squared(anova_results)
    return anova_results


def get_active_factors(df: pd.DataFrame, factors: list[str]) -> list[str]:
    active = [factor for factor in factors if df[factor].nunique() > 1]
    return active


def add_split_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add a split_id column (repeat_id x fold_id) identifying each of the 30
    CV splits that are shared across all conditions.
    """
    df = df.copy()
    df["split_id"] = df["repeat_id"].astype(str) + "_fold" + df["fold_id"].astype(str)
    return df


def add_block_adjusted_auc(
    df: pd.DataFrame,
    value_col: str = "auc_recording",
    block_col: str = "split_id",
    adjusted_col: str = "auc_recording_block_adj",
) -> pd.DataFrame:
    """Remove the split-level (block) mean before running Tukey HSD.

    Tukey HSD on the raw AUC would ignore the fact that the same 30 CV splits
    recur across every condition (they are the blocking factor in the ANOVA).
    Subtracting each split's mean AUC and re-adding the grand mean keeps the
    factor-level differences intact while removing the split-to-split
    variance, so the pairwise comparisons are consistent with the blocked
    ANOVA model rather than confounded by which splits happen to be easier
    or harder.
    """
    df = df.copy()
    split_mean = df.groupby(block_col)[value_col].transform("mean")
    grand_mean = df[value_col].mean()
    df[adjusted_col] = df[value_col] - split_mean + grand_mean
    return df


def run_tukey(
    df: pd.DataFrame,
    factor: str,
    output_dir: Path,
    endog_col: str = "auc_recording_block_adj",
) -> pd.DataFrame:
    group = df[factor].astype(str)
    tukey = pairwise_tukeyhsd(endog=df[endog_col], groups=group, alpha=0.05)
    tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
    tukey_df.to_csv(output_dir / f"tukey_{factor}.csv", index=False)
    return tukey_df


def save_text_report(output_dir: Path, df: pd.DataFrame, path: str) -> None:
    with open(output_dir / path, "w", encoding="utf-8") as f:
        f.write(df.to_string())
        f.write("\n")


def main() -> None:
    args = parse_args()
    output_dir = make_output_dir(args.output_dir)
    df = pd.read_csv(args.input)

    print("Loaded metrics:", args.input)
    print("Rows:", len(df), "Conditions:", df["condition_name"].nunique())

    base_factors = ["n_fft", "n_mels", "fmax"]
    active_factors = get_active_factors(df, base_factors)
    if not active_factors:
        raise ValueError("No factor has more than one unique value for ANOVA/Tukey analysis.")

    use_block = not args.no_block
    if use_block:
        df = add_split_id(df)
        df = add_block_adjusted_auc(df)
        n_splits = df["split_id"].nunique()
        print(f"Blocking factor split_id (repeat_id x fold_id): {n_splits} levels")

    if args.formula == "auto":
        terms = [f"C({factor})" for factor in active_factors]
        if not args.no_interactions:
            for f1, f2 in itertools.combinations(active_factors, 2):
                terms.append(f"C({f1}):C({f2})")
        if use_block:
            terms.append("C(split_id)")
        formula = "auc_recording ~ " + " + ".join(terms)
    else:
        formula = args.formula

    skipped = [factor for factor in base_factors if factor not in active_factors]
    if skipped:
        print("Skipping single-level factors for ANOVA/Tukey:", skipped)
    print("Using formula:", formula)

    anova_df = run_anova(df, formula)
    anova_df.to_csv(output_dir / "anova_table.csv")
    save_text_report(output_dir, anova_df, "anova_table.txt")
    print("Saved ANOVA results:", output_dir / "anova_table.csv")

    factor_means = (
        df.groupby(["n_fft", "n_mels", "fmax"])["auc_recording"]
        .mean()
        .reset_index()
        .rename(columns={"auc_recording": "mean_auc_recording"})
    )
    factor_means.to_csv(output_dir / "condition_means.csv", index=False)
    print("Saved condition means:", output_dir / "condition_means.csv")

    tukey_endog_col = "auc_recording_block_adj" if use_block else "auc_recording"

    summary = []
    for factor in active_factors:
        tukey_df = run_tukey(df, factor, output_dir, endog_col=tukey_endog_col)
        summary.append((factor, tukey_df))
        print(f"Saved Tukey HSD for {factor}: {output_dir / f'tukey_{factor}.csv'}")

    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("ANOVA and Tukey HSD analysis\n")
        f.write("===========================\n")
        f.write(f"Input: {args.input}\n")
        f.write(f"Formula: {formula}\n")
        if use_block:
            f.write(
                "Note: split_id (repeat_id x fold_id) is included as a blocking "
                "factor because the same 30 CV splits recur across all conditions. "
                "Tukey HSD below was run on split-mean-adjusted AUC "
                "(auc_recording_block_adj) for consistency with the blocked ANOVA.\n"
            )
        else:
            f.write(
                "Note: --no_block was set; rows are treated as independent, which "
                "is anti-conservative given the shared CV splits across conditions.\n"
            )
        f.write("\nANOVA table:\n")
        f.write(anova_df.to_string())
        f.write("\n\n")
        for factor, tukey_df in summary:
            f.write(f"Tukey HSD: {factor}\n")
            f.write(tukey_df.to_string(index=False))
            f.write("\n\n")

    print("Saved summary report:", output_dir / "summary.txt")
    print("ANOVA/Tukey analysis complete.")


if __name__ == "__main__":
    main()
