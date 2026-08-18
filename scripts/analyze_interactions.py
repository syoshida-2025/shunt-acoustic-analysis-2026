"""scripts/analyze_interactions.py

n_fft x n_mels、n_mels x fmax の交互作用を、CV分割をランダム切片とする線形混合
効果モデルで評価する感度分析。analyze_statistics.py のANOVAとは独立に、要因間
の交互作用の頑健性を確認する目的で用いる。ヒートマップも合わせて出力する。
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze n_fft x n_mels and n_mels x fmax interactions with mixed effects and heatmaps")
    parser.add_argument("--input", default="results/metrics/all_metrics.csv")
    parser.add_argument("--output_dir", default="results/interaction_analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df = df.copy()
    df["split_id"] = df["repeat_id"].astype(str) + "_fold" + df["fold_id"].astype(str)

    # Use the repeated CV split as the random-effect grouping variable.
    # This is more appropriate than grouping by condition_name because the same split
    # is reused across conditions and should share a random intercept.
    # C(n_mels):C(fmax) is included alongside C(n_fft):C(n_mels) because a
    # full factorial ANOVA (see analyze_statistics.py) found the n_mels x fmax
    # interaction significant (p < 1e-3), unlike the n_fft x n_mels interaction.
    model = smf.mixedlm(
        "auc_recording ~ C(n_fft) + C(n_mels) + C(fmax) + C(n_fft):C(n_mels) + C(n_mels):C(fmax)",
        data=df,
        groups=df["split_id"],
    ).fit()

    params = pd.DataFrame({
        "estimate": model.params,
        "pvalue": model.pvalues,
    })
    params.to_csv(output_dir / "mixed_effects_params.csv")

    with open(output_dir / "mixed_effects_summary.txt", "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())

    # Heatmap for mean AUC by n_fft x n_mels, averaged over fmax.
    heatmap_df = (
        df.groupby(["n_fft", "n_mels"], as_index=False)["auc_recording"]
        .mean()
        .pivot(index="n_mels", columns="n_fft", values="auc_recording")
    )

    plt.figure(figsize=(7, 4.5))
    sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="viridis", cbar_kws={"label": "Mean AUC"})
    plt.title("Mean AUC by n_fft x n_mels")
    plt.xlabel("n_fft")
    plt.ylabel("n_mels")
    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_nfft_nmel.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Heatmap for mean AUC by n_fft x fmax, to check whether the pattern is consistent.
    heatmap_df_fmax = (
        df.groupby(["n_fft", "fmax"], as_index=False)["auc_recording"]
        .mean()
        .pivot(index="fmax", columns="n_fft", values="auc_recording")
    )

    plt.figure(figsize=(7, 4.5))
    sns.heatmap(heatmap_df_fmax, annot=True, fmt=".3f", cmap="viridis", cbar_kws={"label": "Mean AUC"})
    plt.title("Mean AUC by n_fft x fmax")
    plt.xlabel("n_fft")
    plt.ylabel("fmax")
    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_nfft_fmax.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Heatmap for mean AUC by n_mels x fmax, averaged over n_fft.
    # This is the interaction that the full factorial ANOVA found significant.
    heatmap_df_mels_fmax = (
        df.groupby(["n_mels", "fmax"], as_index=False)["auc_recording"]
        .mean()
        .pivot(index="fmax", columns="n_mels", values="auc_recording")
    )

    plt.figure(figsize=(7, 4.5))
    sns.heatmap(heatmap_df_mels_fmax, annot=True, fmt=".3f", cmap="viridis", cbar_kws={"label": "Mean AUC"})
    plt.title("Mean AUC by n_mels x fmax")
    plt.xlabel("n_mels")
    plt.ylabel("fmax")
    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_nmel_fmax.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved mixed-effects summary:", output_dir / "mixed_effects_summary.txt")
    print(
        "Saved heatmaps:",
        output_dir / "heatmap_nfft_nmel.png",
        output_dir / "heatmap_nfft_fmax.png",
        output_dir / "heatmap_nmel_fmax.png",
    )
    print(model.summary())


if __name__ == "__main__":
    main()
