"""
All plotting functions for the thesis.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import pearsonr, spearmanr

def plot_combined_feature_screening_figure(
    df_full,
    features,
    target="QRS_Overal",
    strong_threshold=0.5,
    figsize=(12, 9)
):

    # --- Compute correlations ---
    corrs = []
    for feat in features:
        valid = df_full[[feat, target]].dropna()
        r, _ = spearmanr(valid[feat], valid[target])
        corrs.append(r)

    df_corr = pd.DataFrame({
        "feature": features,
        "spearman_corr": corrs
    })

    df_corr["abs_corr"] = df_corr["spearman_corr"].abs()

    # sort by absolute correlation
    df_corr = df_corr.sort_values("abs_corr", ascending=False).reset_index(drop=True)
    ordered_features = df_corr["feature"].tolist()

    # compute intercorrelation matrix
    corr_matrix = df_full[ordered_features].corr().abs()

    n_features = len(df_corr)
    n_top = (df_corr["abs_corr"] >= strong_threshold).sum()

    # --- Create figure with constrained layout ---
    fig, (ax_bar, ax_heat) = plt.subplots(
        1, 2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [2, 4]},
        constrained_layout=True
    )

    y = np.arange(n_features) + 0.5

    # FIX 1: Generate a list of colors based on the sign of the original correlation
    # Using slightly muted blue and red for a polished look
    bar_colors = ["#4C78A8" if val >= 0 else "#E45756" for val in df_corr["spearman_corr"]]

    # ===== LEFT PANEL: correlation bars =====
    ax_bar.barh(y, df_corr["abs_corr"], color=bar_colors)

    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(df_corr["feature"])
    ax_bar.set_ylim(n_features, 0)
    ax_bar.grid(axis="x", linestyle="--", alpha=0.7)

    ax_bar.set_xlabel(r"$|r|$ with GRS")
    #ax_bar.axvline(strong_threshold, linestyle="--", color="gray")

    # separation line
    if n_top > 0:
        # FIX 2: clip_on=False and xmax=1.5 forces the line to shoot into the gap between subplots
        ax_bar.axhline(n_top, color="black", linewidth=1.5, xmin=-0.65, xmax=1.5, clip_on=False, zorder=10)

    # Add a custom legend to explain the colors
    legend_elements = [
        Patch(facecolor='#4C78A8', label='Positive Corr (> 0)'),
        Patch(facecolor='#E45756', label='Negative Corr (< 0)')
    ]
    ax_bar.legend(handles=legend_elements, loc='lower right', fontsize=10)

    # cleaner look
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # ===== RIGHT PANEL: heatmap =====
    sns.heatmap(
        corr_matrix,
        ax=ax_heat,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        linewidths=0.4,
        linecolor="white",
        #cbar_kws={"label": "Absolute Pearson Correlation"}
    )

    ax_heat.collections[0].colorbar.set_label("Absolute Pearson Correlation", fontsize=12)

    ax_heat.set_xticks([])
    ax_heat.set_yticks(y) 
    ax_heat.set_yticklabels([])

    # separation lines
    if n_top > 0:
        # FIX 2 (cont): xmin=-0.1 catches the line coming from the left plot, sealing the gap perfectly
        ax_heat.axhline(n_top, color="black", linewidth=1.5, xmin=-0.1, clip_on=False, zorder=10)
        ax_heat.axvline(n_top, color="black", linewidth=1.5)

    #ax_heat.set_title("Intercorrelation of Motion Descriptors")

    # save 
    plt.savefig(
        "feature_screening_combined.pdf",
        bbox_inches="tight",
        dpi=300
    )

    plt.show()

    return df_corr, corr_matrix