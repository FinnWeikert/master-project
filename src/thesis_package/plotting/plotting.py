"""
All plotting functions not used in thesis but still useful for analysis and debugging.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

##############################################################################################################################################
############################################################## Global Features ###############################################################
##############################################################################################################################################


def plot_panel_a_feature_correlations(
    df_full,
    features,
    target="QRS_Overal",
    threshold=0.5,
    figsize=(8, 8),
    pos_color="#4C78A8AC",   # blue
    neg_color="#E456569B"    # red
):
    corrs = []

    # --- Compute correlations ---
    for feat in features:
        valid = df_full[[feat, target]].dropna()
        r, _ = spearmanr(valid[feat], valid[target])
        corrs.append(r)

    df_corr = pd.DataFrame({
        "feature": features,
        "spearman_corr": corrs
    })

    # --- Rank by absolute correlation ---
    df_corr["abs_corr"] = df_corr["spearman_corr"].abs()
    df_corr["sign"] = np.where(df_corr["spearman_corr"] >= 0, "Positive", "Negative")
    df_corr = df_corr.sort_values("abs_corr", ascending=False).reset_index(drop=True)

    # --- Colors by sign ---
    palette = {
        "Positive": pos_color,
        "Negative": neg_color
    }

    # --- Plot ---
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=df_corr,
        x="spearman_corr",
        y="feature",
        hue="sign",
        dodge=False,
        palette=palette,
        orient="h"
    )

    # remove duplicate legend entries if seaborn creates them oddly
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Sign", loc="lower right")
    #ax.get_legend().remove()

    # --- Reference lines ---
    plt.axvline(0, color="black", linewidth=1.2)
    plt.axvline(threshold, linestyle="--", color="gray", linewidth=1.2)
    plt.axvline(-threshold, linestyle="--", color="gray", linewidth=1.2)

    # --- Highlight strong-correlation zone (top ranked features above threshold) ---
    strong_n = (df_corr["abs_corr"] >= threshold).sum()
    if strong_n > 0:
        ax.axhspan(-0.5, strong_n - 0.5, color="gray", alpha=0.12, zorder=0)
        ax.text(
            0.98, 0.98,
            f"Strong-correlation region\n(|r| ≥ {threshold:.1f}, n = {strong_n})",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="lightgray")
        )

    plt.xlabel("Spearman Correlation with GRS")
    plt.ylabel("Feature")
    plt.title("Ranked Correlation of Motion Descriptors with Surgical Skill")

    plt.tight_layout()
    plt.show()

    return df_corr



def plot_panel_b_intercorrelation_heatmap(
    df_full,
    df_corr,
    target="QRS_Overal",
    plot_threshold=0.0,
    strong_threshold=0.5,
    figsize=(9,8)
):

    # select features above plot threshold
    df_corr = df_corr[df_corr["abs_corr"] >= plot_threshold]

    # features to include in the heatmap
    plot_features = df_corr["feature"].tolist()

    # strongly correlated subset
    strong_features = df_corr[df_corr["abs_corr"] >= strong_threshold]["feature"].tolist()
    n_top = len(strong_features)

    if n_top < 2:
        raise ValueError("Not enough features above threshold to compute intercorrelation.")

    # compute absolute correlation matrix
    corr_matrix = df_full[plot_features].corr(method="pearson").abs()

    plt.figure(figsize=figsize)

    ax = sns.heatmap(
        corr_matrix,
        annot=False,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Absolute Pearson Correlation", "shrink": 0.8}
    )

    # draw separation lines between top-signal features and the rest
    ax.axhline(n_top, color="black", linewidth=1.5)
    ax.axvline(n_top, color="black", linewidth=1.5)

    #plt.title("Intercorrelation of Motion Descriptors")

    # remove x labels to reduce clutter
    plt.xticks([])
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()

    return corr_matrix



##############################################################################################################################################
############################################################## Global Features ###############################################################
##############################################################################################################################################