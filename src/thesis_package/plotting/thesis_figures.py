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
    ax_bar.set_xlabel(r"Spearman $|\rho|$ with GRS")

    #ax_bar.axvline(strong_threshold, linestyle="--", color="gray")

    # separation line
    if n_top > 0:
        # FIX 2: clip_on=False and xmax=1.5 forces the line to shoot into the gap between subplots
        ax_bar.axhline(n_top, color="black", linewidth=1.5, xmin=-0.65, xmax=1.5, clip_on=False, zorder=10)

    # Add a custom legend to explain the colors
    legend_elements = [
        Patch(facecolor='#4C78A8', label=r'Positive $\rho$ (> 0)'),
        Patch(facecolor='#E45756', label=r'Negative $\rho$ (< 0)')
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



# bow sensitivity anaylsis

def plot_single_metric_sensitivity(cluster_range, results_dict, metric='R2', baseline_value=None, top_k=18):
    """
    Plots a single metric across clusters with seed-based error bars and a baseline comparison.
    
    Parameters:
    -----------
    cluster_range : range or list
        The x-axis values (number of clusters).
    results_dict : dict
        The nested dictionary containing the results per seed.
    metric : str
        'R2', 'Corr', or 'MAE'.
    baseline_value : float, optional
        A static value (e.g., performance of baseline-only model) to plot as a horizontal line.
    """
    seeds = list(results_dict[metric].keys())
    cluster_list = list(cluster_range)
    
    # Calculate Mean and Std Dev across seeds
    data = np.array([results_dict[metric][c] for c in cluster_list])
    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)

    # Plotting
    plt.figure(figsize=(8, 5))

    # 1. Plot the error band (Shaded area)
    plt.fill_between(cluster_list, mean - std, mean + std, color='teal', alpha=0.15, label='±1 Std Dev')
    
    # 2. Plot the mean line
    plt.plot(cluster_list, mean, color='teal', marker='o', markersize=6, linewidth=2, label=f'Mean {metric}')

    # 3. Plot the baseline if provided
    if baseline_value is not None:
        plt.axhline(y=baseline_value, color='gray', linestyle='--', linewidth=2, alpha=0.8, label=f'Baseline')
    
    # 4. Plot top number of klusters
    if top_k is not None:
        plt.axvline(x=top_k, color='orange', linestyle='--', linewidth=2, alpha=0.8, label=f'Selected K')

    # Formatting
    #plt.title(f'Model Sensitivity: {metric} vs. Number of Clusters', fontsize=15, pad=15)
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel(f'Overall {metric}', fontsize=12)
    plt.xticks(cluster_list[::2])  # Show every 2nd tick to avoid crowding
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Remove top and right spines for a cleaner look
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.legend(frameon=True, loc='best')
    plt.tight_layout()

    plt.savefig(
        "cluster_sensitivity_plot.pdf",
        bbox_inches="tight",
        dpi=300
    )

    plt.show()


# BoW Selection SurgeMe interpretation

def plot_surgeme_stability(df_stability, global_centroids, title="Kinematic Stability of BoW Centroids Across 28 LOSO Folds", save_fig=False):
    """
    Plots a 2x2 grid of boxplots with selective axis labels and optional title.
    """
    features = ['path_ratio', 'spatial_spread', 'palm_area_cv', 'sparc']
    titles = ['Path Ratio (Efficiency)', 'Spatial Spread (Economy)', 
              'SPARC (Smoothness)', 'Palm Area CV (Pose Stability)']
    
    sns.set_context("paper", font_scale=1.2)
    plt.style.use("seaborn-v0_8-whitegrid")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=False, sharey=False)
    axes = axes.flatten()

    for i, feat in enumerate(features):
        sns.boxplot(
            data=df_stability,
            x='cluster_id',
            y=feat,
            hue='cluster_id',
            ax=axes[i],
            palette="viridis",
            legend=False,
            fliersize=2,
            linewidth=1.2
        )

        # Global Centroid Dot
        x_positions = np.arange(global_centroids.shape[0])
        y_positions = global_centroids[:, i]
        axes[i].scatter(
            x_positions, y_positions,
            color="red", s=20, marker="D",
            label="Global centroid", zorder=10
        )

        axes[i].set_title(titles[i], fontweight='bold', pad=10)

        # --- SELECTIVE AXIS LABELS ---
        # Only set Y-label for the left column (index 0 and 2)
        if i % 2 == 0:
            axes[i].set_ylabel("Feature Value")
        else:
            axes[i].set_ylabel("")

        # Only set X-label for the bottom row (index 2 and 3)
        if i >= 2:
            axes[i].set_xlabel("Surgeme (Global Cluster ID)")
        else:
            axes[i].set_xlabel("")

    plt.tight_layout()
    
    # --- CONDITIONAL TITLE ---
    if title is not None:
        plt.subplots_adjust(top=0.92)
        fig.suptitle(title, fontsize=16)

    # Show legend once
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    if save_fig:
        plt.savefig(
            "bow_loso_stability.pdf",
            bbox_inches="tight",
            dpi=300
        )

    plt.show()


def plot_surgeme_radar(global_centroids, feature_labels, cluster_info):
    """
    Thesis-quality radar plot for selected Surgeme centroids.
    Z-order is inverted: The first item in cluster_info is rendered on top.
    """
    num_vars = len(feature_labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close loop

    # Clean print-friendly style
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(7.8,7.8), subplot_kw=dict(polar=True))

    # Colorblind-friendly, high-contrast palette
    colors = ["#0072B2", "#D55E00", "#009E73"]

    # Radar orientation
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Plot selected clusters
    for i, info in enumerate(cluster_info):
        idx = info["id"]
        values = np.asarray(global_centroids[idx]).tolist()
        values += values[:1]

        label = f"BoW feature {idx} | β = {info['coef']:.2f}"
        
        # --- Z-ORDER LOGIC SWITCHED ---
        # First item (i=0) gets zorder 10, second gets 9, etc.
        current_zorder = 10 - i 
        
        ax.plot(
            angles,
            values,
            color=colors[i % len(colors)],
            linewidth=2.0,
            label=label,
            zorder=current_zorder,
        )
        ax.fill(
            angles,
            values,
            color=colors[i % len(colors)],
            alpha=0.14,
            zorder=current_zorder - 0.5, # Keep fill slightly behind its own line
        )

    # Radial range
    ax.set_ylim(-1.0, 1.5)

    # Radial grid + labels
    grid_values = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    ax.set_yticks(grid_values)
    ax.set_yticklabels([f"{v:.1f}" if v != 0 else "0" for v in grid_values],
                       fontsize=10, color="0.35")
    ax.set_rlabel_position(45)

    # Make the 0-reference ring slightly stronger
    ax.yaxis.grid(True, color="0.7", linewidth=0.8)
    ax.xaxis.grid(True, color="0.7", linewidth=0.8)
    theta_dense = np.linspace(0, 2 * np.pi, 512)
    ax.plot(theta_dense, np.zeros_like(theta_dense),
           linestyle="-", linewidth=1.0, color="0.35", zorder=1)

    # Outer spine
    ax.spines["polar"].set_linewidth(0.6)
    ax.spines["polar"].set_color("0.7")

    # Axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_labels, fontsize=13)

    # Legend
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(-0.14, 1.08),
        frameon=False,
        fontsize=11,
        handlelength=2.0,
    )

    plt.tight_layout()
    plt.savefig("radar_plot.pdf", bbox_inches="tight", dpi=300)
    plt.show()