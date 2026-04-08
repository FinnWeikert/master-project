"""
All plotting functions not used in thesis but still useful for analysis and debugging.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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
############################################################## Winodw Features ###############################################################
##############################################################################################################################################

def plot_correlation_matrix(df, title="Feature Correlation Matrix"):
    # Calculate the correlation matrix
    corr = df.corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))

    # Create the heatmap
    sns.heatmap(corr, 
                annot=True,          # Show the correlation values
                fmt=".2f",           # Format to 2 decimal places
                cmap='coolwarm',     # Standard diverging colormap
                center=0,            # Ensure 0 is the neutral color
                square=True,         # Force square cells
                linewidths=.5,       # Add small lines between cells
                cbar_kws={"shrink": .8})

    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.show()


def plot_sensitivity_results(cluster_range, results_dict, title="Cluster Sensitivity Analysis", base_corr=0.7165, base_mae=5.2231, base_r2=0.5021):
    """
    Plots mean performance metrics with standard deviation bands across multiple seeds.
    
    Parameters:
    -----------
    cluster_range : range or list
        The x-axis values (number of clusters).
    results_dict : dict
        The nested dictionary containing 'MAE', 'Corr', and 'R2' with seed-specific lists.
    """
    n_seeds = len(list(results_dict['R2'].values())[0])
    cluster_list = list(cluster_range)
    
    # Helper to calculate stats
    def get_stats(metric_dict):
        # Convert dict of lists to a 2D array: (seeds, n_clusters)
        arr = np.array([metric_dict[cluster] for cluster in cluster_range])
        return np.mean(arr, axis=1), np.std(arr, axis=1)

    # Calculate statistics
    r2_mean, r2_std = get_stats(results_dict['R2'])
    corr_mean, corr_std = get_stats(results_dict['Spearman_R'])
    mae_mean, mae_std = get_stats(results_dict['MAE'])

    # Create Subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{title} ({n_seeds} Seeds)", fontsize=16, fontweight='bold', y=1.05)

    metrics = [
        ('Overall R²', r2_mean, r2_std, 'royalblue', base_r2),
        ('Spearman Correlation', corr_mean, corr_std, 'forestgreen', base_corr),
        ('Overall MAE', mae_mean, mae_std, 'crimson', base_mae)
    ]

    for i, (label, mean, std, color, base_value) in enumerate(metrics):
        ax = axes[i]
        
        # Plot mean line
        ax.plot(cluster_list, mean, color=color, marker='o', markersize=4, label='Mean', linewidth=2)
        
        # Plot standard deviation band
        ax.fill_between(cluster_list, mean - std, mean + std, color=color, alpha=0.2, label='±1 Std Dev')
        
        # Formatting
        ax.set_title(label, fontsize=14)
        ax.set_xlabel('Number of Clusters', fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_xticks(list(cluster_range)[::2])
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.axhline(base_value, color='gray', linestyle='--', label=f'Baseline', alpha=0.7)
        
        if i == 0: # Only one legend to keep it clean
            ax.legend()

    plt.tight_layout()
    plt.show()

###############################################################################################################################################
############################################################## Predicted vs. True Plots #######################################################
###############################################################################################################################################

def plot_predicted_vs_true(
    df,
    true_col="True",
    pred_col="Pred",
    group_col="Group",
    title="Predicted vs. True GRS",
    ax=None,
    show_group_labels=False,
    annotate_metrics=True,
    add_fit_line=True,
    point_alpha=0.6,
    point_size=50,
):
    """
    Plot predicted vs. true scores for a single model.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns for true and predicted values.
    true_col : str
        Name of the column with ground-truth scores.
    pred_col : str
        Name of the column with predicted scores.
    group_col : str
        Optional grouping/sample identifier column.
    title : str
        Plot title.
    ax : matplotlib.axes.Axes or None
        If provided, draw on this axis. Otherwise create a new figure.
    show_group_labels : bool
        If True, annotate each point with the value in `group_col`.
    annotate_metrics : bool
        If True, show R², RMSE, MAE, and Pearson r in the plot.
    add_fit_line : bool
        If True, add a least-squares regression line.
    point_alpha : float
        Alpha transparency for scatter points.
    point_size : float
        Scatter point size.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis containing the plot.
    metrics : dict
        Dictionary with computed metrics.
    """
    data = df[[true_col, pred_col] + ([group_col] if group_col in df.columns else [])].dropna().copy()

    y_true = data[true_col].to_numpy()
    y_pred = data[pred_col].to_numpy()

    # Metrics
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    spearman_r, _ = spearmanr(y_true, y_pred)

    metrics = {
        "R2": r2,
        "MAE": mae,
        "Spearman_r": spearman_r,
    }

    # Create axis if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Scatter
    ax.scatter(y_true, y_pred, s=point_size, alpha=point_alpha, edgecolors='w', linewidths=0.5)

    # Axis limits with small margin
    combined = np.concatenate([y_true, y_pred])
    data_min = np.min(combined)
    data_max = np.max(combined)
    margin = 0.05 * (data_max - data_min) if data_max > data_min else 1.0
    lim_min = data_min - margin
    lim_max = data_max + margin

    # Identity line
    ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", linewidth=1.0, color="gray", label="Ideal: y = x")

    # Fitted regression line
    if add_fit_line and len(data) >= 2:
        slope, intercept = np.polyfit(y_true, y_pred, 1)
        x_fit = np.linspace(lim_min, lim_max, 200)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, linewidth=1.5, label=f"Fit: y = {slope:.2f}x + {intercept:.2f}")

    # Optional point labels
    if show_group_labels and group_col in data.columns:
        for _, row in data.iterrows():
            ax.annotate(
                str(row[group_col]),
                (row[true_col], row[pred_col]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    # Labels and styling
    ax.set_title(title)
    ax.set_xlabel("True GRS (mean expert score)")
    ax.set_ylabel("Predicted GRS")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal", adjustable="box")
    #ax.grid(True, alpha=0.3)

    if annotate_metrics:
        text = (
            f"$R^2$ = {r2:.3f}\n"
            f"MAE = {mae:.2f}\n"
            f"ρ = {spearman_r:.3f}\n"
            f"n = {len(data)}"
        )
        ax.text(
            0.05,
            0.95,
            text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", alpha=0.1),
        )

    ax.legend(loc="lower right")
    return ax, metrics