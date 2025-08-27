from typing import Iterable, Sequence
import os
import matplotlib.pyplot as plt

#TODO: revamp plotting logic to be more modular and reusable
def plot_and_save_brc_curves(
    bias_diffs: Sequence[float],
    random_diffs: Sequence[float],
    orth_diffs: Sequence[float],
    alpha_values: Iterable[float],
    inj_layer: int,
    read_layer: int,
    inject_site: str,
    read_site: str,
    out_dir: str,
    y_limits: tuple[float, float],
    metric_name: str = "logit_diffs",
    use_log_scale: bool = False,
    dataset_name: str = "reassurance",
    model_name: str = "gpt2",
    log_scale_both: bool = False,  # Add parameter for log scale on both axes
) -> None:
    """
    Plots and saves BRC curves for bias, random, and orth vectors.
    """
    # Handle rank changes separately 
    if metric_name == "rank_changes":
        plot_rank_changes_and_save(
            bias_diffs=bias_diffs,
            random_diffs=random_diffs,
            orth_diffs=orth_diffs,
            alpha_values=alpha_values,
            inj_layer=inj_layer,
            read_layer=read_layer,
            inject_site=inject_site,
            read_site=read_site,
            out_dir=out_dir,
            y_limits=y_limits,
            dataset_name=dataset_name,
            model_name=model_name,
        )
        return
    # ================ STYLES AND COLORS ================
    colors = {
        "bias": ("#0072B2", 2.5),
        "random": ("#D55E00", 2.0),
        "orth": ("#009E73", 2.0),
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6))

    # ================ DATA PREPARATION ================
    # Convert to percentage if prob_diffs for better readability
    if metric_name == "prob_diffs":
        series = {
            "bias": [x * 100 for x in bias_diffs],  # Convert to percentage
            "random": [x * 100 for x in random_diffs],
            "orth": [x * 100 for x in orth_diffs],
        }
        # Scale y-limits to match the percentage conversion
        scaled_y_limits = (y_limits[0] * 100, y_limits[1] * 100)
    else:
        series = {
            "bias": list(bias_diffs),
            "random": list(random_diffs),
            "orth": list(orth_diffs),
        }
        scaled_y_limits = y_limits

    alpha_list = list(alpha_values)

    # ================ PLOTTING ================
    for name in ["bias", "random", "orth"]:
        color, lw = colors[name]
        ax.plot(alpha_list, series[name], label=name, color=color, linewidth=lw, marker="o", markersize=3)

    # Add reference lines based on metric type #TODO: think about what other reference lines to add for other metrics and probably move logic to a separate function
    if metric_name == "odds_ratios":
        ax.axhline(1, color="black", linestyle="--", linewidth=1, alpha=0.7, label="Parity (1×)")
    else:
        ax.axhline(0, color="black", linestyle="--", linewidth=1)

    # ================ LABELS AND TITLE ================
    ax.set_xlabel(r"Steering coefficient $\alpha$", fontsize=14)
    
    # Dynamic y-axis label based on metric
    ylabel = get_ylabel(metric_name)
    
    # Add percentage suffix to y-label if enabled
    if metric_name == "prob_diffs":
        ylabel += " (%)"
    
    ax.set_ylabel(ylabel, fontsize=14)
    
    # Add log scale suffix to title if enabled
    if log_scale_both:
        title_suffix = " (log scale both axes)"
    elif use_log_scale:
        title_suffix = " (log scale)"
    else:
        title_suffix = ""
    ax.set_title(
        f"BRC ({metric_name}){title_suffix} | inject: L{inj_layer}:{inject_site} → read: L{read_layer}:{read_site}",
        fontsize=15,
        weight="bold",
    )
    
    # Apply log scale based on parameters
    # Choose appropriate linthresh based on metric and scaling
    if metric_name == "prob_diffs":
        # For percentage-scaled probability differences
        y_linthresh = 1e-2  # 0.01% is a reasonable linear threshold
    elif metric_name == "odds_ratios":
        # For odds ratios, use threshold of 1.0 (parity line)
        y_linthresh = 1.0
    else:
        # For logit differences and other metrics
        y_linthresh = 1e-6
    
    if log_scale_both:
        # Log scale on both axes - use symlog for both to handle negative values
        ax.set_xscale('symlog', linthresh=1.0)  # Linear region around zero for x-axis
        ax.set_yscale('symlog', linthresh=y_linthresh)  # Use appropriate threshold for y-axis
        # Add grid for log scale
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.grid(True, which="minor", ls=":", alpha=0.2)
    elif use_log_scale:
        # Only y-axis log scale
        ax.set_yscale('symlog', linthresh=y_linthresh)

    # ================ Y-AXIS LIMITS ================
    ax.set_ylim(scaled_y_limits[0], scaled_y_limits[1])
    ax.legend(frameon=True, fontsize=11)
    ax.tick_params(axis="both", which="major", labelsize=12)


    # ================ PLOT, SAVE, and CLOSE FIGURE ================
    plt.tight_layout()
    plt.savefig(get_fig_path(out_dir, dataset_name, model_name, metric_name, inj_layer, read_layer, read_site, inject_site), dpi=300, bbox_inches="tight")
    plt.close(fig)


def get_ylabel(metric_name: str) -> str:
    if metric_name == "logit_diffs":
        ylabel = r"$\Delta_{\mathrm{logit}} = \mathrm{logit}(Choice1) - \mathrm{logit}(Choice2)$"
    elif metric_name == "prob_diffs":
        ylabel = r"$\Delta_{\mathrm{prob}} = P(Choice1) - P(Choice2)$"
    elif metric_name == "odds_ratios":
        ylabel = r"Odds Ratio ($e^{\Delta}$)"
    elif metric_name == "rank_changes":
        ylabel = r"Token Rank"
    elif metric_name == "kl_divergences":
        ylabel = r"KL Divergence"
    elif metric_name == "compute_perplexity":
        ylabel = r"Perplexity"
    else:
        ylabel = f"Metric: {metric_name}"
    return ylabel

def get_fig_path(out_dir, dataset_name, model_name, metric_name, inj_layer, read_layer, read_site, inject_site) -> str:
     # Create directory structure: out_dir/dataset/model/metric_name/injL{inj_layer}/
    # Clean model name for filesystem (remove path separators)
    clean_model_name = model_name.replace("/", "_").replace("\\", "_")
    out_dir_dataset = os.path.join(out_dir, dataset_name)
    out_dir_model = os.path.join(out_dir_dataset, clean_model_name)
    out_dir_metric = os.path.join(out_dir_model, metric_name)
    out_dir_layer = os.path.join(out_dir_metric, f"injL{inj_layer}")

    os.makedirs(out_dir_layer, exist_ok=True)
    fig_path = os.path.join(out_dir_layer, f"brc_{metric_name}_injL{inj_layer}_{inject_site}_readL{read_layer}_{read_site}.png")
    return fig_path


def plot_rank_changes_and_save(
    bias_diffs: Sequence[float],
    random_diffs: Sequence[float], 
    orth_diffs: Sequence[float],
    alpha_values: Iterable[float],
    inj_layer: int,
    read_layer: int,
    inject_site: str,
    read_site: str,
    out_dir: str,
    y_limits: tuple[float, float],
    dataset_name: str = "reassurance",
    model_name: str = "gpt2",
) -> None:
    """
    Plots and saves rank change curves with dual overlaid lines and inverted y-axis.
    
    Note: For rank_changes, bias_diffs contains (choice1_ranks, choice2_ranks) tuple. #TODO: this is a hack and should be removed
    We only use bias vector data since random/orth aren't meaningful for rank analysis.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6))
    
    alpha_list = list(alpha_values)
    
    # Extract rank data from bias vector (only vector that matters for rank analysis)
    choice1_ranks, choice2_ranks = bias_diffs
    
    # Plot dual lines for choice1 and choice2 ranks (match BRC curve style)
    ax.plot(alpha_list, choice1_ranks, label="Choice 1", color="#0072B2", linewidth=2.5, marker="o", markersize=3)
    ax.plot(alpha_list, choice2_ranks, label="Choice 2", color="#D55E00", linewidth=2.0, marker="o", markersize=3)
    
    # Find and annotate crossing point (where choice1 overtakes choice2)
    for i, (r1, r2) in enumerate(zip(choice1_ranks, choice2_ranks)):
        if i > 0:  # Need at least 2 points to detect crossing
            prev_r1, prev_r2 = choice1_ranks[i-1], choice2_ranks[i-1]
            # Check if lines crossed (r1 was worse but now better, or vice versa)
            if (prev_r1 > prev_r2 and r1 < r2) or (prev_r1 < prev_r2 and r1 > r2):
                crossing_alpha = alpha_list[i]
                crossing_rank = min(r1, r2)  # Use the better rank for annotation
                ax.axvline(crossing_alpha, color="red", linestyle="--", alpha=0.7, linewidth=1)
                ax.annotate(f"Crossing: α={crossing_alpha:.1f}", 
                           xy=(crossing_alpha, crossing_rank), 
                           xytext=(10, 10), textcoords="offset points",
                           fontsize=10, ha='left',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
                break
    
    # Set up inverted y-axis (rank 1 at top)
    ax.invert_yaxis()
    
    # Simple y-axis label indicating it's flipped
    ax.set_ylabel("Token Rank (flipped axis)", fontsize=14)
    
    # Use global y-limits passed from experiment
    ax.set_ylim(y_limits[0], y_limits[1])
    
    # Labels and formatting
    ax.set_xlabel(r"Steering coefficient $\alpha$", fontsize=14)
    ax.set_title(
        f"BRC (rank_changes) | inject: L{inj_layer}:{inject_site} → read: L{read_layer}:{read_site}",
        fontsize=15,
        weight="bold",
    )
    
    ax.legend(frameon=True, fontsize=11)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True, alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(get_fig_path(out_dir, dataset_name, model_name, "rank_changes", inj_layer, read_layer, read_site, inject_site), dpi=300, bbox_inches="tight")
    plt.close(fig)