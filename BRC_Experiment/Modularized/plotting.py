from typing import Iterable, Sequence
import os

import matplotlib.pyplot as plt


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
    fixed_y_limits: tuple[float, float],
    metric_name: str = "logit_diffs",
    use_log_scale: bool = False,
    dataset_name: str = "reassurance",
    model_name: str = "gpt2",
    log_scale_both: bool = False,  # Add parameter for log scale on both axes
) -> str:
    """
    Plots and saves BRC curves for bias, random, and orth vectors.
    """
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
        scaled_y_limits = (fixed_y_limits[0] * 100, fixed_y_limits[1] * 100)
    else:
        series = {
            "bias": list(bias_diffs),
            "random": list(random_diffs),
            "orth": list(orth_diffs),
        }
        scaled_y_limits = fixed_y_limits

    alpha_list = list(alpha_values)

    # ================ PLOTTING ================
    for name in ["bias", "random", "orth"]:
        color, lw = colors[name]
        ax.plot(alpha_list, series[name], label=name, color=color, linewidth=lw, marker="o", markersize=3)

    ax.axhline(0, color="black", linestyle="--", linewidth=1)

    # ================ LABELS AND TITLE ================
    ax.set_xlabel(r"Steering coefficient $\alpha$", fontsize=14)
    
    # Dynamic y-axis label based on metric
    if metric_name == "logit_diffs":
        ylabel = r"$\Delta_{\mathrm{logit}} = \mathrm{logit}(Choice1) - \mathrm{logit}(Choice2)$"
    elif metric_name == "prob_diffs":
        ylabel = r"$\Delta_{\mathrm{prob}} = P(Choice1) - P(Choice2)$ (%)"
    elif metric_name == "compute_perplexity":
        ylabel = r"Perplexity"
    else:
        ylabel = f"Metric: {metric_name}"
    
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
    
    # Determine log scale behavior
    # Default to True for prob_diffs (values are very small), False for others
    default_log_scale = metric_name == "prob_diffs"
    effective_log_scale = use_log_scale if use_log_scale is not None else default_log_scale
    
    # Apply log scale based on parameters
    # Choose appropriate linthresh based on metric and scaling
    if metric_name == "prob_diffs":
        # For percentage-scaled probability differences
        y_linthresh = 1e-2  # 0.01% is a reasonable linear threshold
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
    elif effective_log_scale:
        # Only y-axis log scale
        ax.set_yscale('symlog', linthresh=y_linthresh)

    # ================ Y-AXIS LIMITS ================
    ax.set_ylim(scaled_y_limits[0], scaled_y_limits[1])
    ax.legend(frameon=True, fontsize=11)
    ax.tick_params(axis="both", which="major", labelsize=12)

    plt.tight_layout()

    # ================ SAVE FIGURE ================
    # Create directory structure: out_dir/dataset/model/metric_name/injL{inj_layer}/
    # Clean model name for filesystem (remove path separators)
    clean_model_name = model_name.replace("/", "_").replace("\\", "_")
    
    out_dir_dataset = os.path.join(out_dir, dataset_name)
    out_dir_model = os.path.join(out_dir_dataset, clean_model_name)
    out_dir_metric = os.path.join(out_dir_model, metric_name)
    out_dir_layer = os.path.join(out_dir_metric, f"injL{inj_layer}")
    os.makedirs(out_dir_layer, exist_ok=True)
    fig_path = os.path.join(out_dir_layer, f"brc_{metric_name}_injL{inj_layer}_{inject_site}_readL{read_layer}_{read_site}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig_path


