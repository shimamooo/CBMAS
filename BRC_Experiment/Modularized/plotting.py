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
    prepend_bos: bool,
    prefix: str,
    out_dir: str,
    fixed_y_limits: tuple[float, float] | None = None,
    metric_name: str = "logit_diffs",
    use_log_scale: bool = False,
    dataset_name: str = "reassurance",
    model_name: str = "gpt2",
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
    else:
        series = {
            "bias": list(bias_diffs),
            "random": list(random_diffs),
            "orth": list(orth_diffs),
        }

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
    title_suffix = " (log scale)" if use_log_scale else ""
    ax.set_title(
        f"BRC ({metric_name}){title_suffix} | inject: L{inj_layer}:{inject_site} → read: L{read_layer}:{read_site}",
        fontsize=15,
        weight="bold",
    )
    
    # Apply log scale first if requested
    if use_log_scale:
        # Use symlog with a reasonable threshold for prob_diffs data
        ax.set_yscale('symlog', linthresh=1e-6)  # Linear region around zero

    # ================ Y-AXIS LIMITS ================
    if use_log_scale:
        # For log scale, let matplotlib auto-scale for better visualization of orders of magnitude
        pass  # Don't set manual limits with log scale
    else:
        # Normal scale: use fixed limits if provided, otherwise auto-calculate
        if fixed_y_limits is not None:
            ax.set_ylim(fixed_y_limits[0], fixed_y_limits[1])
        else:
            yvals = series["bias"] + series["random"] + series["orth"]
            if yvals:
                ymin, ymax = min(yvals), max(yvals)
                if ymax == ymin:
                    pad = 0.1 if ymax == 0 else abs(ymax) * 0.1
                    ax.set_ylim(ymin - pad, ymax + pad)
                else:
                    yr = ymax - ymin
                    ax.set_ylim(ymin - 0.1 * yr, ymax + 0.1 * yr)
    
    ax.legend(frameon=True, fontsize=11)
    ax.tick_params(axis="both", which="major", labelsize=12)

    # ================ ANNOTATIONS ================
    note = f"BOS={prepend_bos}, prefix='{prefix}'"
    ax.text(0.01, -0.14, note, transform=ax.transAxes, fontsize=10, color="gray")

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


