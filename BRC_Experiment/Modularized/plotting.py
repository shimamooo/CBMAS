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
) -> str:
    """
    Plots and saves BRC curves for bias, random, and orth vectors.
    """
    colors = {
        "bias": ("#0072B2", 2.5),
        "random": ("#D55E00", 2.0),
        "orth": ("#009E73", 2.0),
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6))

    series = {
        "bias": list(bias_diffs),
        "random": list(random_diffs),
        "orth": list(orth_diffs),
    }

    alpha_list = list(alpha_values)

    for name in ["bias", "random", "orth"]:
        color, lw = colors[name]
        ax.plot(alpha_list, series[name], label=name, color=color, linewidth=lw, marker="o", markersize=3)

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Steering coefficient $\alpha$", fontsize=14)
    ax.set_ylabel(r"$\Delta_{\mathrm{logit}} = \mathrm{logit}(He) - \mathrm{logit}(She)$", fontsize=14)
    ax.set_title(
        f"BRC | inject: L{inj_layer}:{inject_site} → read: L{read_layer}:{read_site}",
        fontsize=15,
        weight="bold",
    )
    ax.legend(frameon=True, fontsize=11)
    ax.tick_params(axis="both", which="major", labelsize=12)

    note = f"BOS={prepend_bos}, prefix='{prefix}'"
    ax.text(0.01, -0.14, note, transform=ax.transAxes, fontsize=10, color="gray")

    yvals = series["bias"] + series["random"] + series["orth"]
    if yvals:
        ymin, ymax = min(yvals), max(yvals)
        if ymax == ymin:
            pad = 0.1 if ymax == 0 else abs(ymax) * 0.1
            ax.set_ylim(ymin - pad, ymax + pad)
        else:
            yr = ymax - ymin
            ax.set_ylim(ymin - 0.1 * yr, ymax + 0.1 * yr)

    plt.tight_layout()

    out_dir_layer = os.path.join(out_dir, f"injL{inj_layer}")
    os.makedirs(out_dir_layer, exist_ok=True)
    fig_path = os.path.join(out_dir_layer, f"brc_injL{inj_layer}_{inject_site}_readL{read_layer}_{read_site}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig_path


