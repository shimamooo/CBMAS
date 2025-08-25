"""Thin delegator with a quick-start example.

Usage examples:
  - python -m BRC_Experiment.Modularized.cli --help
  - python -m BRC_Experiment.Modularized.main --help
  - python BRC_Experiment/Modularized/main.py --help

If run without arguments, executes a small quick-start experiment for debugging.
"""

from BRC_Experiment.Modularized.cli import main as cli_main
from BRC_Experiment.Modularized.config import ExperimentConfig
from BRC_Experiment.Modularized.experiment import Experiment
import sys
from pathlib import Path

# Ensure project root is importable when running this file directly
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def run_quickstart() -> None:
    cfg = ExperimentConfig(
        model_name="gpt2-small",
        # Use the updated default prefix that matches our new format
        prepend_bos=True,
        inject_site="hook_resid_mid",
        read_site="hook_resid_post",
        alpha_start=-5.0,  # Smaller range for faster testing
        alpha_stop=5.0,
        alpha_step=2.5,    # Fewer steps for faster testing
        inject_layers=[2], # Single injection layer
        read_layers=None,  # All read layers (will use all 12 layers)
        seed=42,
        out_dir="graphs_debug_fixed",
        metric="logit_diffs",  # Start with logit_diffs to see stronger signals
        steer_all_tokens=True,
        use_log_scale=False,
        dataset="reassurance",  # Test our fixed reassurance dataset
        show_progress=True,  # Enable progress tracking
    )
    Experiment(cfg).run_experiment()


if __name__ == "__main__":
    # If arguments are provided, delegate to CLI; otherwise run a quick example
    if len(sys.argv) > 1:
        cli_main()
    else:
        run_quickstart()
