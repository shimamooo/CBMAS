"""Thin delegator with a quick-start example.

Usage examples:
  - python -m BRC_Experiment.Modularized.cli --help
  - python -m BRC_Experiment.Modularized.main --help
  - python BRC_Experiment/Modularized/main.py --help

If run without arguments, executes a small quick-start experiment for debugging.
"""

from BRC_Experiment.Modularized.experiment import Experiment
from BRC_Experiment.Modularized.config import ExperimentConfig
from BRC_Experiment.Modularized.cli import main as cli_main
import sys
from pathlib import Path

# Ensure project root is importable when running this file directly
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def run_quickstart() -> None:
    cfg = ExperimentConfig(
        model_name="gpt2-small",
        prefix="The doctor said that ",
        prepend_bos=True,
        inject_site="hook_resid_mid",
        read_site="hook_resid_post",
        alpha_start=-10.0,
        alpha_stop=10.0,
        alpha_step=0.5,
        inject_layers=[0],
        read_layers=[9, 10, 11],
        seed=42,
        out_dir="graphs_debug",
    )
    Experiment(cfg).run_experiment()


if __name__ == "__main__":
    # If arguments are provided, delegate to CLI; otherwise run a quick example
    if len(sys.argv) > 1:
        cli_main()
    else:
        run_quickstart()
