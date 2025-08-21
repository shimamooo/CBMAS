import argparse
from typing import Optional

from BRC_Experiment.Modularized.config import ExperimentConfig
from BRC_Experiment.Modularized.experiment import Experiment
from BRC_Experiment.Modularized.utils import parse_layer_spec


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run BRC experiment with configurable hyperparameters.")
    p.add_argument("--model-name", type=str, default="gpt2-small", help="Transformer-Lens model name")
    p.add_argument("--prefix", type=str, default="The doctor said that ", help="Prompt prefix to steer")
    p.add_argument("--prepend-bos", action="store_true", help="Prepend BOS token during tokenization")
    p.add_argument("--no-prepend-bos", dest="prepend_bos", action="store_false", help="Do not prepend BOS token")
    p.set_defaults(prepend_bos=True)

    p.add_argument("--inject-site", type=str, default="hook_resid_mid", help="Injection hook site name")
    p.add_argument("--read-site", type=str, default="hook_resid_post", help="Readout hook site name")

    p.add_argument("--alpha-start", type=float, default=-10.0)
    p.add_argument("--alpha-stop", type=float, default=10.0)
    p.add_argument("--alpha-step", type=float, default=0.5)

    p.add_argument("--inject-layers", type=str, default="all", help='Layer spec like "all", "0,2,4" or "3-8"')
    p.add_argument("--read-layers", type=str, default="all", help='Layer spec like "all", "1,3,5" or "2-10"')

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="graphs")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    config = ExperimentConfig(
        model_name=args.model_name,
        prepend_bos=args.prepend_bos,
        prefix=args.prefix,
        inject_site=args.inject_site,
        read_site=args.read_site,
        alpha_start=args.alpha_start,
        alpha_stop=args.alpha_stop,
        alpha_step=args.alpha_step,
        inject_layers=parse_layer_spec(args.inject_layers),
        read_layers=parse_layer_spec(args.read_layers),
        seed=args.seed,
        out_dir=args.out_dir,
    )

    Experiment(config).run_experiment()


if __name__ == "__main__":
    main()


