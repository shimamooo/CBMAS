# BRC Experiment (Modularized)

A modular, production-style refactor of a Bias-Repelling Control (BRC) experiment built on TransformerLens and GPT-2 small. It loads Winogender prompts, constructs steering vectors, sweeps steering strengths (alpha), and plots differences for He vs She using configurable metrics (logit differences, probability differences, or perplexity).

## Features
- Clean module boundaries: config, data, model, steering, plotting, experiment, CLI
- **Multiple metrics**: Choose between `logit_diffs`, `prob_diffs`, or `compute_perplexity`
- Deterministic runs when possible (cuBLAS/CUDNN settings and seeds)
- CLI to run experiments with configurable hyperparameters
- Quick-start example in `main.py` for fast debugging
- Saved figures per injection/read layer under `graphs/`

## Installation
1. Python 3.10+ recommended
2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
If you need CUDA-enabled PyTorch, adjust the torch/torchvision/torchaudio wheels as per your system.

## Project Structure
```
BRC_Experiment/
  Modularized/
    __init__.py
    config.py          # ExperimentConfig dataclass
    utils.py           # device, determinism, alpha grid, layer parsing
    data.py            # Winogender loader and prompt pairs
    model.py           # HookedTransformer loader and pronoun token ids
    steering.py        # residual capture, vector building, steering sweep
    plotting.py        # plot_and_save_brc_curves
    experiment.py      # Experiment orchestration
    cli.py             # CLI entrypoint
    main.py            # Quick-start runner or delegate to CLI
```

## Quick Start (no install)
We ship `main.py` with a built-in import shim, so you can run directly without installing the package:
```bash
python BRC_Experiment/Modularized/main.py
```
This runs a small, fast example (layers 0→1, alphas -1,0) and saves outputs to `graphs_debug/`.

## Testing Different Metrics

Test each metric with a small 3-layer setup:

**Logit differences:**
```bash
python -c "
from BRC_Experiment.Modularized.config import ExperimentConfig
from BRC_Experiment.Modularized.experiment import Experiment
cfg = ExperimentConfig(inject_layers=[0,1], read_layers=[2,3,4], alpha_start=-5, alpha_stop=5, alpha_step=2.5, out_dir='test_logit', metric='logit_diffs')
Experiment(cfg).run_experiment()
"
```

**Probability differences:**
```bash
python -c "
from BRC_Experiment.Modularized.config import ExperimentConfig
from BRC_Experiment.Modularized.experiment import Experiment
cfg = ExperimentConfig(inject_layers=[0,1], read_layers=[2,3,4], alpha_start=-5, alpha_stop=5, alpha_step=2.5, out_dir='test_prob', metric='prob_diffs')
Experiment(cfg).run_experiment()
"
```

**Perplexity:**
```bash
python -c "
from BRC_Experiment.Modularized.config import ExperimentConfig
from BRC_Experiment.Modularized.experiment import Experiment
cfg = ExperimentConfig(inject_layers=[0,1], read_layers=[2,3,4], alpha_start=-5, alpha_stop=5, alpha_step=2.5, out_dir='test_perplexity', metric='compute_perplexity')
Experiment(cfg).run_experiment()
"
```

## CLI Usage

**Basic Usage:**
```bash
# Show help
python -m BRC_Experiment.Modularized.cli --help

# Run with defaults
python -m BRC_Experiment.Modularized.cli

# Alternative without package install
PYTHONPATH=$(pwd) python -m BRC_Experiment.Modularized.cli
```

**Common Options:**
```bash
python -m BRC_Experiment.Modularized.cli \
  --model-name gpt2-small \
  --prefix "The doctor said that " \
  --metric logit_diffs \
  --alpha-start -10 --alpha-stop 10 --alpha-step 0.5 \
  --inject-layers 0-4 \
  --read-layers 1-6 \
  --out-dir graphs
```

**Available Metrics:**
- `logit_diffs` (default): Raw logit differences He - She
- `prob_diffs`: Probability differences P(He) - P(She) (auto-scaled to % and log scale)
- `compute_perplexity`: Perplexity for target token

**Layer Specifications:**
- `all` (default): All layers 0 to n_layers-1
- Comma-separated: `0,2,5`
- Range: `3-8` (start inclusive, end exclusive)

## Notes
- The dataset `oskarvanderwal/winogender` is pulled automatically via `datasets`. Internet access is required the first time.
- Figures are saved as PNG: `graphs/{metric}/injL{inj}/brc_{metric}_injL{inj}_{inject_site}_readL{read}_{read_site}.png`.
- Determinism is best-effort due to CUDA/BLAS constraints.

## Testing
A pytest suite can be created to mock heavy dependencies. Example categories:
- utils: seeds, device, alpha grid, layer parsing
- data: dataset loader mocked
- model: TransformerLens loader mocked
- steering: fake model for sweeps
- plotting: file creation
- experiment: orchestration with monkeypatched components
- cli: argument plumbing

## License
MIT
