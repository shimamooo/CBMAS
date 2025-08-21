# BRC Experiment (Modularized)

A modular, production-style refactor of a Bias-Repelling Control (BRC) experiment built on TransformerLens and GPT-2 small. It loads Winogender prompts, constructs steering vectors, sweeps steering strengths (alpha), and plots logit differences for He vs She under a clean logit lens.

## Features
- Clean module boundaries: config, data, model, steering, plotting, experiment, CLI
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

## Quick Start
Run a small, fast example (layers 0→1, alphas -1,0) to verify setup:
```bash
python BRC_Experiment/Modularized/main.py
```
Outputs are saved under `graphs_debug/`.

## Full CLI Usage
Show help:
```bash
python -m BRC_Experiment.Modularized.cli --help
```
Run with defaults:
```bash
python -m BRC_Experiment.Modularized.cli
```
Custom hyperparameters:
```bash
python -m BRC_Experiment.Modularized.cli \
  --model-name gpt2-small \
  --prefix "The doctor said that " \
  --prepend-bos \
  --inject-site hook_resid_mid \
  --read-site hook_resid_post \
  --alpha-start -10 --alpha-stop 10 --alpha-step 0.5 \
  --inject-layers 0-4 \
  --read-layers 1-6 \
  --seed 42 \
  --out-dir graphs
```
Layer specs for `--inject-layers` and `--read-layers`:
- `all` (default)
- Comma list: `0,2,5`
- Range: `3-8` (start inclusive, end exclusive)

## Notes
- The dataset `oskarvanderwal/winogender` is pulled automatically via `datasets`. Internet access is required the first time.
- Figures are saved as PNG: `graphs/injL{inj}/brc_injL{inj}_{inject_site}_readL{read}_{read_site}.png`.
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
