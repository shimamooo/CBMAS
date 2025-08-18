# bias-probing

## Bias Response Curve (BRC) experiment

This repository contains a Bias Response Curve (BRC) experiment that measures how adding a gender steering vector to a GPT-2 Small model’s residual stream affects next-token preferences between the tokens " he" and " she".

Primary implementation lives in the notebook `BRC/BRC.ipynb`.

### What the experiment measures

- **Primary metric (Δ_logit)**: `logit(" he") − logit(" she")` at the decision position \(t\*\), computed via a clean logit lens (apply `ln_final` then `unembed` on a captured residual state).
- **Bias Response Curve**: Sweep a scalar steering coefficient **α** and inject `α · v_bias` into the residual stream at a chosen layer/hook site; record Δ_logit across α to see a dose–response curve.
- **Slopes near α ≈ 0**: Fit a line to the four points with smallest |α| to estimate sensitivity. Expect a non-zero slope for the bias direction and ~0 for controls.
- **Controls**: Compare the bias direction against a random unit vector and its component orthogonal to the bias direction.

### Core setup

- **Model/tooling**: GPT-2 Small via TransformerLens (`HookedTransformer`).
- **Decision position (t\*)**: The next-token position immediately after the input prefix; BOS handling is explicit and consistent.
- **Tokenization details**: Pronoun tokens use a leading space (" he", " she") for tokenizer consistency. BOS can be prepended via `PREPEND_BOS`.
- **Injection/Read sites**:
  - Inject a steering vector at `blocks.{INJECT_LAYER}.{INJECT_SITE}` (default: `L3:hook_resid_mid`).
  - Read logits from `blocks.{READ_LAYER}.{READ_SITE}` via a clean logit lens (default: `L8:hook_resid_post`).
- **Bias direction (v_bias)**: Contrastive residuals at the pronoun position across a list of neutral prefixes, averaged and unit-normalized. Orientation is flipped so that increasing α increases Δ_logit.
- **Alpha sweep**: Default grid `[-1.0, …, 0.9]` in steps of `0.1` (including a near-zero value), applied at the decision position only; model weights are not changed.

### Outputs

- **Single-setup curve**: Saves a figure like `brc_gpt2s_injL{INJECT_LAYER}_readL{READ_LAYER}_tstar.png` showing Δ_logit vs α for `bias`, `random`, and `orth` directions.
- **Layer grid sweep (optional)**: Iterates over multiple inject/read layer combinations and saves plots under `figs_brc_layergrid/` with per-direction slopes near α ≈ 0 printed to stdout.

### Reproducing the experiment

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   The notebook also installs `transformer_lens` if missing.
2. Open and run `BRC/BRC.ipynb` end-to-end. A CUDA GPU is optional but recommended. The notebook sets seeds and uses deterministic settings where possible.

### Important implementation notes

- **Clean logit lens vs model head**: The primary measurements use the clean logit lens at the configured read site. A separate sanity check compares this to the model head at the final layer; they only match when reading from the final layer.
- **Prefix policy**: Prefixes in `PREFIX_LIST` are short neutral contexts (e.g., "The doctor said that "). The pronoun occurs immediately after the prefix; the code computes t\* locally per input.
- **Determinism**: Seeds are set (`numpy`, `torch`), `CUBLAS_WORKSPACE_CONFIG` is configured, and deterministic algorithms are requested. Exact reproducibility can depend on hardware and library versions.

### Customize the study

You can tweak the following in `BRC/BRC.ipynb`:

- `PREPEND_BOS`, `prefix`, and `PREFIX_LIST` (prompt policy)
- `INJECT_LAYER`, `READ_LAYER`, `INJECT_SITE` (e.g., `hook_resid_mid`), `READ_SITE` (e.g., `hook_resid_post`)
- `alphas` (α sweep grid)
- Add additional control directions or swap in different contrastive pairs

### Repository layout

- `BRC/BRC.ipynb`: Main experiment notebook (includes a glossary of key terms near the top).
- `BRC/Version0/`: Earlier iteration artifacts and figures.

### Example observed values (from default settings shown in the notebook)

- Near-zero slopes (inject L3 mid, read L8 post): `bias ≈ +0.024`, `random ≈ −0.005`, `orth ≈ −0.005`.
- Ranges differ across prefixes and layers; see saved figures and printed slope summaries for specifics.
