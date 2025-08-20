# bias-probing

## Bias Response Curve (BRC) experiment

This repository contains a Bias Response Curve (BRC) experiment that measures how adding a gender steering vector to a GPT-2 Small model's residual stream affects next-token preferences between the tokens " he" and " she".

**Primary implementations:**
- **Version 2 (Current)**: `BRC_Experiment/Version2/BRC_v2.ipynb` - Improved implementation with better structure and comprehensive layer sweeps
- **Version 1**: `BRC_Experiment/Version1/` - Intermediate iteration
- **Version 0**: `BRC_Experiment/Version0/` - Original implementation in `BRC/BRC.ipynb`

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

### Key improvements in Version 2

**BRC_v2** introduces several enhancements over the original implementation:

1. **Comprehensive Layer Sweeps**: Automatically sweeps across all injection/readout layer combinations, generating plots for each valid combination
2. **Improved File Organization**: Results are organized in a hierarchical directory structure (`graphs/injL{layer}/`) for better organization
3. **Enhanced Plotting**: Better visualization with improved styling, consistent color schemes, and informative titles
4. **Streamlined Workflow**: Cleaner function structure and better separation of concerns
5. **Automatic Vector Generation**: Bias, random, and orthogonal vectors are automatically generated for each injection layer
6. **Better Error Handling**: More robust implementation with proper device handling and tensor operations

### Outputs

- **Comprehensive layer sweep**: Automatically generates plots for all valid injection/readout layer combinations
- **Organized file structure**: Results saved under `graphs/injL{injection_layer}/` with descriptive filenames
- **Enhanced visualizations**: Professional-quality plots with consistent styling and informative annotations
- **Single-setup curves**: Each plot shows Δ_logit vs α for `bias`, `random`, and `orth` directions with clear labeling

### Reproducing the experiment

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   The notebook also installs `transformer_lens` if missing.

2. **For the latest implementation**: Open and run `BRC_Experiment/Version2/BRC_v2.ipynb` end-to-end. A CUDA GPU is optional but recommended.

3. **For the original implementation**: Use `BRC/BRC.ipynb` (Version 0).

Both notebooks set seeds and use deterministic settings where possible.

### Important implementation notes

- **Clean logit lens vs model head**: The primary measurements use the clean logit lens at the configured read site. A separate sanity check compares this to the model head at the final layer; they only match when reading from the final layer.
- **Prefix policy**: Prefixes in `PREFIX_LIST` are short neutral contexts (e.g., "The doctor said that "). The pronoun occurs immediately after the prefix; the code computes t\* locally per input.
- **Determinism**: Seeds are set (`numpy`, `torch`), `CUBLAS_WORKSPACE_CONFIG` is configured, and deterministic algorithms are requested. Exact reproducibility can depend on hardware and library versions.
- **Version 2 enhancements**: Automatically handles all layer combinations, provides better visualization, and organizes results systematically.

### Customize the study

You can tweak the following in `BRC_Experiment/Version2/BRC_v2.ipynb`:

- `PREPEND_BOS`, `prefix`, and `PREFIX_LIST` (prompt policy)
- `INJECT_LAYER`, `READ_LAYER`, `INJECT_SITE` (e.g., `hook_resid_mid`), `READ_SITE` (e.g., `hook_resid_post`)
- `ALPHA_RANGE` (α sweep grid)
- `inject_layers` and `read_layers` arrays to control which layer combinations to explore
- Add additional control directions or swap in different contrastive pairs

### Repository layout

- `BRC_Experiment/Version2/BRC_v2.ipynb`: **Current main experiment notebook** with comprehensive layer sweeps and improved visualization
- `BRC_Experiment/Version1/`: Intermediate iteration artifacts
- `BRC_Experiment/Version0/`: Earlier iteration artifacts and figures
- `BRC/BRC.ipynb`: Original experiment notebook (includes a glossary of key terms near the top)

### Example observed values (from default settings shown in the notebook)

- **Version 2**: Automatically generates results for all layer combinations, with plots saved in organized directories
- **Version 0**: Near-zero slopes (inject L3 mid, read L8 post): `bias ≈ +0.024`, `random ≈ −0.005`, `orth ≈ −0.005`.
- Ranges differ across prefixes and layers; see saved figures and printed slope summaries for specifics.

### Migration from Version 0 to Version 2

If you're familiar with the original implementation:
- **Core methodology**: Remains the same - the bias response curve concept and measurement approach are identical
- **New features**: Version 2 adds automatic layer sweeping, better organization, and enhanced visualization
- **File structure**: Results are now organized by injection layer in dedicated directories
- **Function names**: Most function names and interfaces remain the same for compatibility
