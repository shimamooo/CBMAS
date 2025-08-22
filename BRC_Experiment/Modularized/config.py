from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class ExperimentConfig:
    model_name: str = "gpt2-small"
    prepend_bos: bool = True
    prefix: str = "The doctor said that "

    inject_site: str = "hook_resid_mid"
    read_site: str = "hook_resid_post"

    alpha_start: float = -10.0
    alpha_stop: float = 10.0
    alpha_step: float = 0.5

    # If None, use all layers (0..n_layers-1)
    inject_layers: Optional[Sequence[int]] = None
    read_layers: Optional[Sequence[int]] = None

    seed: int = 42
    out_dir: str = "graphs"
    
    # Metric to use for calculating differences: logit_diffs, prob_diffs, or compute_perplexity
    metric: str = "logit_diffs"
    
    # Whether to steer all token positions or just the last token
    steer_all_tokens: bool = True
    
    # Whether to use log scale for plotting (useful for very small values)
    use_log_scale: bool = False
    
    # Dataset to use: "winogender" or "reassurance"
    dataset: str = "reassurance"


