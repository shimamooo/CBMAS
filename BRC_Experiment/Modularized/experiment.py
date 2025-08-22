from __future__ import annotations

from typing import Iterable, List, Sequence
import torch

from BRC_Experiment.Modularized.config import ExperimentConfig
from BRC_Experiment.Modularized.data import load_winogender_pairs
from BRC_Experiment.Modularized.model import load_model, get_pronoun_token_ids
from BRC_Experiment.Modularized.plotting import plot_and_save_brc_curves
from BRC_Experiment.Modularized.steering import build_vectors, sweep_alpha, logit_diffs, prob_diffs, compute_perplexity
from BRC_Experiment.Modularized.utils import build_alpha_range, configure_determinism, get_device


class Experiment:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config 

        configure_determinism(self.config.seed) # Set seed for reproducibility
        self.device = get_device() # Get device for model

        self.model = load_model(self.config.model_name, self.device) # Load model
        self.he_id, self.she_id = get_pronoun_token_ids(self.model) # Get token ids for he and she #TODO: switch to A and B pairs

        self.alpha_values = build_alpha_range(
            self.config.alpha_start, self.config.alpha_stop, self.config.alpha_step
        ) # Build alpha range

        self.prompt_pairs = load_winogender_pairs()

        # Expand layer lists
        n_layers = self.model.cfg.n_layers # Get total number of layers in model
        self.inject_layers = (
            list(self.config.inject_layers) if self.config.inject_layers is not None else list(range(n_layers)) # If inject_layers is not specified, use all layers
        )
        self.read_layers = (
            list(self.config.read_layers) if self.config.read_layers is not None else list(range(n_layers)) # If read_layers is not specified, use all layers
        )

    def _get_metric_function(self):
        """Return the appropriate metric function based on config."""
        if self.config.metric == "logit_diffs":
            return lambda logits: logit_diffs(logits, self.he_id, self.she_id)
        elif self.config.metric == "prob_diffs":
            return lambda logits: prob_diffs(logits, self.he_id, self.she_id)
        elif self.config.metric == "compute_perplexity":
            # For perplexity, we'll compute for "he" token by default
            return lambda logits: compute_perplexity(logits, self.he_id)
        else:
            raise ValueError(f"Unknown metric: {self.config.metric}")

    def run_experiment(self) -> None:
        # Phase 1: accumulate all curves and compute global y-limits
        results: list[dict[str, object]] = []
        global_min: float | None = None
        global_max: float | None = None
        
        # Get the metric function to use
        metric_func = self._get_metric_function()
        
        # Automatically enable log scale for prob_diffs since values are very small
        use_log_scale = self.config.use_log_scale or (self.config.metric == "prob_diffs") #TODO: logic is messy, clean this up later

        for inj_layer in self.inject_layers:
            vectors = build_vectors(
                self.model,
                inj_layer,
                self.prompt_pairs,
                self.config.prepend_bos,
                self.device,
                inject_site=self.config.inject_site,
            ) # Build vectors for each inject_layer

            for read_layer in self.read_layers:
                if read_layer <= inj_layer:
                    continue # Skip if read_layer is less than or equal to inject_layer (impossible combo)

                inject_hook = f"blocks.{inj_layer}.{self.config.inject_site}" # Get inject hook name TODO: maybe move to config and change to inject_hook_name
                read_hook = f"blocks.{read_layer}.{self.config.read_site}" # Get read hook name TODO: maybe move to config and change to read_hook_name

                #TODO: no reasone to do this thrice lol, clean this up later
                bias_logits = sweep_alpha(
                    self.model,
                    vectors["bias"],
                    self.alpha_values,
                    self.config.prefix,
                    inj_layer,
                    read_layer,
                    inject_hook,
                    read_hook,
                    self.config.prepend_bos,
                    self.device,
                    self.config.steer_all_tokens,
                ) # get list of logits for each alpha value for bias vector

                random_logits = sweep_alpha(
                    self.model,
                    vectors["random"],
                    self.alpha_values,
                    self.config.prefix,
                    inj_layer,
                    read_layer,
                    inject_hook,
                    read_hook,
                    self.config.prepend_bos,
                    self.device,
                    self.config.steer_all_tokens,
                ) # get list of logits for each alpha value for random vector

                orth_logits = sweep_alpha(
                    self.model,
                    vectors["orth"],
                    self.alpha_values,
                    self.config.prefix,
                    inj_layer,
                    read_layer,
                    inject_hook,
                    read_hook,
                    self.config.prepend_bos,
                    self.device,
                    self.config.steer_all_tokens,
                ) # get list of logits for each alpha value for orth vector

                # Calculate differences using selected metric
                bias_diffs = metric_func(bias_logits)
                random_diffs = metric_func(random_logits)
                orth_diffs = metric_func(orth_logits)

                # Update global min/max
                for v in (*bias_diffs, *random_diffs, *orth_diffs):
                    global_min = v if global_min is None else min(global_min, v)
                    global_max = v if global_max is None else max(global_max, v)

                results.append(
                    {
                        "inj": inj_layer,
                        "read": read_layer,
                        "bias": bias_diffs,
                        "random": random_diffs,
                        "orth": orth_diffs,
                    }
                )

        # Determine fixed y-limits with a small pad
        if global_min is None or global_max is None:
            return  # nothing to plot
        if global_max == global_min:
            pad = 0.1 if global_max == 0 else abs(global_max) * 0.1
            fixed_limits = (global_min - pad, global_max + pad)
        else:
            yr = global_max - global_min
            fixed_limits = (global_min - 0.1 * yr, global_max + 0.1 * yr)

        # Phase 2: plot with fixed y-limits for consistency across figures
        for item in results:
            fig_path = plot_and_save_brc_curves(
                item["bias"],  # type: ignore[arg-type]
                item["random"],  # type: ignore[arg-type]
                item["orth"],  # type: ignore[arg-type]
                self.alpha_values,
                item["inj"],  # type: ignore[arg-type]
                item["read"],  # type: ignore[arg-type]
                self.config.inject_site,
                self.config.read_site,
                self.config.prepend_bos,
                self.config.prefix,
                self.config.out_dir,
                fixed_y_limits=fixed_limits,
                metric_name=self.config.metric,
                use_log_scale=use_log_scale,
            )
            print("Saved:", fig_path)
    