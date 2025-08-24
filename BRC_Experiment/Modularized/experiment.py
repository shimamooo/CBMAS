from __future__ import annotations

from typing import Iterable, List, Sequence
import torch

from BRC_Experiment.Modularized.config import ExperimentConfig
from BRC_Experiment.Modularized.data import load_winogender_pairs, load_reassurance_pairs
from BRC_Experiment.Modularized.model import load_model, get_pronoun_token_ids, get_choice_token_ids
from BRC_Experiment.Modularized.plotting import plot_and_save_brc_curves
from BRC_Experiment.Modularized.steering import build_vectors, sweep_alpha, logit_diffs, prob_diffs, compute_perplexity
from BRC_Experiment.Modularized.utils import build_alpha_range, configure_determinism, get_device


class Experiment:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config 

        configure_determinism(self.config.seed) # Set seed for reproducibility
        self.device = get_device() # Get device for model
        self.model = load_model(self.config.model_name, self.device) # Load model
        
        # Load dataset and get appropriate token IDs based on dataset choice
        if self.config.dataset == "reassurance":
            self.prompt_pairs = load_reassurance_pairs()
            self.choice1_id, self.choice2_id = get_choice_token_ids(self.model)
        else:  # winogender
            self.prompt_pairs = load_winogender_pairs()
            self.choice1_id, self.choice2_id = get_pronoun_token_ids(self.model)

        # Build alpha range
        self.alpha_values = build_alpha_range(
            self.config.alpha_start, self.config.alpha_stop, self.config.alpha_step
        ) 

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
            return lambda logits: logit_diffs(logits, self.choice1_id, self.choice2_id)
        elif self.config.metric == "prob_diffs":
            return lambda logits: prob_diffs(logits, self.choice1_id, self.choice2_id)
        elif self.config.metric == "compute_perplexity":
            # For perplexity, we'll compute for first choice token by default
            return lambda logits: compute_perplexity(logits, self.choice1_id)
        else:
            raise ValueError(f"Unknown metric: {self.config.metric}")

    def run_experiment(self) -> None:
        # ====== PHASE 1: Accumulate all curves and compute global y-limits ======
        results: list[dict[str, object]] = [] # List of results for each injection layer and read layer
        global_min: float 
        global_max: float 
    
        metric_func = self._get_metric_function()
        
        print(f"=== DEBUG INFO ===")
        print(f"Dataset: {self.config.dataset}")
        print(f"Metric: {self.config.metric}")
        print(f"Choice1 ID: {self.choice1_id}")
        print(f"Choice2 ID: {self.choice2_id}")
        print(f"Prompt pairs length: {len(self.prompt_pairs)}")
        print(f"Alpha values: {self.alpha_values}")
        print(f"Inject layers: {self.inject_layers}")
        print(f"Read layers: {self.read_layers}")
        print(f"==================")


        for inj_layer in self.inject_layers:
            print(f"\n--- Processing injection layer {inj_layer} ---")
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

                print(f"  Processing read layer {read_layer}...")
                inject_hook = f"blocks.{inj_layer}.{self.config.inject_site}" # Get inject hook name TODO: maybe move to config and change to inject_hook_name
                read_hook = f"blocks.{read_layer}.{self.config.read_site}" # Get read hook name TODO: maybe move to config and change to read_hook_name

                # Factor out common parameters to avoid duplication
                common_args = {
                    'model': self.model,
                    'alpha_values': self.alpha_values,
                    'prompt': self.config.prefix,
                    'inj_layer': inj_layer,
                    'read_layer': read_layer,
                    'inject_hook_name': inject_hook,
                    'read_hook_name': read_hook,
                    'prepend_bos': self.config.prepend_bos,
                    'device': self.device,
                    'steer_all_tokens': self.config.steer_all_tokens,
                }
                
                # Get logits for all vector types
                logits_by_type = {}
                for vector_name in ["bias", "random", "orth"]:
                    logits_by_type[vector_name] = sweep_alpha(
                        vectors[vector_name],
                        **common_args
                    )
                
                bias_logits = logits_by_type["bias"]
                random_logits = logits_by_type["random"]
                orth_logits = logits_by_type["orth"]

                # Debug prints for logits
                print(f"    Bias logits length: {len(bias_logits) if bias_logits else 'None'}")
                print(f"    Random logits length: {len(random_logits) if random_logits else 'None'}")
                print(f"    Orth logits length: {len(orth_logits) if orth_logits else 'None'}")

                # Calculate differences using selected metric
                bias_diffs = metric_func(bias_logits)
                random_diffs = metric_func(random_logits)
                orth_diffs = metric_func(orth_logits)

                # Debug prints for metric results
                print(f"    Bias diffs: {bias_diffs}")
                print(f"    Random diffs: {random_diffs}")
                print(f"    Orth diffs: {orth_diffs}")

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

        # ====== PHASE 2: Plot with fixed y-limits for consistency across figures ======
        if global_max == global_min:
            pad = 0.1 if global_max == 0 else abs(global_max) * 0.1
            fixed_limits = (global_min - pad, global_max + pad)
        else:
            yr = global_max - global_min
            fixed_limits = (global_min - 0.1 * yr, global_max + 0.1 * yr)

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
                self.config.out_dir,
                fixed_y_limits=fixed_limits,
                metric_name=self.config.metric,
                use_log_scale=self.config.use_log_scale,
                dataset_name=self.config.dataset,
                model_name=self.config.model_name,
                log_scale_both=self.config.log_scale_both,
            )
            print("Saved:", fig_path)
    