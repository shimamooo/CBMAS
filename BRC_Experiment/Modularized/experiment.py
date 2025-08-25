from __future__ import annotations
from BRC_Experiment.Modularized.config import ExperimentConfig
from BRC_Experiment.Modularized.data import load_winogender_pairs, load_reassurance_pairs
from BRC_Experiment.Modularized.model import load_model, get_pronoun_token_ids, get_choice_token_ids
from BRC_Experiment.Modularized.plotting import plot_and_save_brc_curves
from BRC_Experiment.Modularized.vectors import build_vectors
from BRC_Experiment.Modularized.steering import sweep_alpha
from BRC_Experiment.Modularized.metrics import logit_diffs, prob_diffs, compute_perplexity
from BRC_Experiment.Modularized.utils import build_alpha_range, configure_determinism, get_device, build_hook_name
from BRC_Experiment.Modularized.observability import create_progress_tracker


class Experiment:
    def __init__(self, config: ExperimentConfig) -> None:
        print("init config")
        self.config = config 
        print("init progress tracker")
        self.progress_tracker = create_progress_tracker(enabled=self.config.show_progress)
        print("determinism set")
        configure_determinism(self.config.seed) # Set seed for reproducibility
        print("get device")
        self.device = get_device() # Get device for model
        print("load model")
        self.model = load_model(self.config.model_name, self.device, self.progress_tracker) # Load model with progress tracking
        print("load dataset")
        
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
        global_min: float | None = None
        global_max: float | None = None 
    
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


        for inj_layer in self.progress_tracker.track_injection_layers(self.inject_layers):
            print(f"\n--- Processing injection layer {inj_layer} ---")
            vectors = build_vectors(
                self.model,
                inj_layer,
                self.prompt_pairs,
                self.config.prepend_bos,
                self.device,
                inject_site=self.config.inject_site,
            ) # Build vectors for each inject_layer

            # Filter read layers that are greater than injection layer
            valid_read_layers = [rl for rl in self.read_layers if rl > inj_layer]
            
            for read_layer in self.progress_tracker.track_read_layers(valid_read_layers):
                print(f"  Processing read layer {read_layer}...")
                inject_hook = build_hook_name(inj_layer, self.config.inject_site)
                read_hook = build_hook_name(read_layer, self.config.read_site)

                # Get steered logits for all vector types using sweep_alpha
                logits_by_vec = {}
                for vector_name in self.progress_tracker.track_vector_types(["bias", "random", "orth"]):
                    logits_by_vec[vector_name] = sweep_alpha(
                        self.model,
                        vectors[vector_name],
                        self.alpha_values,
                        self.config.prefix,
                        inj_layer,
                        read_layer,
                        inject_hook,
                        read_hook,
                        self.config.prepend_bos,
                        self.device,
                        self.config.steer_all_tokens,
                    )
                
                bias_logits = logits_by_vec["bias"]
                random_logits = logits_by_vec["random"]
                orth_logits = logits_by_vec["orth"]

                # Debug prints for logits
                print(f"    Bias logits length: {len(bias_logits) if bias_logits else 'None'}")
                print(f"    Random logits length: {len(random_logits) if random_logits else 'None'}")
                print(f"    Orth logits length: {len(orth_logits) if orth_logits else 'None'}")

                # Calculate differences using selected metric
                bias_diffs = metric_func(bias_logits)
                random_diffs = metric_func(random_logits)
                orth_diffs = metric_func(orth_logits)

                # Debug prints for metric results (first 5 values to check for NaNs)
                print(f"    Bias diffs (first 5): {bias_diffs[:5]}")
                print(f"    Random diffs (first 5): {random_diffs[:5]}")
                print(f"    Orth diffs (first 5): {orth_diffs[:5]}")

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

        for item in self.progress_tracker.track_plotting(results):
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
