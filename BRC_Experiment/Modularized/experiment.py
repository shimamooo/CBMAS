from __future__ import annotations
from BRC_Experiment.Modularized.config import ExperimentConfig
from BRC_Experiment.Modularized.data import load_winogender_pairs, load_reassurance_pairs
from BRC_Experiment.Modularized.model import load_model, get_pronoun_token_ids, get_choice_token_ids
from BRC_Experiment.Modularized.plotting import plot_and_save_brc_curves
from BRC_Experiment.Modularized.vectors import build_vectors
from BRC_Experiment.Modularized.steering import sweep_alpha
from BRC_Experiment.Modularized.metrics import logit_diffs, prob_diffs, compute_perplexity, odds_ratios, rank_changes
from BRC_Experiment.Modularized.utils import build_alpha_range, configure_determinism, get_device, build_hook_name
from BRC_Experiment.Modularized.observability import create_progress_tracker


class Experiment:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config 
        self.progress_tracker = create_progress_tracker(enabled=self.config.show_progress)
        configure_determinism(self.config.seed) # Set seed for reproducibility
        self.device = get_device() # Get device for model
        self.model = load_model(self.config.model_name, self.device, self.progress_tracker) # Load model with progress tracking
        
        # Load dataset and get appropriate token IDs based on dataset choice
        if self.config.dataset == "reassurance":
            self.prompt_pairs = load_reassurance_pairs()
            self.choice1_id, self.choice2_id = get_choice_token_ids(self.model)
        else:  # winogender
            self.prompt_pairs = load_winogender_pairs()
            self.choice1_id, self.choice2_id = get_pronoun_token_ids(self.model) #TODO: consolidate to one data loading function

        # Build alpha range
        self.alpha_values = build_alpha_range(
            self.config.alpha_start, self.config.alpha_stop, self.config.alpha_step
        ) 

        # Get sequence of inject and read layers
        n_layers = self.model.cfg.n_layers # Get total number of layers in model
        self.inject_layers = (
            list(self.config.inject_layers) if self.config.inject_layers is not None else list(range(n_layers)) # If inject_layers is not specified, use all layers
        )
        self.read_layers = (
            list(self.config.read_layers) if self.config.read_layers is not None else list(range(n_layers)) # If read_layers is not specified, use all layers
        )

    def _get_all_metric_functions(self):
        """Return all available metric functions."""
        return {
            "logit_diffs": lambda logits: logit_diffs(logits, self.choice1_id, self.choice2_id),
            "prob_diffs": lambda logits: prob_diffs(logits, self.choice1_id, self.choice2_id),
            "odds_ratios": lambda logits: odds_ratios(logits, self.choice1_id, self.choice2_id),
            "compute_perplexity": lambda logits: compute_perplexity(logits, self.choice1_id),
            "rank_changes": lambda logits: rank_changes(logits, self.choice1_id, self.choice2_id),
        }
    
    def _get_metrics_to_run(self):
        """Determine which metrics to run based on config."""
        all_metrics = self._get_all_metric_functions()
        if self.config.metric is None:
            return all_metrics
        else:
            if self.config.metric not in all_metrics:
                raise ValueError(f"Unknown metric: {self.config.metric}")
            return {self.config.metric: all_metrics[self.config.metric]}
    #TODO: Above do basically the same thing as below, so consolidate
    

    def run_experiment(self) -> None:
        # ====== PHASE 1: Setup and determine metrics to run ======
        metrics_to_run = self._get_metrics_to_run()
        progress_tracker = self.progress_tracker
        
        # ====== PHASE 2: Iterate through layer combinations and compute metrics ======
        # For rank_changes, collect global data and defer plotting
        all_rank_data = []
        rank_results = []
        for inj_layer in progress_tracker.track_injection_layers(self.inject_layers):

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
            
            for read_layer in progress_tracker.track_read_layers(valid_read_layers):
    
                inject_hook = build_hook_name(inj_layer, self.config.inject_site)
                read_hook = build_hook_name(read_layer, self.config.read_site)

                # ====== PHASE 2a: Compute steered logits (batch computation) ======
                # Get steered logits for all vector types using sweep_alpha (computed once, used for all metrics)
                logits_by_vec = {}
                for vector_name in progress_tracker.track_vector_types(["bias", "random", "orth"]):
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
                bias_logits, random_logits, orth_logits = logits_by_vec["bias"], logits_by_vec["random"], logits_by_vec["orth"]

                # ====== PHASE 2b: Compute and plot each metric separately ======
                # Process metrics with per-metric y-limits TODO: possibly move this to plotting.py
                for metric_name, metric_func in metrics_to_run.items():
                    bias_diffs = metric_func(bias_logits)
                    random_diffs = metric_func(random_logits)
                    orth_diffs = metric_func(orth_logits)
                    
                    # Collect rank data for global y-limits if needed
                    if metric_name == "rank_changes":
                        choice1_ranks, choice2_ranks = bias_diffs
                        all_rank_data.extend(choice1_ranks + choice2_ranks)
                    
                    # Store all results for plotting (including rank_changes)
                    rank_results.append((inj_layer, read_layer, bias_diffs, random_diffs, orth_diffs, metric_name))
        
        # ====== PHASE 3: Plot all results ======
        # Compute global rank limits if needed
        if "rank_changes" in metrics_to_run and all_rank_data:
            min_rank, max_rank = min(all_rank_data), max(all_rank_data)
            rank_range = max_rank - min_rank
            # Ensure padding doesn't make min_rank negative
            y_padding = min(rank_range * 0.1, min_rank - 1) if min_rank > 1 else 0
            global_rank_limits = (max_rank + y_padding, max(1, min_rank - y_padding))
        else:
            global_rank_limits = (1000, 1)
        
        # Plot all results with appropriate y-limits
        for inj_layer, read_layer, bias_diffs, random_diffs, orth_diffs, metric_name in rank_results:
            if metric_name == "rank_changes":
                y_limits = global_rank_limits
            else:
                # Compute per-metric y-limits for scalar metrics
                all_values = (*bias_diffs, *random_diffs, *orth_diffs)
                metric_min, metric_max = min(all_values), max(all_values)
                if metric_max == metric_min:
                    pad = 0.1 if metric_max == 0 else abs(metric_max) * 0.1
                    y_limits = (metric_min - pad, metric_max + pad)
                else:
                    yr = metric_max - metric_min
                    y_limits = (metric_min - 0.1 * yr, metric_max + 0.1 * yr)
            
            plot_and_save_brc_curves(
                bias_diffs=bias_diffs,
                random_diffs=random_diffs,
                orth_diffs=orth_diffs,
                alpha_values=self.alpha_values,
                inj_layer=inj_layer,
                read_layer=read_layer,
                inject_site=self.config.inject_site,
                read_site=self.config.read_site,
                out_dir=self.config.out_dir,
                y_limits=y_limits,
                metric_name=metric_name,
                use_log_scale=self.config.use_log_scale,
                dataset_name=self.config.dataset,
                model_name=self.config.model_name,
                log_scale_both=self.config.log_scale_both,
            )
