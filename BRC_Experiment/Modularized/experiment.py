from __future__ import annotations

from typing import Iterable, List, Sequence
import torch

from BRC_Experiment.Modularized.config import ExperimentConfig
from BRC_Experiment.Modularized.data import load_winogender_pairs
from BRC_Experiment.Modularized.model import load_model, get_pronoun_token_ids
from BRC_Experiment.Modularized.plotting import plot_and_save_brc_curves
from BRC_Experiment.Modularized.steering import build_vectors, sweep_alpha, logit_diffs
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

    def run_experiment(self) -> None:
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
                ) # get list of logits for each alpha value for orth vector

                #TODO: dedicate code to calculate different metrics for each vector
                bias_diffs = logit_diffs(bias_logits, self.he_id, self.she_id)
                random_diffs = logit_diffs(random_logits, self.he_id, self.she_id)
                orth_diffs = logit_diffs(orth_logits, self.he_id, self.she_id)

                fig_path = plot_and_save_brc_curves(
                    bias_diffs,
                    random_diffs,
                    orth_diffs,
                    self.alpha_values,
                    inj_layer,
                    read_layer,
                    self.config.inject_site,
                    self.config.read_site,
                    self.config.prepend_bos,
                    self.config.prefix,
                    self.config.out_dir,
                ) # plot and save BRC curves
                print("Saved:", fig_path) # print path to saved figure
    