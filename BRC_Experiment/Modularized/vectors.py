"""Vector extraction and construction for bias response curves."""

from typing import List
import torch
from transformer_lens import HookedTransformer

from BRC_Experiment.Modularized.utils import unit_vector
from BRC_Experiment.Modularized.cache import VectorCache


@torch.no_grad()
def residual_at_last_token(model: HookedTransformer, prompt: str, layer: int, site: str, prepend_bos: bool, device: torch.device) -> torch.Tensor:
    """Extract the residual activation at the last token position from a specific layer and site."""
    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos).to(device)
    last_idx = tokens.shape[1] - 1
    cache: dict[str, torch.Tensor] = {}

    def grab(activation: torch.Tensor, hook) -> torch.Tensor:  # type: ignore[no-redef]
        cache["resid"] = activation.detach()
        return activation

    _ = model.run_with_hooks(tokens, return_type=None, stop_at_layer=layer + 1, fwd_hooks=[(f"blocks.{layer}.{site}", grab)])
    return cache["resid"][0, last_idx, :].clone().to(device)


def build_vectors(
    model: HookedTransformer,
    inj_layer: int,
    prompt_pairs: List[tuple[str, str]],
    prepend_bos: bool,
    device: torch.device,
    inject_site: str,
    model_name: str = "",
    dataset: str = "",
) -> dict[str, torch.Tensor]:
    """
    Build bias, random, and orthogonal steering vectors.
    
    Uses cache-aside pattern: always check cache first, compute and cache if missing,
    then return the cached result.
    
    Args:
        model: The transformer model
        inj_layer: Layer to extract activations from
        prompt_pairs: List of (positive, negative) prompt pairs
        prepend_bos: Whether to prepend BOS token
        device: Device to run computations on
        inject_site: Hook site name (e.g., 'hook_resid_mid')
        model_name: Name of the model (for caching)
        dataset: Name of the dataset (for caching)
    
    Returns:
        Dictionary with 'bias', 'random', and 'orth' unit vectors
    """
    # Initialize cache
    cache = VectorCache()
    
    # Always try to load from cache first
    if model_name and dataset:
        cached_vectors = cache.load(model_name, dataset, inj_layer, inject_site, device)
        if cached_vectors is not None:
            print(f"USING CACHED VECTORS FOR {model_name}_{dataset}_layer{inj_layer}_{inject_site}")
            return cached_vectors
    
    # Compute vectors if not cached
    print(f"Computing vectors for {model_name}_{dataset}_layer{inj_layer}_{inject_site}")
    bias_vec = torch.stack([
        residual_at_last_token(model, p_f, inj_layer, inject_site, prepend_bos, device) - residual_at_last_token(model, p_m, inj_layer, inject_site, prepend_bos, device)
        for (p_f, p_m) in prompt_pairs
    ]).mean(dim=0) # Average difference between all pairs

    bias_vec = unit_vector(bias_vec)
    rand_vec = unit_vector(torch.randn_like(bias_vec))
    orth_seed = torch.randn_like(bias_vec)
    orth_vec = unit_vector(orth_seed - (orth_seed @ bias_vec) * bias_vec) # Orthogonal to bias vector
    
    vectors = {"bias": bias_vec, "random": rand_vec, "orth": orth_vec}
    
    # Always save to cache after computing
    if model_name and dataset:
        cache.save(vectors, model_name, dataset, inj_layer, inject_site)
    
    return vectors
