from typing import Iterable, List

import torch
try:  # Optional import for typing only; tests use fakes
    from transformer_lens import HookedTransformer  # type: ignore
except Exception:  # pragma: no cover
    HookedTransformer = object  # type: ignore
#TODOL passing in the same parameters to different functions, probably first consolidate with passing experiment config object and make that more robust

@torch.no_grad()
def residual_at_last_token(model: HookedTransformer, prompt: str, layer: int, site: str, prepend_bos: bool, device: torch.device) -> torch.Tensor:
    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos).to(device)
    last_idx = tokens.shape[1] - 1
    cache: dict[str, torch.Tensor] = {}

    def grab(activation: torch.Tensor, hook) -> torch.Tensor:  # type: ignore[no-redef]
        cache["resid"] = activation.detach()
        return activation

    _ = model.run_with_hooks(tokens, return_type=None, stop_at_layer=layer + 1, fwd_hooks=[(f"blocks.{layer}.{site}", grab)])
    return cache["resid"][0, last_idx, :].clone().to(device)


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm() + 1e-8)


def build_vectors(
    model: HookedTransformer,
    inj_layer: int,
    prompt_pairs: List[tuple[str, str]],
    prepend_bos: bool,
    device: torch.device,
    inject_site: str,
) -> dict[str, torch.Tensor]:
    bias_vec = torch.stack([
        residual_at_last_token(model, p_f, inj_layer, inject_site, prepend_bos, device)
        - residual_at_last_token(model, p_m, inj_layer, inject_site, prepend_bos, device)
        for (p_f, p_m) in prompt_pairs
    ]).mean(dim=0)
    bias_vec = _unit(bias_vec)

    rand_vec = _unit(torch.randn_like(bias_vec))
    orth_vec = _unit(rand_vec - (rand_vec @ bias_vec) * bias_vec / (bias_vec @ bias_vec))
    return {"bias": bias_vec, "random": rand_vec, "orth": orth_vec}


@torch.no_grad()
def get_steered_logits(
    model: HookedTransformer,
    prompt: str,
    steer_vec: torch.Tensor,
    alpha: float,
    inject_hook_name: str,
    read_hook_name: str,
    inject_layer: int,
    read_layer: int,
    prepend_bos: bool,
    device: torch.device,
    steer_all_tokens: bool = True,
) -> torch.Tensor:
    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos).to(device)
    last_idx = tokens.shape[1] - 1
    cache: dict[str, torch.Tensor] = {}

    def do_steer(act: torch.Tensor, hook) -> torch.Tensor:  # type: ignore[no-redef]
        vec = steer_vec.to(act.device)
        if steer_all_tokens:
            # Steer all token positions
            act[:, :, :] = act[:, :, :] + (alpha * vec)
        else:
            # Steer only the last token position (original behavior)
            act[:, last_idx, :] = act[:, last_idx, :] + (alpha * vec)
        return act

    def do_read(act: torch.Tensor, hook) -> torch.Tensor:  # type: ignore[no-redef]
        cache["resid"] = act.detach().clone()
        return act

    _ = model.run_with_hooks(
        tokens,
        return_type=None,
        stop_at_layer=max(inject_layer, read_layer) + 1,
        fwd_hooks=[(inject_hook_name, do_steer), (read_hook_name, do_read)],
    )

    resid = model.ln_final(cache["resid"][:, last_idx : last_idx + 1, :])
    logits = model.unembed(resid)[0, 0, :]
    return logits


@torch.no_grad()
def sweep_alpha(
    model: HookedTransformer,
    vector: torch.Tensor,
    alpha_values: Iterable[float],
    prompt: str,
    inj_layer: int,
    read_layer: int,
    inject_hook_name: str,
    read_hook_name: str,
    prepend_bos: bool,
    device: torch.device,
    steer_all_tokens: bool = True,
) -> List[torch.Tensor]:
    """
    Returns: A list of logits for each alpha value.
    """
    logits_list: List[torch.Tensor] = []
    for alpha in alpha_values:
        logits = get_steered_logits(
            model,
            prompt,
            vector,
            float(alpha),
            inject_hook_name,
            read_hook_name,
            inj_layer,
            read_layer,
            prepend_bos,
            device,
            steer_all_tokens,
        )
        logits_list.append(logits)
    return logits_list


@torch.no_grad()
def logit_diffs(logit_list: List[torch.Tensor], choice1_id: int, choice2_id: int) -> List[float]:
    return [float((logits[choice1_id] - logits[choice2_id]).item()) for logits in logit_list]

@torch.no_grad()
def prob_diffs(logit_list: List[torch.Tensor], choice1_id: int, choice2_id: int) -> List[float]:
    """Compute probability differences (Choice1 - Choice2) from logits using softmax. Values are bounded between -1 and 1."""
    diffs = []
    for logits in logit_list:
        probs = torch.softmax(logits, dim=-1)
        # Compute probability difference: P(Choice1) - P(Choice2)
        prob_diff = float((probs[choice1_id] - probs[choice2_id]).item())
        diffs.append(prob_diff)
    return diffs

@torch.no_grad()
def compute_perplexity(logit_list: List[torch.Tensor], target_token_id: int) -> List[float]:
    """
    Compute perplexity for each set of logits given a target token.
    
    Args:
        logit_list: List of logit tensors
        target_token_id: ID of the target token (e.g., he or she)
    
    Returns:
        List of perplexity values
    """
    perplexities = []
    for logits in logit_list:
        probs = torch.softmax(logits, dim=-1)
        target_prob = probs[target_token_id].item()
        target_prob = max(target_prob, 1e-10)
        # negative log likelihood
        nll = -torch.log(torch.tensor(target_prob))
        perplexity = torch.exp(nll).item()
        perplexities.append(perplexity)
    return perplexities