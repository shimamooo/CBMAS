"""Model steering operations for bias response curves."""

from typing import Iterable, List
import torch
from transformer_lens import HookedTransformer

#TODO: passing in the same parameters to different functions, probably first consolidate with passing experiment config object and make that more robust


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
    """
    Returns: Logits for the last token of the prompt after steering.
    """

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


