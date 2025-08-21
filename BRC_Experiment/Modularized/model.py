from typing import Tuple, TYPE_CHECKING

import torch

try:  # Optional at import time; tests may patch this symbol
    from transformer_lens import HookedTransformer  # type: ignore
except Exception:  # pragma: no cover - fallback for environments without transformer_lens
    HookedTransformer = None  # type: ignore


def load_model(model_name: str, device: torch.device):
    global HookedTransformer  # use patched symbol in tests if provided
    if HookedTransformer is None:  # lazy import if not available yet
        from transformer_lens import HookedTransformer as _HT  # type: ignore
        HookedTransformer = _HT  # type: ignore
    model = HookedTransformer.from_pretrained(model_name).to(device).eval()  # type: ignore[attr-defined]
    return model


def get_pronoun_token_ids(model) -> Tuple[int, int]:
    """Return token ids for " he" and " she" (with leading space)."""
    he_id = int(model.to_tokens(" he", prepend_bos=False)[0, 0])
    she_id = int(model.to_tokens(" she", prepend_bos=False)[0, 0])
    return he_id, she_id


