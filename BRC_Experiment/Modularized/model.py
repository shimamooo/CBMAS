from typing import Tuple
import torch
from transformer_lens import HookedTransformer


def load_model(model_name: str, device: torch.device) -> HookedTransformer:
    model = HookedTransformer.from_pretrained(model_name).to(device).eval()
    return model


def get_pronoun_token_ids(model) -> Tuple[int, int]:
    """Return token ids for " he" and " she" (with leading space)."""
    he_id = int(model.to_tokens(" he", prepend_bos=False)[0, 0])
    she_id = int(model.to_tokens(" she", prepend_bos=False)[0, 0])
    return he_id, she_id


def get_choice_token_ids(model) -> Tuple[int, int]:
    """Return token ids for '1' and '2' tokens (simple numbers) used in reassurance dataset."""
    choice1_id = int(model.to_tokens("1", prepend_bos=False)[0, 0])
    choice2_id = int(model.to_tokens("2", prepend_bos=False)[0, 0])
    return choice1_id, choice2_id


