"""Metric computation functions for bias response curves."""

from typing import List
import torch


@torch.no_grad()
def logit_diffs(logit_list: List[torch.Tensor], choice1_id: int, choice2_id: int) -> List[float]:
    """Compute logit differences (Choice1 - Choice2) from a list of logit tensors."""
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
