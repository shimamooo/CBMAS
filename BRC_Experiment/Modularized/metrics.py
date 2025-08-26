"""
Metric computation functions for bias response curves.

This module provides comprehensive metrics for analyzing bias response curves,
including both local effects (on specific token pairs) and global effects 
(on the entire distribution).

Local Effects Metrics:
- logit_diffs: Raw margin between tokens (linear in α, clean signal)
- prob_diffs: Normalized probability differences (bounded [-1,1])
- odds_ratios: Human-interpretable relative likelihood (e^Δ)
- rank_changes: Position movement of tokens in sorted vocabulary

Global Effects Metrics:
- compute_perplexity: Model fluency/naturalness preservation
- kl_divergences: Overall distribution shift measurement
- top_k_analysis: Qualitative inspection of distributional changes
"""

from typing import List, Tuple, Dict, Any, Optional
import torch


# ==================== LOCAL EFFECTS METRICS ====================

@torch.no_grad()
def logit_diffs(logit_list: List[torch.Tensor], choice1_id: int, choice2_id: int) -> List[float]:
    """
    Compute logit differences (Choice1 - Choice2) from a list of logit tensors.
    
    Args:
        logit_list: List of logit tensors from model forward passes
        choice1_id: Token ID for first choice 
        choice2_id: Token ID for second choice
    
    Returns:
        List of logit differences (Δ = z_tok1 - z_tok2)
    """
    # Stack all logit tensors into a single batch dimension
    logits_batch = torch.stack(logit_list)  # [batch_size, vocab_size]
    
    # Compute differences in a single operation
    diffs_batch = logits_batch[:, choice1_id] - logits_batch[:, choice2_id]  # [batch_size]
    
    return diffs_batch.tolist()


@torch.no_grad()
def prob_diffs(logit_list: List[torch.Tensor], choice1_id: int, choice2_id: int) -> List[float]:
    """
    Compute probability differences (Choice1 - Choice2) from logits using softmax.
    
    Args:
        logit_list: List of logit tensors from model forward passes
        choice1_id: Token ID for first choice
        choice2_id: Token ID for second choice
    
    Returns:
        List of probability differences (P(Choice1) - P(Choice2)), bounded in [-1,1]
    """
    # Stack all logit tensors into a single batch dimension
    logits_batch = torch.stack(logit_list)  # [batch_size, vocab_size]
    
    # Compute softmax probabilities in a single operation
    probs_batch = torch.softmax(logits_batch, dim=-1)  # [batch_size, vocab_size]
    
    # Compute probability differences in a single operation
    prob_diffs_batch = probs_batch[:, choice1_id] - probs_batch[:, choice2_id]  # [batch_size]
    
    return prob_diffs_batch.tolist()


@torch.no_grad()
def odds_ratios(logit_list: List[torch.Tensor], choice1_id: int, choice2_id: int) -> List[float]:
    """
    Compute odds ratios (e^Δ) from logit differences.
    
    Human-interpretable metric showing relative likelihood. E.g., value of 7.0
    means "Token A is 7× more likely than token B."
    
    Args:
        logit_list: List of logit tensors from model forward passes
        choice1_id: Token ID for first choice
        choice2_id: Token ID for second choice
    
    formula: e^(logit_diff) = e^(z_tok1 - z_tok2)
    
    Returns:
        List of odds ratios (e^(logit_diff))
    """
    # Stack all logit tensors into a single batch dimension
    logits_batch = torch.stack(logit_list)  # [batch_size, vocab_size]
    
    # Compute logit differences and exponentiate in a single operation
    logit_diffs_batch = logits_batch[:, choice1_id] - logits_batch[:, choice2_id]  # [batch_size]
    odds_ratios_batch = torch.exp(logit_diffs_batch)  # [batch_size]
    
    return odds_ratios_batch.tolist()


@torch.no_grad()
def rank_changes(
    logit_list: List[torch.Tensor], 
    target_token_ids: List[int],
) -> List[Dict[str, int]]:
    """
    Compute rank changes for target tokens across steering strengths.
    
    Shows how token positions move in the sorted vocabulary as steering strength varies.
    
    Args:
        logit_list: List of logit tensors from model forward passes
        target_token_ids: List of token IDs to track (e.g., [a_id, b_id])
    
    Returns:
        List of dictionaries mapping token_id -> rank for each alpha value.
    """
    rank_changes_list = []
    
    for logits in logit_list:
        # Sort logits in descending order and get ranks
        sorted_indices = torch.argsort(logits, descending=True)
        
        # Create rank mapping: token_id -> rank (0-indexed)
        rank_map = {int(token_id): int(rank) for rank, token_id in enumerate(sorted_indices)}
        
        # Extract ranks for target tokens
        target_ranks = {token_id: rank_map[token_id] for token_id in target_token_ids}
        rank_changes_list.append(target_ranks)
    
    return rank_changes_list


# ==================== GLOBAL EFFECTS METRICS ====================

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
    # Stack all logit tensors into a single batch dimension
    logits_batch = torch.stack(logit_list)  # [batch_size, vocab_size]
    
    # Compute softmax probabilities in a single operation
    probs_batch = torch.softmax(logits_batch, dim=-1)  # [batch_size, vocab_size]
    
    # Extract target token probabilities and clamp to avoid log(0)
    target_probs = probs_batch[:, target_token_id].clamp(min=1e-10)  # [batch_size]
    
    # Compute negative log likelihood and perplexity in a single operation
    nll_batch = -torch.log(target_probs)  # [batch_size]
    perplexity_batch = torch.exp(nll_batch)  # [batch_size]
    
    return perplexity_batch.tolist()


@torch.no_grad()
def kl_divergences(
    original_logits: List[torch.Tensor], 
    steered_logits: List[torch.Tensor]
) -> List[float]:
    """
    Compute KL divergence between original and steered distributions.
    
    Measures overall shift of the whole distribution, not just target tokens.
    High KL = steering perturbs many logits (broad/noisy effect).
    Low KL = steering is specific to bias direction.
    
    Args:
        original_logits: List of baseline logit tensors (α=0)
        steered_logits: List of steered logit tensors (various α values)
    
    Returns:
        List of KL divergence values D_KL(P_original || P_steered)
    """    
    kl_divs = []
    for orig_logits, steer_logits in zip(original_logits, steered_logits):
        orig_probs = torch.softmax(orig_logits, dim=-1)
        steer_probs = torch.softmax(steer_logits, dim=-1)
        
        # KL divergence: D_KL(P||Q) = Σ P(x) * log(P(x) / Q(x))
        kl_div = torch.sum(orig_probs * torch.log(orig_probs / (steer_probs + 1e-10))).item()
        kl_divs.append(kl_div)
    
    return kl_divs


#vault for now.
# @torch.no_grad()
# def top_k_analysis(
#     logit_list: List[torch.Tensor],
#     k: int = 10,
#     tokenizer = None
# ) -> List[Dict[str, Any]]:
#     """
#     Analyze top-k most likely tokens for qualitative inspection.
    
#     Provides concrete examples of distributional shifts beyond target pair.
#     Excellent for intuition-building about steering side effects.
    
#     Args:
#         logit_list: List of logit tensors from model forward passes
#         k: Number of top tokens to analyze
#         tokenizer: Optional tokenizer for decoding token IDs to strings
    
#     Returns:
#         List of dictionaries containing top-k analysis for each alpha value
#         schema: {"top_k_ids": [...], "top_k_probs": [...], "top_k_tokens": [...]}
#     """
#     analyses = []
    
#     for logits in logit_list:
#         probs = torch.softmax(logits, dim=-1)
#         top_k_probs, top_k_ids = torch.topk(probs, k)
        
#         analysis = {
#             "top_k_ids": [int(token_id) for token_id in top_k_ids],
#             "top_k_probs": [float(prob) for prob in top_k_probs],
#         }
        
#         # Add decoded tokens if tokenizer provided
#         if tokenizer is not None:
#             try:
#                 analysis["top_k_tokens"] = [
#                     tokenizer.decode([token_id]) for token_id in analysis["top_k_ids"]
#                 ]
#             except Exception as e:
#                 analysis["top_k_tokens"] = [f"<decode_error_{token_id}>" for token_id in analysis["top_k_ids"]]
        
#         analyses.append(analysis)
    
#     return analyses

