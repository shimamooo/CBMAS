#!/usr/bin/env python3
"""
Bias Response Curve Experiment on GPT-2 Small using TransformerLens

This script:
- Loads GPT-2 small via TransformerLens
- Computes a gender bias steering vector ("He" vs "She") at a chosen layer
- Sweeps alpha values, injecting the vector at inference
- Compares to random and orthogonal directions
- Plots Δ = logit("He") - logit("She") vs alpha for all three
- Saves the plot as bias_response_curve.png

Author: (Your Name)
"""

import torch
import transformer_lens
import matplotlib.pyplot as plt
import numpy as np
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

def main():
    # 1. Set device (CUDA if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load GPT-2 small model to device
    model = HookedTransformer.from_pretrained("gpt2-small")
    model.to(device)
    model.eval()

    # 3. Define template sentences
    prompt_he = "He is a doctor."
    prompt_she = "She is a doctor."

    # 4. Choose layer and position for intervention
    LAYER = 5  # 0-indexed, so this is the 6th layer
    HOOK_NAME = f"blocks.{LAYER}.hook_resid_mid"

    # 5. Tokenize prompts and get position of interest
    tokens_he = model.to_tokens(prompt_he).to(device)
    tokens_she = model.to_tokens(prompt_she).to(device)
    TARGET_POS = 0

    # Get token IDs for "He" and "She"
    he_id = model.tokenizer.encode("He")[0]
    she_id = model.tokenizer.encode("She")[0]

    # 6. Get activations for "He" and "She" prompts at the chosen layer
    with torch.no_grad():
        activations = {}
        def save_hook(act, hook):
            activations[hook.name] = act.detach().clone()
        _ = model(tokens_he, return_type=None, stop_at_layer=LAYER+1, 
                  fwd_hooks=[(HOOK_NAME, save_hook)])
        act_he = activations[HOOK_NAME][0, TARGET_POS, :].clone().to(device)
        _ = model(tokens_she, return_type=None, stop_at_layer=LAYER+1, 
                  fwd_hooks=[(HOOK_NAME, save_hook)])
        act_she = activations[HOOK_NAME][0, TARGET_POS, :].clone().to(device)

    # 7. Compute bias direction vector
    v_bias = (act_he - act_she)
    v_bias = v_bias / v_bias.norm()

    # 8. Prepare control vectors
    torch.manual_seed(42)
    v_rand = torch.randn_like(v_bias, device=device)
    v_rand = v_rand / v_rand.norm()
    v_orth = v_rand - (v_rand @ v_bias) * v_bias
    v_orth = v_orth / v_orth.norm()

    # 9. Sweep alpha values
    alphas = np.arange(-5, 5.1, 0.5)
    results = {"bias": [], "random": [], "orth": []}

    # Helper: Run model with vector injection and get Δ = logit("He") - logit("She")
    def get_delta(prompt, vector, alpha):
        tokens = model.to_tokens(prompt).to(device)
        def steer_hook(act, hook):
            act[:, TARGET_POS, :] = act[:, TARGET_POS, :] + alpha * vector
            return act
        with torch.no_grad():
            logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(HOOK_NAME, steer_hook)],
                return_type="logits"
            )
        logits_pos = logits[0, TARGET_POS, :]
        return (logits_pos[he_id] - logits_pos[she_id]).item()

    target_prompt = "He is a doctor."
    for alpha in alphas:
        delta_bias = get_delta(target_prompt, v_bias, alpha)
        delta_rand = get_delta(target_prompt, v_rand, alpha)
        delta_orth = get_delta(target_prompt, v_orth, alpha)
        results["bias"].append(delta_bias)
        results["random"].append(delta_rand)
        results["orth"].append(delta_orth)

    # 10. Plot results
    plt.figure(figsize=(8,6))
    plt.plot(alphas, results["bias"], label="Bias vector (He-She)", color="C0", linewidth=2)
    plt.plot(alphas, results["random"], label="Random vector", color="C1", linestyle="--")
    plt.plot(alphas, results["orth"], label="Orthogonal vector", color="C2", linestyle=":")
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel(r"Steering coefficient $\alpha$")
    plt.ylabel(r"$\Delta$ = logit('He') - logit('She') at position 0")
    plt.title("Bias Response Curve (GPT-2 Small, Layer 5)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()
    plt.savefig("bias_response_curve.png", dpi=200)
    print("Plot saved as bias_response_curve.png")

if __name__ == "__main__":
    main()

