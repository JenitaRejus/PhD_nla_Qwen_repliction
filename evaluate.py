"""
evaluate.py
===========
Step 4 of the NLA pipeline — evaluation, analysis, and figure generation.

This script does four things:

  1. Core FVE evaluation: generates text for 200 held-out activations,
     reconstructs each via the AR, and reports Fraction of Variance Explained.

  2. Training curve figure: plots per-epoch reward and Val FVE from the
     training_log.json saved by train_nla.py.

  3. FVE distribution figure: histogram of per-sample FVE values, showing
     the bimodal distribution (good reconstructions vs degenerate outputs).

  4. Layer sweep evaluation (optional): runs the same evaluation pipeline
     across activations extracted at different layers (requires running
     extract_activations_multilayer() first in data_pipeline.py).

All figures are saved to results/ as PNG files ready to embed in the README.

Key evaluation decisions
------------------------
* We use GREEDY generation (argmax, not sampling) for evaluation so results
  are deterministic and reproducible.
* The AR reads z_ids directly (not via sft_av hidden states) — this matches
  the RL training setup where the AR model is the standalone ARModel class.
* FVE is computed as: 1 - avg_MSE / global_var
  where global_var is the total variance of activations across the full
  dataset (not just the eval subset), giving a stable denominator.
* Per-sample FVE is also reported so we can see the distribution, not just
  the mean. The bimodal distribution (good samples ~0.90, bad ~0.35) is an
  important finding worth discussing in the README.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")   # non-interactive backend, works on Colab / headless
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


# ================================================================== #
# Shared constants
# ================================================================== #

MIN_GEN_LENGTH = 8   # Must match train_nla.py


# ================================================================== #
# Dataset
# ================================================================== #

class NLADataset(Dataset):
    def __init__(self, data_path: str):
        self.data = torch.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]["activation"]


# ================================================================== #
# AR Model (must match generate_sft.py / train_nla.py)
# ================================================================== #

class ARModel(nn.Module):
    """
    Frozen LM backbone + trainable MLP head.
    Identical definition to train_nla.py so evaluate.py is self-contained.
    """

    def __init__(self, base_model_name: str, hidden_dim: int):
        super().__init__()
        self.base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            output_hidden_states=True,
        )
        for p in self.base.parameters():
            p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, input_ids, attention_mask):
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = out.hidden_states[-1]
        mask_exp    = attention_mask.unsqueeze(-1).float()
        pooled      = (last_hidden * mask_exp).sum(1)
        pooled      = pooled / mask_exp.sum(1).clamp(min=1e-9)
        return self.head(pooled)


# ================================================================== #
# Generation helper
# ================================================================== #

@torch.no_grad()
def generate_greedy(
    av_model, embed_layer, h, scale,
    special_id, eos_id, pad_id,
    max_gen_length, device, tokenizer=None,
):
    """
    Greedy deterministic generation for a single activation h (1, D).
    Identical EOS-blocking and whitespace-blocking as training generation.
    """
    generated = torch.full((1, 1), special_id, dtype=torch.long, device=device)

    ws_ids = []
    if tokenizer is not None:
        for ws_char in ["\t", "\n", "  "]:
            toks = tokenizer.encode(ws_char, add_special_tokens=False)
            ws_ids.extend(toks)
    ws_ids = list(set(ws_ids))

    for step in range(max_gen_length):
        token_embeds = embed_layer(generated)
        token_embeds[:, 0, :] = (h / scale).to(token_embeds.dtype)

        outputs    = av_model(inputs_embeds=token_embeds)
        logits     = outputs.logits[:, -1, :].float()

        if step < MIN_GEN_LENGTH:
            logits[:, eos_id] = -1e9
            if pad_id is not None and pad_id != eos_id:
                logits[:, pad_id] = -1e9
            for ws_id in ws_ids:
                logits[:, ws_id] = -1e9

        next_token = torch.argmax(logits, dim=-1)
        generated  = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

        if next_token.item() == eos_id:
            break

    return generated


# ================================================================== #
# Core evaluation
# ================================================================== #

def run_evaluation(
    data_path="data/activations.pt",
    scale_path="data/scale.txt",
    av_path="nla_trained/best_av_final",
    ar_path="nla_trained/best_ar_final.bin",
    output_dir="results",
    max_gen_length=60,
    num_eval_samples=200,
    device="cuda",
):
    """
    Evaluate the trained NLA on held-out activations.
    Returns a results dict with global_fve, avg_mse, global_var, examples.
    """
    print(f"\n{'='*60}")
    print("NLA Evaluation")
    print(f"{'='*60}")

    with open(scale_path) as f:
        scale = float(f.read().strip())

    tokenizer  = AutoTokenizer.from_pretrained(av_path)
    special_id = tokenizer.convert_tokens_to_ids("<|activation|>")
    pad_id     = tokenizer.pad_token_id
    eos_id     = tokenizer.eos_token_id

    av_model = AutoModelForCausalLM.from_pretrained(
        av_path, torch_dtype=torch.float32
    ).to(device)
    av_model.eval()
    embed_layer = av_model.get_input_embeddings()

    hidden_dim = AutoConfig.from_pretrained(av_path).hidden_size
    ar_model   = ARModel(av_path, hidden_dim).to(device)

    head_state = torch.load(ar_path, map_location=device)

    # Handle both key formats automatically
    if "head.0.weight" in head_state:
        head_state = {
            k.replace("head.", ""): v for k, v in head_state.items()
        }

    ar_model.head.load_state_dict(head_state)
    ar_model.eval()

    dataset    = NLADataset(data_path)
    all_h      = torch.stack([dataset[i] for i in range(len(dataset))], dim=0)
    global_var = all_h.var(dim=0).sum().item()
    print(f"Full dataset: {len(dataset)} samples")
    print(f"Global variance of h: {global_var:.4f}")

    mse_loss = nn.MSELoss()
    indices  = torch.randperm(len(dataset))[:num_eval_samples].tolist()

    total_mse = 0.0
    examples  = []

    print(f"\nRunning greedy generation on {num_eval_samples} samples...")

    with torch.no_grad():
        for rank, idx in enumerate(indices):
            h = dataset[idx].unsqueeze(0).to(device)

            generated = generate_greedy(
                av_model, embed_layer, h, scale,
                special_id, eos_id, pad_id,
                max_gen_length, device, tokenizer,
            )

            z_ids = generated[:, 1:]
            text  = tokenizer.decode(z_ids[0], skip_special_tokens=True)

            z_attn_mask = (z_ids != pad_id).long()
            h_hat       = ar_model(z_ids, z_attn_mask)

            mse_val    = mse_loss(h_hat, h).item()
            total_mse += mse_val
            fve_sample = 1.0 - (mse_val / global_var) if global_var > 0 else 0.0

            valid_tokens = z_ids[z_ids != pad_id]
            valid_tokens = valid_tokens[valid_tokens != eos_id]
            gen_len      = valid_tokens.size(0)

            ws_ids_set = set()
            for ws_char in ["\t", "\n", "  "]:
                ws_ids_set.update(
                    tokenizer.encode(ws_char, add_special_tokens=False)
                )
            ws_count     = sum(1 for t in valid_tokens.tolist() if t in ws_ids_set)
            ws_ratio     = ws_count / max(gen_len, 1)
            unique_ratio = (
                len(torch.unique(valid_tokens)) / gen_len if gen_len > 0 else 0.0
            )

            examples.append({
                "source_idx":     idx,
                "generated_text": text,
                "mse":            round(mse_val, 6),
                "fve":            round(fve_sample, 4),
                "gen_length":     gen_len,
                "unique_ratio":   round(unique_ratio, 3),
                "ws_ratio":       round(ws_ratio, 3),
            })

            if (rank + 1) % 50 == 0:
                print(f"  {rank+1}/{num_eval_samples} done...")

    avg_mse    = total_mse / len(indices)
    global_fve = 1.0 - (avg_mse / global_var) if global_var > 0 else 0.0

    fve_values = [e["fve"] for e in examples]
    print(f"\n{'─'*50}")
    print(f"Results over {len(indices)} samples:")
    print(f"  global FVE : {global_fve:.4f}")
    print(f"  avg MSE    : {avg_mse:.6f}")
    print(f"  global_var : {global_var:.6f}")
    print(f"  FVE  min   : {min(fve_values):.4f}")
    print(f"  FVE  p25   : {np.percentile(fve_values, 25):.4f}")
    print(f"  FVE median : {np.median(fve_values):.4f}")
    print(f"  FVE  p75   : {np.percentile(fve_values, 75):.4f}")
    print(f"  FVE  max   : {max(fve_values):.4f}")
    print(f"{'─'*50}")

    print("\n--- Top 5 reconstructions (highest FVE) ---")
    for e in sorted(examples, key=lambda x: x["fve"], reverse=True)[:5]:
        print(f"  FVE={e['fve']:.3f} len={e['gen_length']:>3} "
              f"uniq={e['unique_ratio']:.2f} | \"{e['generated_text'][:80]}\"")

    print("\n--- Bottom 5 reconstructions (lowest FVE) ---")
    for e in sorted(examples, key=lambda x: x["fve"])[:5]:
        print(f"  FVE={e['fve']:.3f} len={e['gen_length']:>3} "
              f"uniq={e['unique_ratio']:.2f} | \"{e['generated_text'][:80]}\"")

    os.makedirs(output_dir, exist_ok=True)
    results = {
        "global_fve":  global_fve,
        "avg_mse":     avg_mse,
        "global_var":  global_var,
        "num_samples": len(indices),
        "fve_stats": {
            "min":    min(fve_values),
            "p25":    float(np.percentile(fve_values, 25)),
            "median": float(np.median(fve_values)),
            "p75":    float(np.percentile(fve_values, 75)),
            "max":    max(fve_values),
        },
        "examples": examples,
    }
    out_path = os.path.join(output_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved -> {out_path}")

    return results


# ================================================================== #
# Figure 1: FVE distribution histogram
# ================================================================== #

def plot_fve_distribution(results, output_dir="results"):
    """
    Histogram of per-sample FVE values.
    Supports the README finding about bimodal distribution.
    """
    fve_values = [e["fve"] for e in results["examples"]]
    global_fve = results["global_fve"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(fve_values, bins=30, color="#4C72B0", edgecolor="white",
            linewidth=0.5, alpha=0.85)
    ax.axvline(global_fve, color="#DD4444", linewidth=2,
               linestyle="--", label=f"Mean FVE = {global_fve:.3f}")
    ax.axvline(0.0, color="gray", linewidth=1,
               linestyle=":", label="FVE = 0 (mean predictor baseline)")

    ax.set_xlabel("Per-sample FVE", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of per-sample FVE\n(Qwen2.5-0.5B, Layer 8)",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(-0.1, 1.05)

    arr   = np.array(fve_values)
    n_good = (arr > 0.7).sum()
    n_bad  = (arr < 0.5).sum()
    if n_good > 0 and n_bad > 0:
        ylim = ax.get_ylim()[1]
        ax.annotate(f"Good ({n_good})\nFVE > 0.7",
                    xy=(0.85, ylim * 0.6), fontsize=9, color="#2D6A2D", ha="center")
        ax.annotate(f"Degenerate ({n_bad})\nFVE < 0.5",
                    xy=(0.3, ylim * 0.6), fontsize=9, color="#882222", ha="center")

    plt.tight_layout()
    path = os.path.join(output_dir, "fve_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved -> {path}")


# ================================================================== #
# Figure 2: Training curve
# ================================================================== #

def plot_training_curve(
    training_log_path="nla_trained/training_log.json",
    output_dir="results",
):
    """
    Three-panel plot: Val FVE, avg reward, and KL divergence per epoch.
    Rising reward + falling FVE = reward hacking (discuss in README).
    """
    if not os.path.exists(training_log_path):
        print(f"Training log not found at {training_log_path} -- skipping.")
        return

    with open(training_log_path) as f:
        log = json.load(f)

    epochs_data = log["epochs"]
    epochs      = [e["epoch"]      for e in epochs_data]
    val_fve     = [e["val_fve"]    for e in epochs_data]
    avg_reward  = [e["avg_reward"] for e in epochs_data]
    kl_vals     = [e["kl"]         for e in epochs_data]
    best_fve    = log["best_val_fve"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    ax.plot(epochs, val_fve, "o-", color="#4C72B0", linewidth=2, markersize=6)
    best_epoch = val_fve.index(max(val_fve)) + 1
    ax.axvline(best_epoch, color="#DD4444", linestyle="--", linewidth=1.5,
               label=f"Best epoch {best_epoch}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val FVE")
    ax.set_title("Validation FVE over training")
    ax.set_ylim(-0.1, 1.05); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, avg_reward, "s-", color="#55A868", linewidth=2, markersize=6)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Avg reward per sample")
    ax.set_title("Average training reward"); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(epochs, kl_vals, "^-", color="#C44E52", linewidth=2, markersize=6)
    ax.set_xlabel("Epoch"); ax.set_ylabel("KL divergence")
    ax.set_title("KL divergence from SFT reference"); ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"NLA Training -- Qwen2.5-0.5B Layer 8  |  Best Val FVE = {best_fve:.4f}",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "training_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved -> {path}")


# ================================================================== #
# Figure 3: Layer sweep
# ================================================================== #

def run_layer_sweep(
    layer_indices=None,
    data_base_dir="data",
    av_path="nla_trained/av_sft_l8",
    ar_path="nla_trained/ar_head_l8.bin",
    output_dir="results",
    max_gen_length=60,
    num_eval_samples=100,
    device="cuda",
):
    """
    Evaluate FVE for activations extracted at different layers.
    Requires extract_activations_multilayer() from data_pipeline.py.
    """
    if layer_indices is None:
        layer_indices = [6, 12, 16, 20, 24]

    sweep_results = {}

    for layer_idx in layer_indices:
        data_dir   = os.path.join(data_base_dir, f"layer_{layer_idx}")
        data_path  = os.path.join(data_dir, "activations.pt")
        scale_path = os.path.join(data_dir, "scale.txt")

        if not os.path.exists(data_path):
            print(f"Layer {layer_idx}: data not found, skipping.")
            continue

        print(f"\nEvaluating layer {layer_idx}...")
        results = run_evaluation(
            data_path=data_path,
            scale_path=scale_path,
            av_path=av_path,
            ar_path=ar_path,
            output_dir=os.path.join(output_dir, f"layer_{layer_idx}"),
            max_gen_length=max_gen_length,
            num_eval_samples=num_eval_samples,
            device=device,
        )
        sweep_results[layer_idx] = results["global_fve"]
        print(f"  Layer {layer_idx}: FVE = {results['global_fve']:.4f}")

    if not sweep_results:
        print("No layer data found. Run extract_activations_multilayer() first.")
        return sweep_results

    layers = sorted(sweep_results.keys())
    fves   = [sweep_results[l] for l in layers]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layers, fves, "o-", color="#4C72B0", linewidth=2.5,
            markersize=8, markerfacecolor="white", markeredgewidth=2)
    best_layer = layers[fves.index(max(fves))]
    ax.axvline(best_layer, color="#DD4444", linestyle="--", linewidth=1.5,
               label=f"Best: Layer {best_layer} (FVE={max(fves):.3f})")
    ax.set_xlabel("Layer index", fontsize=12)
    ax.set_ylabel("FVE", fontsize=12)
    ax.set_title("FVE by Layer -- Qwen2.5-0.5B", fontsize=13)
    ax.set_ylim(0, 1.05); ax.set_xticks(layers)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "layer_sweep.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nLayer sweep figure saved -> {path}")

    sweep_path = os.path.join(output_dir, "layer_sweep.json")
    with open(sweep_path, "w") as f:
        json.dump({"layer_fve": sweep_results}, f, indent=2)
    print(f"Layer sweep data saved -> {sweep_path}")
    return sweep_results


# ================================================================== #
# Qualitative analysis
# ================================================================== #

def analyse_failure_modes(results, output_dir="results"):
    """
    Categorise samples into good / mediocre / degenerate and diagnose
    the root cause of each degenerate output. Saved to failure_analysis.json.
    """
    examples   = results["examples"]
    good       = [e for e in examples if e["fve"] > 0.7]
    mediocre   = [e for e in examples if 0.4 < e["fve"] <= 0.7]
    degenerate = [e for e in examples if e["fve"] <= 0.4]

    print(f"\n{'='*60}")
    print("Qualitative Failure Mode Analysis")
    print(f"{'='*60}")
    print(f"Good       (FVE > 0.70): {len(good):>4}  ({100*len(good)/len(examples):.1f}%)")
    print(f"Mediocre (0.40-0.70):    {len(mediocre):>4}  ({100*len(mediocre)/len(examples):.1f}%)")
    print(f"Degenerate (FVE <= 0.40):{len(degenerate):>4}  ({100*len(degenerate)/len(examples):.1f}%)")

    ws_collapse    = [e for e in degenerate if e["ws_ratio"] > 0.5]
    rep_collapse   = [e for e in degenerate if e["unique_ratio"] < 0.3 and e["ws_ratio"] <= 0.5]
    short_collapse = [e for e in degenerate if e["gen_length"] < MIN_GEN_LENGTH]
    other          = [e for e in degenerate
                      if e not in ws_collapse and e not in rep_collapse
                      and e not in short_collapse]

    print(f"\nDegenerate breakdown:")
    print(f"  Whitespace collapse (ws_ratio > 0.5):  {len(ws_collapse)}")
    print(f"  Repetition collapse (uniq < 0.3):      {len(rep_collapse)}")
    print(f"  Too short (len < {MIN_GEN_LENGTH}):                {len(short_collapse)}")
    print(f"  Other (low FVE despite normal text):   {len(other)}")

    print("\n--- 3 good examples ---")
    for e in good[:3]:
        print(f"  FVE={e['fve']:.3f} | \"{e['generated_text'][:90]}\"")

    print("\n--- 3 degenerate examples ---")
    for e in degenerate[:3]:
        tag = "ws" if e["ws_ratio"] > 0.5 else ("rep" if e["unique_ratio"] < 0.3 else "?")
        print(f"  FVE={e['fve']:.3f} [{tag}] | \"{e['generated_text'][:90]}\"")

    print("\n--- 3 mediocre examples ---")
    for e in mediocre[:3]:
        print(f"  FVE={e['fve']:.3f} | \"{e['generated_text'][:90]}\"")

    analysis = {
        "total": len(examples),
        "good_count": len(good),
        "mediocre_count": len(mediocre),
        "degenerate_count": len(degenerate),
        "degenerate_breakdown": {
            "whitespace_collapse": len(ws_collapse),
            "repetition_collapse": len(rep_collapse),
            "too_short": len(short_collapse),
            "other": len(other),
        },
        "good_examples":       good[:5],
        "degenerate_examples": degenerate[:5],
        "mediocre_examples":   mediocre[:5],
    }

    path = os.path.join(output_dir, "failure_analysis.json")
    with open(path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nFailure analysis saved -> {path}")
    return analysis


# ================================================================== #
# Entry point
# ================================================================== #

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("results/layer_8", exist_ok=True)

results_l8 = run_evaluation(
    data_path        = "data/layer_8/activations.pt",
    scale_path       = "data/layer_8/scale.txt",
    av_path          = "nla_trained_l8/best_av_final",
    ar_path          = "nla_trained_l8/best_ar_final.bin",
    output_dir       = "results/layer_8",
    max_gen_length   = 60,
    num_eval_samples = 200,
    device           = device,
)
print(f"Layer 8 FVE: {results_l8['global_fve']:.4f}")


plot_fve_distribution(results_l8, output_dir="results/layer_8")

plot_training_curve(
    training_log_path = "nla_trained_l8/training_log.json",
    output_dir        = "results/layer_8"
)

analyse_failure_modes(results_l8, output_dir="results/layer_8")

# 5. Optional — only run after extract_activations_multilayer()
# run_layer_sweep(
#     layer_indices=[6, 12, 16, 20, 24],
#     data_base_dir="data",
#     av_path="nla_trained/best_av_final",
#     ar_path="nla_trained/best_ar_final.bin",
#     output_dir="results",
# )