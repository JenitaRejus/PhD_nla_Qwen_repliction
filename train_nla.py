"""
train_nla.py
============
Step 3 of the NLA pipeline — the core RL training loop.

This script jointly trains:
  - The Activation Verbalizer (AV): generates text z given activation h
  - The Activation Reconstructor (AR) head: maps text back to activation h_hat

Training algorithm: REINFORCE (policy gradient)
------------------------------------------------
The AV is the "policy". It takes an activation h as input (injected into
the embedding of a special token) and samples a token sequence z. The AR
then tries to reconstruct h from z. The reconstruction error becomes the
reward signal:

    reward = -log(MSE(AR(z), h) + ε)   [high reward = low MSE]

We then update the AV to increase the probability of token sequences that
received high reward (REINFORCE: ∇J = E[∇log π(z|h) · R]).

A KL-divergence penalty against a frozen copy of the SFT AV prevents the
model from collapsing into degenerate outputs that happen to fool the AR
(e.g. outputting whitespace or repetitive tokens). This mirrors the RLHF
training setup used for ChatGPT and is explicitly described in the paper.

Key fixes vs the initial implementation
----------------------------------------
1. EOS collapse fix: EOS and PAD are blocked for the first MIN_GEN_LENGTH
   steps during generation, forcing the AV to produce real content.

2. Whitespace degeneration fix: A heavy penalty fires when >30% of
   generated tokens are whitespace (tabs, newlines). This was the dominant
   failure mode in the first run (40% of samples were pure tab sequences).

3. Repetition penalty: Scaled continuously by (0.6 - uniqueness_ratio)
   rather than a hard binary threshold, so partial repetition is also
   penalised.

4. Full LM AR: The AR uses a frozen LM backbone + MLP head (defined in
   generate_sft.py), making reconstruction much more expressive than a
   single linear layer.

5. Longer generation: max_gen_length=60 (vs 20 before). The paper uses
   ~200 tokens; 60 is a practical middle ground on free-tier GPU.

6. Early stopping: Training stops when Val FVE has not improved for
   `patience` consecutive epochs, avoiding the epoch-3 overfitting seen
   in the first run.

7. Correct sft conditioning: The frozen SFT reference model always
   receives the FULL generated sequence with the activation injected at
   position 0, matching training conditioning exactly.

Outputs
-------
nla_trained/best_av_final/   — best AV checkpoint (HuggingFace format)
nla_trained/best_ar_final.bin — best AR head state dict
nla_trained/training_log.json — per-epoch metrics for README figures
"""

import copy
import json
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer

import bitsandbytes as bnb

# ------------------------------------------------------------------ #
# Minimum number of real tokens the AV must generate before EOS is
# allowed. Prevents the model from collapsing to empty outputs.
# ------------------------------------------------------------------ #
MIN_GEN_LENGTH = 8


# ================================================================== #
# Dataset
# ================================================================== #

class NLADataset(Dataset):
    """Wraps the saved activations.pt file as a PyTorch Dataset."""

    def __init__(self, data_path: str):
        self.data = torch.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]["activation"]


def compute_global_var(dataset) -> float:
    """
    Compute total variance of activation vectors across the dataset.

    FVE denominator: Var[h] = sum over dimensions of per-dimension variance.
    If FVE = 0, the AR is no better than predicting the mean.
    If FVE = 1, the AR perfectly reconstructs every activation.
    """
    all_h = torch.stack([dataset[i] for i in range(len(dataset))], dim=0)
    return all_h.var(dim=0).sum().item()


# ================================================================== #
# AR Model (must match generate_sft.py definition)
# ================================================================== #

class ARModel(nn.Module):
    """
    Activation Reconstructor: frozen LM backbone + trainable MLP head.

    The backbone reads generated text with full causal self-attention;
    mean-pooling aggregates the sequence into a single vector that the
    MLP projects to activation space.
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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden  = out.hidden_states[-1]                        # (B, T, D)
        mask_exp     = attention_mask.unsqueeze(-1).float()         # (B, T, 1)
        pooled       = (last_hidden * mask_exp).sum(1)              # (B, D)
        pooled       = pooled / mask_exp.sum(1).clamp(min=1e-9)    # (B, D)
        return self.head(pooled)


# ================================================================== #
# Generation helper
# ================================================================== #

@torch.no_grad()
def generate_with_min_length(
    av_model,
    embed_layer,
    batch_h: torch.Tensor,
    scale: float,
    special_id: int,
    eos_id: int,
    pad_id: int,
    max_gen_length: int,
    device: str,
    sample: bool = True,
    tokenizer=None,
) -> torch.Tensor:
    """
    Autoregressive generation with three degeneration guards:

    1. EOS/PAD blocking for the first MIN_GEN_LENGTH steps.
    2. Whitespace token blocking for the first MIN_GEN_LENGTH steps
       (prevents tab-only outputs from the very start).
    3. Sampling (during training) vs greedy (during evaluation).

    The activation h is injected into the embedding of the <|activation|>
    token at position 0. Every forward pass re-injects h so the model always
    has access to the original activation, regardless of sequence length.

    Returns:
        generated: (B, 1 + gen_len) token IDs including the initial special token.
    """
    B         = batch_h.size(0)
    generated = torch.full((B, 1), special_id, dtype=torch.long, device=device)

    # Identify whitespace token IDs to block early in generation.
    ws_ids = set()
    if tokenizer is not None:
        for ws_char in ["\t", "\n", "  "]:
            toks = tokenizer.encode(ws_char, add_special_tokens=False)
            ws_ids.update(toks)
    ws_ids = list(ws_ids)

    for step in range(max_gen_length):
        token_embeds = embed_layer(generated)                        # (B, T, D)

        # Inject activation at position 0 on every step.
        # Dividing by scale keeps the injected vector ~unit magnitude
        # in embedding space, compatible with normal token embeddings.
        token_embeds[:, 0, :] = (batch_h / scale).to(token_embeds.dtype)

        outputs = av_model(inputs_embeds=token_embeds)
        logits  = outputs.logits[:, -1, :].float()                  # (B, vocab)

        # --- Degeneration guards ---
        if step < MIN_GEN_LENGTH:
            # Block EOS and PAD: force model to produce real content.
            logits[:, eos_id] = -1e9
            if pad_id is not None and pad_id != eos_id:
                logits[:, pad_id] = -1e9
            # Block whitespace tokens in the first few steps.
            for ws_id in ws_ids:
                logits[:, ws_id] = -1e9

        if sample:
            probs      = F.softmax(logits, dim=-1)
            dist       = torch.distributions.Categorical(probs)
            next_token = dist.sample()
        else:
            next_token = torch.argmax(logits, dim=-1)

        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

        # Stop if every item in the batch has produced EOS.
        if (next_token == eos_id).all():
            break

    return generated


# ================================================================== #
# SFT reference forward pass
# ================================================================== #

def get_sft_hidden(
    sft_av,
    generated: torch.Tensor,
    batch_h: torch.Tensor,
    scale: float,
    layer_idx: int,
    device: str,
):
    """
    Run the frozen SFT reference model on the full generated sequence with
    the activation injected at position 0.

    This MUST match the conditioning used by the AV model exactly.
    If the conditioning differs, the KL divergence is computed between
    incomparable distributions, destabilising training.

    Returns:
        hidden_at_layer: (B, D) hidden state at layer_idx, last position.
        sft_logits:      (B, T-1, V) logits shifted to match full_logits shape.
    """
    sft_av.to(device)

    sft_embed   = sft_av.get_input_embeddings()
    sft_embeds  = sft_embed(generated)                              # (B, T, D)
    sft_embeds[:, 0, :] = (batch_h / scale).to(sft_embeds.dtype)

    attn_mask = torch.ones(
        generated.shape[:2], dtype=torch.long, device=device
    )

    with torch.no_grad():
        sft_out = sft_av(
            inputs_embeds=sft_embeds,
            attention_mask=attn_mask,
            output_hidden_states=True,
        )

    hidden_at_layer = sft_out.hidden_states[layer_idx][:, -1, :].float()
    # Shift logits: logits[i] predicts token[i+1], same as full_logits.
    sft_logits = sft_out.logits[:, :-1, :].float()

    sft_av.to("cpu")
    return hidden_at_layer, sft_logits


# ================================================================== #
# Reward computation
# ================================================================== #

def compute_reward(
    h_hat: torch.Tensor,
    batch_h: torch.Tensor,
    z_ids: torch.Tensor,
    pad_id: int,
    eos_id: int,
    tokenizer,
    device: str,
    mse_loss: nn.MSELoss,
) -> tuple:
    """
    Compute the per-sample reward for a batch of generated sequences.

    Reward components:
    ------------------
    1. Reconstruction reward: -log(MSE(h_hat, h) + ε)
       High reward when the AR reconstructs the activation accurately.
       Log-scaling gives large gradients early in training when MSE is high.

    2. Length penalty: -5.0 if gen_length < MIN_GEN_LENGTH.
       Belt-and-suspenders guard on top of the EOS-blocking in generation.
       Catches any edge cases that slip through.

    3. Repetition penalty: -4.0 * max(0, 0.6 - uniqueness_ratio)
       Continuous penalty scaled by severity. If 40% of tokens are unique
       (uniqueness_ratio=0.4), penalty = -4.0 * 0.2 = -0.8.
       If all tokens are the same (uniqueness_ratio≈0), penalty = -2.4.

    4. Whitespace penalty: -8.0 if >30% of tokens are whitespace.
       This directly targets the tab/newline degeneration seen in run 1.
       -8.0 is large enough to overwhelm the reconstruction reward.

    Returns:
        reward:         (B,) per-sample scalar reward.
        scalar_mse_val: float, mean MSE across the batch (for logging).
    """
    # 1. Reconstruction reward
    scalar_mse_val      = mse_loss(h_hat, batch_h).item()
    per_sample_mse      = ((h_hat - batch_h) ** 2).mean(dim=1)     # (B,)
    reconstruction_reward = -torch.log(per_sample_mse + 1e-8)

    # Token validity mask: real tokens only (not pad or eos).
    valid_mask = (z_ids != pad_id) & (z_ids != eos_id)
    gen_length = valid_mask.sum(dim=1).float()                      # (B,)

    # 2. Length penalty
    length_penalty = torch.where(
        gen_length < MIN_GEN_LENGTH,
        torch.full_like(gen_length, -5.0),
        torch.zeros_like(gen_length),
    )

    # 3. Repetition penalty (continuous)
    unique_counts = torch.tensor(
        [
            len(torch.unique(seq[mask])) if mask.any() else 0
            for seq, mask in zip(z_ids, valid_mask)
        ],
        device=device,
        dtype=torch.float32,
    )
    uniqueness_ratio  = unique_counts / (gen_length + 1e-5)
    repetition_penalty = -4.0 * torch.clamp(0.6 - uniqueness_ratio, min=0.0)

    # 4. Whitespace penalty
    ws_token_ids = []
    if tokenizer is not None:
        for ws_char in ["\t", "\n", "  "]:
            toks = tokenizer.encode(ws_char, add_special_tokens=False)
            ws_token_ids.extend(toks)
    ws_token_ids = list(set(ws_token_ids))

    if ws_token_ids:
        ws_tensor = torch.tensor(ws_token_ids, device=device)
        ws_counts = torch.tensor(
            [
                torch.isin(seq[mask], ws_tensor).sum().item() if mask.any() else 0
                for seq, mask in zip(z_ids, valid_mask)
            ],
            device=device,
            dtype=torch.float32,
        )
        ws_ratio = ws_counts / (gen_length + 1e-5)
        whitespace_penalty = torch.where(
            ws_ratio > 0.3,
            torch.full_like(gen_length, -8.0),
            torch.zeros_like(gen_length),
        )
    else:
        whitespace_penalty = torch.zeros_like(gen_length)

    reward = (
        reconstruction_reward
        + length_penalty
        + repetition_penalty
        + whitespace_penalty
    )
    return reward, scalar_mse_val


# ================================================================== #
# Validation
# ================================================================== #

@torch.no_grad()
def evaluate_validation(
    av_model,
    sft_av,
    ar_model: ARModel,
    val_loader: DataLoader,
    tokenizer,
    scale: float,
    max_gen_length: int,
    layer_idx: int,
    global_var: float,
    device: str,
) -> float:
    """
    Run greedy generation on the validation set and compute FVE.

    FVE = 1 - avg_MSE / global_var

    FVE > 0: AR reconstructs better than predicting the mean.
    FVE = 1: perfect reconstruction.
    FVE < 0: AR is worse than the mean (something has gone wrong).
    """
    av_model.eval()
    ar_model.eval()

    special_id  = tokenizer.convert_tokens_to_ids("<|activation|>")
    pad_id      = tokenizer.pad_token_id
    eos_id      = tokenizer.eos_token_id
    mse_loss    = nn.MSELoss()
    embed_layer = av_model.get_input_embeddings()

    total_mse = 0.0
    count     = 0

    for batch_h in val_loader:
        batch_h = batch_h.to(device)
        B       = batch_h.size(0)

        # Greedy generation (deterministic for reproducible val metrics).
        generated = generate_with_min_length(
            av_model, embed_layer, batch_h, scale,
            special_id, eos_id, pad_id,
            max_gen_length, device,
            sample=False, tokenizer=tokenizer,
        )

        # AR reconstruction via the full LM encoder.
        z_ids    = generated[:, 1:]
        attn_mask = (z_ids != pad_id).long()
        h_hat    = ar_model(z_ids, attn_mask)

        mse      = mse_loss(h_hat, batch_h)
        total_mse += mse.item() * B
        count    += B

    avg_mse = total_mse / count if count > 0 else 0.0
    fve     = 1.0 - (avg_mse / global_var) if global_var > 0 else 0.0

    av_model.train()
    ar_model.train()
    return fve


# ================================================================== #
# Main training function
# ================================================================== #

def train_nla(
    data_path      = "data/layer_24/activations.pt",    # layer-specific
    scale_path     = "data/layer_24/scale.txt",          # layer-specific
    av_path        = "av_sft_l24",                       # layer-specific
    ar_head_path   = "ar_head_l24.bin",                  # layer-specific
    output_dir     = "nla_trained_l24",                  # layer-specific
    batch_size:      int   = 4,
    lr_av:           float = 1e-5,
    lr_ar:           float = 1e-4,
    kl_coeff:        float = 0.2,
    epochs:          int   = 5,
    max_gen_length:  int   = 60,
    layer_idx:       int   = 24,
    val_size:        int   = 200,
    patience:        int   = 2,
    device:          str   = "cuda",
):
    """
    Joint RL training of AV and AR.

    Args:
        data_path:      Path to activations.pt.
        scale_path:     Path to scale.txt.
        av_path:        Path to SFT AV model directory.
        ar_head_path:   Path to pretrained AR head state dict.
        output_dir:     Where to save best checkpoints and training log.
        batch_size:     Training batch size.
        lr_av:          Learning rate for AV (low — fine-tuning).
        lr_ar:          Learning rate for AR head (higher — smaller network).
        kl_coeff:       Weight of KL divergence penalty.
        epochs:         Maximum training epochs.
        max_gen_length: Maximum tokens the AV generates per activation.
        layer_idx:      Which layer's hidden state the AR reads from sft_av.
        val_size:       Number of samples held out for validation.
        patience:       Early stopping patience (epochs without improvement).
        device:         "cuda" or "cpu".
    """
    print(f"\n{'='*60}")
    print("NLA RL Training")
    print(f"{'='*60}")

    # ---- Load scale ----
    with open(scale_path) as f:
        scale = float(f.read().strip())
    print(f"Scale factor (p75 norm): {scale:.4f}")

    # ---- Tokeniser ----
    tokenizer  = AutoTokenizer.from_pretrained(av_path)
    special_id = tokenizer.convert_tokens_to_ids("<|activation|>")
    pad_id     = tokenizer.pad_token_id
    eos_id     = tokenizer.eos_token_id
    print(f"special_id={special_id}, pad_id={pad_id}, eos_id={eos_id}")

    # ---- AV model (trainable) ----
    av_model = AutoModelForCausalLM.from_pretrained(
        av_path, torch_dtype=torch.float32
    ).to(device)
    av_model.train()

    # ---- SFT reference AV (frozen) ----
    # Kept in float16 to halve its memory footprint.
    # Offloaded to CPU between uses to save GPU memory.
    sft_av = AutoModelForCausalLM.from_pretrained(
        av_path, torch_dtype=torch.float16
    )
    sft_av.eval()
    for p in sft_av.parameters():
        p.requires_grad = False
    sft_av.to("cpu")

    # ---- AR model ----
    from transformers import AutoConfig
    hidden_dim = AutoConfig.from_pretrained(av_path).hidden_size
    ar_model   = ARModel(av_path, hidden_dim).to(device)

    # Load pretrained MLP head weights.
    head_state = torch.load(ar_head_path, map_location=device)
    ar_model.head.load_state_dict(head_state)
    print("AR head loaded successfully.")

    # ---- Optimisers ----
    # AdamW8bit quantises optimizer states → ~2x memory saving for AV.
    opt_av  = bnb.optim.AdamW8bit(av_model.parameters(), lr=lr_av)
    opt_ar  = AdamW(ar_model.head.parameters(), lr=lr_ar)
    scaler  = torch.amp.GradScaler("cuda")

    # ---- Dataset split ----
    full_dataset  = NLADataset(data_path)
    indices       = torch.randperm(len(full_dataset)).tolist()
    train_indices = indices[:-val_size]
    val_indices   = indices[-val_size:]
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset   = Subset(full_dataset, val_indices)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    global_var = compute_global_var(train_dataset)
    print(f"Global variance of h (train set): {global_var:.4f}")

    # ---- Training state ----
    mse_loss    = nn.MSELoss()
    embed_layer = av_model.get_input_embeddings()
    best_fve    = -float("inf")
    best_state  = None
    no_improve  = 0
    training_log = []

    # ================================================================ #
    for epoch in range(epochs):
        total_reward  = 0.0
        total_pg_loss = 0.0
        total_kl      = 0.0
        total_mse_val = 0.0
        num_batches   = 0

        for batch_h in train_loader:
            batch_h = batch_h.to(device)
            B       = batch_h.size(0)

            # -------------------------------------------------------- #
            # Step 1: Generate token sequences (no grad)
            # -------------------------------------------------------- #
            generated = generate_with_min_length(
                av_model, embed_layer, batch_h, scale,
                special_id, eos_id, pad_id,
                max_gen_length, device,
                sample=True, tokenizer=tokenizer,
            )
            z_ids = generated[:, 1:]   # strip the leading special token

            # -------------------------------------------------------- #
            # Step 2: Forward pass WITH grad → log-probs for REINFORCE
            # -------------------------------------------------------- #
            token_embeds_full = embed_layer(generated)
            token_embeds_full[:, 0, :] = (batch_h / scale).to(
                token_embeds_full.dtype
            )

            with torch.amp.autocast("cuda", dtype=torch.float16):
                av_out      = av_model(inputs_embeds=token_embeds_full)
                # Shift: logits[i] predicts z_ids[i] (token at position i+1)
                full_logits = av_out.logits[:, :-1, :].float()          # (B, T, V)
                dist_full   = torch.distributions.Categorical(logits=full_logits)
                sum_log_probs = dist_full.log_prob(z_ids).sum(dim=1)    # (B,)

                # ---------------------------------------------------- #
                # Step 3: SFT reference — compute KL and AR hidden state
                # ---------------------------------------------------- #
                hidden_at_layer, sft_logits = get_sft_hidden(
                    sft_av, generated, batch_h, scale, layer_idx, device
                )
                torch.cuda.empty_cache()

                # AR reconstruction via the full LM encoder.
                z_attn_mask = (z_ids != pad_id).long()
                h_hat       = ar_model(z_ids, z_attn_mask)

                # ---------------------------------------------------- #
                # Step 4: Rewards
                # ---------------------------------------------------- #
                reward, scalar_mse = compute_reward(
                    h_hat, batch_h, z_ids,
                    pad_id, eos_id, tokenizer, device, mse_loss,
                )

                total_reward  += reward.sum().item()
                total_mse_val += scalar_mse

                # ---------------------------------------------------- #
                # Step 5: Losses
                # ---------------------------------------------------- #
                # REINFORCE policy gradient loss.
                # reward.detach(): treat reward as a fixed scalar — no
                # gradients flow through the reward computation itself.
                pg_loss = -(sum_log_probs * reward.detach()).mean()

                # KL divergence: keeps AV close to SFT reference.
                # Prevents reward hacking (e.g. finding weird token
                # sequences that fool the AR without being meaningful text).
                kl = F.kl_div(
                    F.log_softmax(full_logits, dim=-1),
                    F.softmax(sft_logits, dim=-1),
                    reduction="batchmean",
                )

                loss_av = pg_loss + kl_coeff * kl

                # AR loss: straight MSE, trained in float32.
                loss_ar = mse_loss(h_hat, batch_h)

                total_pg_loss += pg_loss.item()
                total_kl      += kl.item()
                num_batches   += 1

            # -------------------------------------------------------- #
            # Step 6: Backward passes
            # -------------------------------------------------------- #

            # AV backward (mixed precision).
            opt_av.zero_grad()
            scaler.scale(loss_av).backward(retain_graph=True)
            scaler.unscale_(opt_av)
            torch.nn.utils.clip_grad_norm_(av_model.parameters(), max_norm=1.0)
            scaler.step(opt_av)
            scaler.update()

            # AR backward (float32, plain optimizer).
            opt_ar.zero_grad()
            loss_ar.backward()
            torch.nn.utils.clip_grad_norm_(ar_model.head.parameters(), max_norm=1.0)
            opt_ar.step()

        # ---- End of epoch ---- #
        n_train       = len(train_dataset)
        avg_reward    = total_reward  / n_train
        avg_pg_loss   = total_pg_loss / num_batches if num_batches else 0.0
        avg_kl        = total_kl      / num_batches if num_batches else 0.0
        avg_mse       = total_mse_val / num_batches if num_batches else 0.0

        val_fve = evaluate_validation(
            av_model, sft_av, ar_model, val_loader,
            tokenizer, scale, max_gen_length, layer_idx,
            global_var, device,
        )

        log_entry = {
            "epoch":      epoch + 1,
            "avg_reward": round(avg_reward, 4),
            "pg_loss":    round(avg_pg_loss, 4),
            "kl":         round(avg_kl, 6),
            "train_mse":  round(avg_mse, 6),
            "val_fve":    round(val_fve, 4),
        }
        training_log.append(log_entry)

        print(
            f"Epoch {epoch+1}/{epochs}: "
            f"reward={avg_reward:.4f}  "
            f"pg_loss={avg_pg_loss:.4f}  "
            f"kl={avg_kl:.6f}  "
            f"train_mse={avg_mse:.6f}  "
            f"Val FVE={val_fve:.4f}"
        )

# ---- Checkpoint ---- #
        # Always create the output directory if it doesn't exist yet
        os.makedirs(output_dir, exist_ok=True)

        if val_fve > best_fve:
            best_fve   = val_fve
            no_improve = 0

            # 1. Update the in-memory best state tracker
            best_state = {
                "av_state": copy.deepcopy(av_model.state_dict()),
                "ar_state": copy.deepcopy(ar_model.head.state_dict()),
            }
            print(f"  -> New best Val FVE: {best_fve:.4f}  (checkpoint saved to memory)")

            # 2. EMERGENCY OVERWRITE: Immediately save these new best weights directly to disk.
            # This cleanly overwrites any previous epoch files.
            av_model.save_pretrained(f"{output_dir}/best_av_final")
            tokenizer.save_pretrained(f"{output_dir}/best_av_final")
            torch.save(
                {
                    "head.0.weight": ar_model.head[0].weight.data,
                    "head.0.bias":   ar_model.head[0].bias.data,
                    "head.2.weight": ar_model.head[2].weight.data,
                    "head.2.bias":   ar_model.head[2].bias.data,
                },
                f"{output_dir}/best_ar_final.bin",
            )
            print(f"  -> 💾 Hard drive checkpoint updated on disk for Epoch {epoch+1}!")
        else:
            no_improve += 1
            print(f"  -> No improvement ({no_improve}/{patience}). Disk files left untouched.")

        # ---- Early stopping ---- #
        if no_improve >= patience:
            print(f"\nEarly stopping after {patience} epochs without improvement.")
            break

    # ================================================================ #
    # Save best model
    # ================================================================ #
    os.makedirs(output_dir, exist_ok=True)

    if best_state is not None:
        av_model.load_state_dict(best_state["av_state"])
        ar_model.head.load_state_dict(best_state["ar_state"])

    av_model.save_pretrained(f"{output_dir}/best_av_final")
    tokenizer.save_pretrained(f"{output_dir}/best_av_final")
    torch.save(
        {
            "head.0.weight": ar_model.head[0].weight.data,
            "head.0.bias":   ar_model.head[0].bias.data,
            "head.2.weight": ar_model.head[2].weight.data,
            "head.2.bias":   ar_model.head[2].bias.data,
        },
        f"{output_dir}/best_ar_final.bin",
    )

    # Save training log for README figures.
    log_path = f"{output_dir}/training_log.json"
    with open(log_path, "w") as f:
        json.dump(
            {
                "best_val_fve": best_fve,
                "global_var":   global_var,
                "config": {
                    "model":          "Qwen/Qwen2.5-0.5B",
                    "layer_idx":      layer_idx,
                    "max_gen_length": max_gen_length,
                    "kl_coeff":       kl_coeff,
                    "batch_size":     batch_size,
                    "lr_av":          lr_av,
                    "lr_ar":          lr_ar,
                },
                "epochs": training_log,
            },
            f,
            indent=2,
        )

    print(f"\nBest Val FVE = {best_fve:.4f}")
    print(f"Model saved  → {output_dir}/best_av_final/")
    print(f"AR head saved → {output_dir}/best_ar_final.bin")
    print(f"Training log → {log_path}")


# ================================================================== #
# Entry point
# ================================================================== #

if __name__ == "__main__":
    train_nla()