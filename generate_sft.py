"""
generate_sft.py
===============
Step 2 of the NLA pipeline.

This script does two things:

  A. Supervised Fine-Tuning (SFT) of the Activation Verbalizer (AV):
     Fine-tunes a copy of the base LLM to produce text that begins with the
     special token <|activation|> followed by a short description of the
     input context. This gives the AV a warm-start before RL training in
     train_nla.py.

  B. Training the Activation Reconstructor (AR):
     The AR reads the generated text and reconstructs the original activation
     vector. Unlike the previous version (which used a single linear layer),
     the AR here uses a FROZEN base LLM as a text encoder, followed by a
     small trainable MLP projection head.

     Why a full LM encoder?
     -----------------------
     A linear layer maps a single 896-dim hidden state to a 896-dim output —
     it has no capacity to integrate information across the sequence. The base
     LLM, by contrast, uses causal self-attention across all generated tokens,
     giving the AR a rich contextual representation to invert. This matches
     the Anthropic paper's AR design more closely and substantially increases
     the ceiling FVE.

     We freeze the LM base and only train the MLP head for efficiency: the
     LM already knows how to read text; we just need to teach it to output
     activations.

Key design choices
------------------
* SFT uses synthetic summaries: "<|activation|> Summary: {first 100 chars of text}"
  This is a deliberate simplification. The Anthropic paper uses the full
  input context as the supervision signal; we use a short prefix because the
  AV will later be trained with RL anyway (SFT is only a warm-start).
* We add exactly two special tokens: <|activation|> and <|pad|>.
  Their embeddings are initialised to the mean of the original vocabulary —
  this prevents gradient explosions from random initialisation.
* The AR head is a 2-layer MLP with GELU activation, which is more expressive
  than a single linear layer while remaining cheap to train.

Outputs
-------
av_sft/          — SFT fine-tuned AV model + tokeniser
ar_head.bin      — trained AR projection head state dict
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


# ================================================================== #
# 1.  Synthetic summary generator
# ================================================================== #

def generate_fake_summary(input_ids: torch.Tensor, tokenizer) -> str:
    """
    Decode token IDs back to text and wrap them in the SFT template.

    The <|activation|> token marks the position where the real activation
    vector will be injected during RL training. The AV learns to produce
    coherent text after seeing this special token.
    """
    text = tokenizer.decode(input_ids, skip_special_tokens=True)[:100]
    return f"<|activation|> Summary: {text}"


# ================================================================== #
# 2.  Activation Reconstructor (AR) — full LM encoder + MLP head
# ================================================================== #

class ARModel(nn.Module):
    """
    Activation Reconstructor.

    Architecture:
      [frozen base LLM] → mean-pooled last hidden state (896-dim)
                        → [trainable 2-layer MLP]
                        → predicted activation (896-dim)

    The base LLM is frozen: only the MLP head parameters are updated
    during training, which keeps memory usage and training time low.

    Mean-pooling (weighted by attention mask) is more robust than
    last-token pooling when sequence lengths vary, because pad tokens
    don't contribute to the pooled representation.
    """

    def __init__(self, base_model_name: str, hidden_dim: int):
        super().__init__()

        # Frozen LM backbone — used as a rich text encoder.
        self.base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            output_hidden_states=True,
        )
        for p in self.base.parameters():
            p.requires_grad = False

        # Trainable MLP projection head.
        # hidden_dim → hidden_dim (with bottleneck and non-linearity).
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
        """
        Args:
            input_ids:      (B, T) token IDs of generated text.
            attention_mask: (B, T) 1 for real tokens, 0 for padding.

        Returns:
            h_hat: (B, hidden_dim) predicted activation vector.
        """
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Last layer hidden states: (B, T, D)
        last_hidden = out.hidden_states[-1]

        # Mean-pool over real (non-padding) token positions.
        # mask_expanded: (B, T, 1) — broadcasts over hidden_dim.
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (last_hidden * mask_expanded).sum(dim=1)
        pooled = pooled / mask_expanded.sum(dim=1).clamp(min=1e-9)  # (B, D)

        return self.head(pooled)


# ================================================================== #
# 3.  SFT of the Activation Verbalizer
# ================================================================== #

def sft_av_lm(
    base_model_name: str,
    summaries: list,
    output_dir: str = "av_sft",
    epochs: int = 2,
    batch_size: int = 4,
    lr: float = 1e-5,
    device: str = "cuda",
):
    """
    Fine-tune a copy of the base LLM on synthetic activation summaries.

    The model learns to generate text of the form:
        "<|activation|> Summary: ..."
    This warm-start means the AV already knows the expected output format
    before RL training begins, which dramatically stabilises early training.

    Args:
        base_model_name: HuggingFace model identifier.
        summaries:       List of strings (one per training sample).
        output_dir:      Where to save the fine-tuned model and tokeniser.
        epochs:          Number of training epochs.
        batch_size:      Batch size for DataLoader.
        lr:              Learning rate for AdamW.
        device:          "cuda" or "cpu".

    Returns:
        output_dir (str) — path where model was saved.
    """
    print(f"\n{'='*50}")
    print("Step A: SFT of Activation Verbalizer")
    print(f"{'='*50}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Record vocab size BEFORE adding new tokens so we can correctly
    # initialise their embeddings from the existing vocabulary mean.
    original_vocab_size = len(tokenizer)

    # Add our two special tokens in a single call.
    # add_special_tokens returns the count of newly added tokens.
    num_added = tokenizer.add_special_tokens({
        "additional_special_tokens": ["<|activation|>"],
        "pad_token": "<|pad|>",
    })
    print(f"Added {num_added} token(s). "
          f"Vocab: {original_vocab_size} → {len(tokenizer)}")

    # Load model in float32 for training stability.
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float32
    ).to(device)

    # Expand the embedding matrix to cover the new tokens.
    model.resize_token_embeddings(len(tokenizer))

    # Initialise new token embeddings to the mean of the original vocab.
    # Random initialisation can cause large gradient updates early on.
    with torch.no_grad():
        embed_weight = model.get_input_embeddings().weight
        # Mean computed only over original tokens, not the new ones.
        mean_emb = embed_weight[:original_vocab_size].mean(dim=0)
        for i in range(1, num_added + 1):
            embed_weight[-i] = mean_emb
        print(f"Initialised {num_added} new embedding(s) to vocab mean.")

    model.train()

    # Tokenise all summaries at once with padding/truncation.
    enc = tokenizer(
        summaries,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    input_ids      = enc.input_ids
    attention_mask = enc.attention_mask

    # Labels: same as input_ids but with -100 at pad positions.
    # PyTorch cross-entropy ignores -100, so the model is not penalised
    # for not predicting pad tokens.
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt     = AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss  = 0.0
        num_batches = 0

        for ids, mask, lbls in loader:
            ids, mask, lbls = ids.to(device), mask.to(device), lbls.to(device)

            outputs = model(input_ids=ids, attention_mask=mask, labels=lbls)
            loss    = outputs.loss

            if torch.isnan(loss):
                print("  Warning: NaN loss — skipping batch.")
                continue

            opt.zero_grad()
            loss.backward()
            # Gradient clipping prevents large updates from destabilising
            # the pretrained weights early in fine-tuning.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            opt.step()

            total_loss  += loss.item()
            num_batches += 1

        avg = total_loss / num_batches if num_batches > 0 else float("nan")
        print(f"  Epoch {epoch+1}/{epochs} — loss: {avg:.4f}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"AV saved to {output_dir}/")
    return output_dir


# ================================================================== #
# 4.  Train the Activation Reconstructor
# ================================================================== #

def train_ar_head(
    base_model_name: str,
    data_path: str,
    tokenizer,
    output_ar_head: str = "ar_head.bin",
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: str = "cuda",
):
    """
    Train the AR's MLP projection head to reconstruct activation vectors
    from the hidden states of the frozen base LLM.

    For each saved activation, we:
      1. Decode the original input_ids back to a summary string.
      2. Tokenise that string and run it through the frozen AR base.
      3. Mean-pool the last hidden layer.
      4. Pass through the MLP head to predict the activation.
      5. Minimise MSE between predicted and true activation.

    This gives the AR a strong initialisation before RL training.

    Args:
        base_model_name: HuggingFace model identifier for the AR backbone.
        data_path:       Path to activations.pt saved by data_pipeline.py.
        tokenizer:       Tokeniser from the SFT AV (has the special tokens).
        output_ar_head:  Where to save the trained AR head state dict.
        epochs:          Training epochs.
        batch_size:      DataLoader batch size.
        lr:              Learning rate (higher than AV — head only).
        device:          "cuda" or "cpu".
    """
    print(f"\n{'='*50}")
    print("Step B: Training Activation Reconstructor head")
    print(f"{'='*50}")

    data = torch.load(data_path)

    # Build (text, target_activation) pairs.
    summaries = []
    targets   = []
    for item in data:
        text = tokenizer.decode(item["input_ids"], skip_special_tokens=True)[:100]
        summaries.append(f"<|activation|> Summary: {text}")
        targets.append(item["activation"])

    enc = tokenizer(
        summaries,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    input_ids = enc.input_ids
    mask      = enc.attention_mask
    targets   = torch.stack(targets)  # (N, hidden_dim)

    # Determine hidden_dim from model config without loading weights.
    from transformers import AutoConfig
    cfg        = AutoConfig.from_pretrained(base_model_name)
    hidden_dim = cfg.hidden_size

    ar_model  = ARModel(base_model_name, hidden_dim).to(device)
    dataset_t = torch.utils.data.TensorDataset(input_ids, mask, targets)
    loader    = DataLoader(dataset_t, batch_size=batch_size, shuffle=True)

    # Only optimise the MLP head — base is frozen.
    opt = AdamW(ar_model.head.parameters(), lr=lr)
    mse = nn.MSELoss()

    for epoch in range(epochs):
        total_loss  = 0.0
        num_batches = 0

        for ids, m, t in loader:
            ids, m, t = ids.to(device), m.to(device), t.to(device)

            pred = ar_model(ids, m)
            loss = mse(pred, t)

            if torch.isnan(loss):
                print("  Warning: NaN AR loss — skipping batch.")
                continue

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss  += loss.item()
            num_batches += 1

        avg = total_loss / num_batches if num_batches > 0 else float("nan")
        print(f"  Epoch {epoch+1}/{epochs} — MSE: {avg:.4f}")

    # Save only the MLP head weights — the frozen backbone is not saved
    # (it can be reloaded from HuggingFace at inference time).
    torch.save(ar_model.head.state_dict(), output_ar_head)
    print(f"AR head saved to {output_ar_head}")


# ================================================================== #
# Entry point
# ================================================================== #

if __name__ == "__main__":
    BASE_MODEL = "Qwen/Qwen2.5-0.5B"
    DATA_PATH  = "data/layer_24/activations.pt"

    # Load base tokeniser to decode saved input_ids into summary strings.
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    all_data       = torch.load(DATA_PATH)

    summaries = [
        generate_fake_summary(d["input_ids"], base_tokenizer)
        for d in all_data
    ]

    # A. Fine-tune the AV.
    sft_av_lm(BASE_MODEL, summaries, output_dir="av_sft_l24", epochs=2)

    # B. Train the AR head.
    #    Use the SFT tokeniser (which has the special tokens) for the AR too.
    sft_tokenizer = AutoTokenizer.from_pretrained("av_sft_l24")
    train_ar_head(
        BASE_MODEL,
        DATA_PATH,
        sft_tokenizer,
        output_ar_head="ar_head_l24.bin",
        epochs=3,
    )