"""
data_pipeline.py
================
Step 1 of the NLA pipeline.

Loads a pretrained target LLM (Qwen/Qwen2.5-0.5B by default), feeds it
real web text from the C4 validation split, and saves the residual-stream
activation vector at a chosen layer for every sample.

These activation vectors become the training targets for the NLA:
  - The AV must learn to generate text that *describes* each vector.
  - The AR must learn to reconstruct each vector from that text.

Key design choices vs the Anthropic paper
------------------------------------------
* We use Qwen2.5-0.5B (494M params) instead of Claude models. This is a
  deliberate simplification: the model is small enough to run on a single
  free-tier GPU while still producing rich residual-stream representations.
* We extract the last-token hidden state at layer 16 (of 24). Layer 16 is
  in the upper-middle of the network, where representations are typically
  semantic rather than purely syntactic. We run a layer sweep in evaluate.py
  to show how FVE varies with layer depth.
* We do NOT L2-normalise activations before saving. The raw magnitudes are
  preserved so that global_var in FVE is meaningful. A separate scale factor
  (p75 of raw norms) is saved and used during injection to prevent the
  injected vector from dominating the embedding space.

Outputs
-------
data/activations.pt  — list of dicts: {"input_ids": Tensor, "activation": Tensor}
data/scale.txt       — single float: 75th-percentile norm across all samples
"""

import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_activations(
    model_name="Qwen/Qwen2.5-0.5B",
    layer_idx=24,
    max_samples=2000,
    max_length=128,
    output_dir="data/layer_24"
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
    )
    model.eval()

    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    data = []
    raw_norms = []
    count = 0

    for example in dataset:
        text = example["text"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        if inputs["input_ids"].size(1) < 10:
            continue

        with torch.no_grad():
            outputs = model(**inputs.to(model.device))
            hidden = outputs.hidden_states[layer_idx]
            h = hidden[0, -1, :].cpu().float()

        norm = torch.norm(h).item()
        raw_norms.append(norm)

        # FIX: L2-normalise to unit norm so global_var is meaningful (~0.5-1.0)
        h_unit = h / (norm + 1e-8)

        data.append({
            "input_ids": inputs["input_ids"][0].cpu(),
            "activation": h_unit
        })
        count += 1
        if count % 200 == 0:
            print(f"Processed {count}/{max_samples}...")
        if count >= max_samples:
            break

    os.makedirs(output_dir, exist_ok=True)
    torch.save(data, os.path.join(output_dir, "activations.pt"))

    # FIX: scale = 1.0 because vectors are already unit norm
    scale = 1.0
    with open(os.path.join(output_dir, "scale.txt"), "w") as f:
        f.write(str(scale))

    print(f"\nSaved {len(data)} samples to {output_dir}/")
    print(f"Raw norm stats (before normalisation):")
    print(f"  min={min(raw_norms):.2f}  median={np.median(raw_norms):.2f}  "
          f"max={max(raw_norms):.2f}")
    print(f"Scale factor: {scale} (fixed, vectors are unit norm)")
    return data

# Run it
# extract_activations(layer_idx=8, max_samples=2000, output_dir="data/layer_8")


# ------------------------------------------------------------------ #
# Layer sweep helper — call this to generate activations at multiple
# layers for the analysis in evaluate.py.
# ------------------------------------------------------------------ #
def extract_activations_multilayer(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    layer_indices: list = None,
    max_samples: int = 2000,
    max_length: int = 128,
    output_base_dir: str = "data",
):
    """
    Convenience wrapper: extracts activations at each layer in layer_indices
    and saves them to output_base_dir/layer_{idx}/.

    Qwen2.5-0.5B has 24 transformer layers (indices 1–24 in hidden_states).
    Recommended sweep: [6, 12, 16, 20, 24].
    """
    if layer_indices is None:
        layer_indices = [6, 12, 16, 20, 24]

    for idx in layer_indices:
        print(f"\n{'='*50}")
        print(f"Extracting layer {idx} activations...")
        print(f"{'='*50}")
        out_dir = os.path.join(output_base_dir, f"layer_{idx}")
        extract_activations(
            model_name=model_name,
            layer_idx=idx,
            max_samples=max_samples,
            max_length=max_length,
            output_dir=out_dir,
        )
    print("\nLayer sweep complete.")


if __name__ == "__main__":
    # Default run: extract layer-16 activations for 2000 samples.
    extract_activations()

    # Uncomment to run the full layer sweep (takes ~5x longer):
    # extract_activations_multilayer()