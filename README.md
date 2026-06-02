# Reimplementing Natural Language Autoencoders on Qwen2.5-0.5B

A reimplementation of the [Natural Language Autoencoder (NLA) approach](https://transformer-circuits.pub/2026/nla/index.html) from Anthropic, applied to Qwen/Qwen2.5-0.5B across three representative layers. All experiments run on a single free-tier Colab T4 GPU.

---

## Motivation and What This Is Trying to Do

Modern language models are largely black boxes. We know they work, but we have little visibility into what is actually happening inside them. One way to probe this is to look at the **residual stream** — the running vector representation that accumulates information as a token passes through successive transformer layers. At any layer, this vector is a high-dimensional snapshot of "what the model knows" about the input so far.

The Anthropic paper asks: can we translate these internal snapshots into plain English, then translate that English back into the original snapshot with low error? If yes, we have a form of unsupervised interpretability. The round-trip quality is measured by **Fraction of Variance Explained (FVE)**:

```
FVE = 1 − (avg_MSE / global_var)
```

FVE near 1 means the text-based round-trip almost perfectly reconstructs the original activation. FVE near 0 means it adds no information beyond guessing the mean. Anthropic reports 0.6–0.8 on Claude models.

---

## Model Choice and Why

I chose **Qwen/Qwen2.5-0.5B** — 494M parameters, 24 transformer layers, 896-dimensional hidden state.

- Small enough to train multiple configurations on a free T4 GPU within a few hours per run.
- Recent enough (released late 2024) to have well-structured residual stream representations.
- The 896-dim hidden state is large enough to carry semantic content while staying tractable for a frozen-backbone AR.
- Uses rotary positional embeddings and grouped query attention — architectural choices closer to modern frontier models than older alternatives like GPT-2.

I deliberately avoided larger models. The point is to understand the methodology and find interesting results, not to match the paper's numbers by brute force.

---

## Architecture and Training Pipeline

The pipeline has four stages. Full code is in the linked scripts.

### Stage 1: Extracting Activations ([`data_pipeline.py`](data_pipeline.py))

I feed 2,000 examples from the C4 validation split through the frozen Qwen model and save the **last-token hidden state** at a chosen layer. The last-token position is used because causal attention means it has attended to every preceding token — it is the most information-dense position in the sequence.

One important decision: I L2-normalise each activation vector to unit norm before saving. This keeps the global variance `global_var` in a meaningful range (~0.5–0.7) regardless of which layer is being extracted. Without normalisation, deeper layers produce larger-magnitude vectors that inflate `global_var` to ~300, making FVE essentially meaningless as a metric (any small absolute MSE looks like perfect reconstruction when divided by 300).

The scale factor saved alongside activations is set to 1.0 because vectors are already unit norm. During injection, the activation is divided by this scale before being written into the embedding matrix — ensuring the injected vector has magnitude compatible with normal token embeddings.

### Stage 2: SFT Warm-Start ([`generate_sft.py`](generate_sft.py))

The Activation Verbalizer (AV) starts as a copy of the Qwen base model. Before RL training, it is supervised fine-tuned on synthetic summaries of the form:

```
<|activation|> Summary: {first 100 characters of the C4 input text}
```

This gives the AV a warm start: before RL begins, it already knows the expected output format and the distribution of plausible summaries. Without this, early RL training is chaotic because the model has no idea what it is supposed to produce.

We add two special tokens: `<|activation|>` marks the position where the actual activation vector is injected, and `<|pad|>` is the padding token. Both are initialised to the mean of the existing vocabulary embeddings to avoid gradient instability from random initialisation.

The **Activation Reconstructor (AR)** is also initialised here. It uses a frozen copy of the Qwen base model as a text encoder — mean-pooled last-layer hidden states — feeding into a trainable 2-layer MLP projection head (Linear → GELU → Linear). Only the ~1.6M-parameter MLP head is trained during this stage. The frozen backbone already knows how to read language; we are just teaching it to map language representations to activation space.

This is a deliberate simplification from the Anthropic paper, which fine-tunes the full AR model. I chose the frozen-backbone approach for two reasons: it is much faster to train on limited compute, and it still gives the AR substantially more expressive capacity than a single linear layer, since the frozen backbone uses full causal self-attention across the generated sequence rather than operating on a single pooled vector.

### Stage 3: RL Training ([`train_nla.py`](train_nla.py))

The core training loop uses **REINFORCE** (the policy gradient algorithm). For each batch of activations:

1. The AV generates a token sequence conditioned on the injected activation (stochastic sampling during training).
2. The AR reads that sequence and reconstructs a predicted activation.
3. Reconstruction quality becomes the reward:

```
reward = −log(MSE(AR(z), h) + ε)
```

The log scaling is important — it gives large gradient signal early in training when MSE is high, and gradually flattens as the system improves. Without log scaling, the reward signal is too weak in early epochs.

Four penalty terms discourage degenerate outputs:
- **Length penalty (−5.0)**: fires if the generated sequence is shorter than 8 tokens, preventing immediate EOS collapse.
- **Repetition penalty (continuous)**: `−4.0 × max(0, 0.6 − uniqueness_ratio)` — penalises repetitive outputs proportionally to severity rather than as a hard threshold.
- **Whitespace penalty (−8.0)**: fires when more than 30% of tokens are whitespace characters, targeting the tab/newline collapse mode seen in early training runs.
- **Diversity bonus**: rewards sequences where unique token ratio exceeds 0.4.

A **KL divergence penalty** against the frozen SFT reference model prevents the AV from drifting too far from coherent language:

```
loss_AV = pg_loss + 0.2 × KL(AV || SFT_reference)
```

This KL constraint — borrowed directly from RLHF training for ChatGPT — is essential. Without it, the AV finds degenerate text sequences that minimise reconstruction error but carry no semantic content. With `kl_coeff = 0.2` (raised from an initial 0.05 after observing collapse), the AV stays anchored to the SFT distribution while still being updated toward high-reward outputs.

Training uses early stopping with `patience=2` — halts when validation FVE has not improved for two consecutive epochs.

### Stage 4: Evaluation ([`evaluate.py`](evaluate.py))

Evaluation uses greedy (deterministic) generation rather than sampling, for reproducibility. For each of 200 held-out activations, the AV generates text, the AR reconstructs the activation, and per-sample FVE is computed. Three figures are produced automatically: FVE distribution histogram, training curve, and (for the layer sweep) FVE-by-layer plot.

---

## Results

I tested three layers — 8, 16, and 24 — representing early, middle, and final positions in the 24-layer network. The results tell a clear story about how the nature of representations changes with depth, and what that means for verbalisability.

### Layer 16: Real Verbalisation

**Global FVE: 0.906 | global_var: 0.606 | avg MSE: 0.057**

| Stat | Value |
|---|---|
| FVE min | −0.284 |
| FVE p25 | 0.894 |
| FVE median | 0.942 |
| FVE p75 | 0.949 |
| FVE max | 0.965 |

![FVE Distribution Layer 16](results/layer_16/fve_distribution.png)

Layer 16 is where the results are most meaningful. Here is a selection of the actual generated outputs:

> *"Summary: The 2018-19 school year has come to an end. While there have been some good practices and some worrisome issues, the 2017-18 school year has been a good one for many schools."* — FVE 0.945

> *"Summary: The 2010 presidential race between Barack Obama and John McCain was the first time in US history that a 24-year-old 4-term president was challenged..."* — FVE 0.914

> *"Summary: The 2010 Mid Term Budget of the Government of India 2010-11 (Himachal Pradesh, Uttarakhand, Jammu & Kashmir...)"* — FVE 0.951

> *"Summary: The 2010 MidCamp is the largest in-house training and development event in the country. As a result of the 2009 MidCamp, 2010 MidCamp will be a 3-day event."* — FVE 0.953

These outputs share a recognisable template — educational/institutional summaries, often involving the year 2010 — but they differ in topic, entity, and detail across inputs. The model is not generating the same text every time. Something about the activation content at layer 16 is influencing what gets produced.

A critical caveat: these outputs are not semantically faithful descriptions of the C4 inputs that produced the activations. The AV hallucinates specific details (the "Vikings", the "MidCamp event", Barack Obama) that may have nothing to do with the actual source text. This is expected — the SFT warm-start trained on synthetic summaries rather than real ones, so the AV learned a distribution of plausible-sounding summaries rather than a faithful description mechanism. The Anthropic paper addresses this with better SFT supervision; we did not have the data pipeline to replicate that within compute budget.

The failure cases at layer 16 are also informative:

> *"super fun fun fun fun fun fun fun..."* — FVE 0.525  
> *"Summary: 1.1.1.1.1.1.1.1.1.1.1.1.1.1.1..."* — FVE 0.849  
> *"'S''''''''''''',,''''''''..."* — FVE 0.504  

These represent roughly 10–15% of layer 16 outputs — cases where the repetition penalty was not strong enough and the model collapsed to a degenerate fallback. The visible lower tail in the FVE distribution (down to −0.28 in the worst case) corresponds to these samples. The negative FVE means the AR's reconstruction for those inputs was actively worse than predicting the dataset mean — a signal that the generated text encoded nothing useful.

![Training Curve Layer 16](results/layer_16/training_curve.png)

The training curve shows FVE reaching 0.982 at epoch 1 and 0.998 at epoch 2 before the session was interrupted. The reported evaluation FVE (0.906) is lower than the training validation FVE because evaluation uses greedy decoding, which is more conservative than the stochastic sampling used during validation. This gap between training FVE and evaluation FVE is expected and healthy — it means the model is not purely overfitting to the stochastic sampling regime.

---

### Layer 8: Early Layer — High Variance, Low Verbalisability

**Global FVE: 0.967 | global_var: 0.684 | avg MSE: 0.023**

![FVE Distribution Layer 8](results/layer_8/fve_distribution.png)

Layer 8 achieves a higher FVE than layer 16 — but every single generated output looks like this:

> `"0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0."`  
> `"| = 2 +0. | = 2 +0. | = 2 +0. | = 2 +0."`  
> `"0.0......................................................."`

These are mathematically structured numeric patterns. The AV has collapsed to generating a consistent template that it has learned maps to a vector reliably close to the target. Yet FVE is 0.967. How?

The answer lies in what layer 8 representations actually encode. Layer 8 is in the lower third of a 24-layer network. At this depth, the residual stream primarily captures **syntactic and positional features**: token identities, local dependency patterns, part-of-speech information, relative positions. This information is highly variable across different inputs — which is why `global_var` is at its highest here (0.684) — but it is also fundamentally non-semantic. There is no natural-language description of "this is the 47th token in a sentence beginning with a determiner followed by a noun phrase." These features are real and informative for the model's processing, but they resist verbalisation.

The AV cannot produce meaningful text capturing these features, so it finds a fixed output pattern that the AR maps to a vector in the expected neighbourhood of layer-8 activations. Because layer-8 representations cluster in a particular region of the unit sphere after L2 normalisation, this works well enough to achieve high FVE — but it is not verbalisation, it is metric gaming.

The uniqueness ratios confirm this: virtually every layer-8 output has `unique_ratio = 0.05–0.12`. The repetition penalty fires, but the reconstruction reward at layer 8 (very low MSE) is large enough to overcome it.

---

### Layer 24: Final Layer — Low Variance, Trivial Reconstruction

**Global FVE: 0.994 | global_var: 0.468 | avg MSE: 0.003**

![FVE Distribution Layer 24](results/layer_24/fve_distribution.png)

Layer 24 produces the highest FVE — 0.994, nearly perfect — but with completely degenerate text:

> `"Summary 1000 1000 1000 1000000000000000000000000..."`  
> `"= 100 100 100 100 1000 100 1000 1000 1000..."`  
> `"Summary, 1, 1, 10, 100000000000000000000000000..."`

The FVE is not just high — it is also remarkably **tight**: the minimum is 0.946 and the maximum is 0.996, a range of only 0.05. This unimodal, narrow distribution is a completely different shape from layers 8 and 16.

This is the most interesting result of the three, and it reveals something real about final-layer representations. Layer 24 is the last transformer layer before the language model head. Its job is to transform the accumulated representation into something that directly predicts the next-token probability distribution. As a result, final-layer activations are **geometrically clustered** — many different inputs end up in the same neighbourhood of representation space, because they all need to produce reasonable next-token distributions. The `global_var` at layer 24 (0.468) is the lowest of the three layers tested, confirming this clustering.

When variance is this low, the FVE formula is easily inflated. Even an AR that always predicts a fixed vector near the cluster centre will achieve high FVE, because the distance from any individual activation to the cluster centre is small relative to the total spread. The AV exploits this: it outputs numeric sequences that the AR maps to a fixed cluster-centre prediction, achieving near-zero MSE on every input. The near-constant FVE across all 200 samples (tight distribution) is the smoking gun — real verbalisation would show variance in FVE, since some inputs are harder to describe than others.

This finding mirrors a well-known result in representation learning and probing studies. Final-layer representations in causal language models are "collapsed" toward output space — they have been optimised to predict the next token, not to carry general semantic information that can be read back out. The information you lose by passing through the final few layers is exactly the semantic richness that makes verbalisations meaningful.

---

## Layer Comparison and the Verbalisability Sweet Spot

| Layer | global_var | FVE | Text quality | Interpretation |
|---|---|---|---|---|
| 8 | 0.684 | 0.967 | Degenerate numeric | High variance but syntactic only — not verbalisable |
| 16 | 0.606 | 0.906 | Genuine diverse text | Semantic representations — verbalisable |
| 24 | 0.468 | 0.994 | Degenerate numeric | Low-variance cluster — trivially reconstructable |

The pattern is clear: **middle layers are the sweet spot for natural language verbalisability**. This aligns with extensive prior work in mechanistic interpretability (e.g., probing studies on BERT and GPT-2) that finds middle layers encode the richest semantic content. Early layers encode syntax and tokens; late layers encode output distributions. It is only in the middle that representations carry the kind of structured semantic content — topics, entities, discourse roles — that a language model can describe in words.

The FVE metric alone cannot distinguish between these cases. A high FVE at layer 24 does not mean the system is doing useful interpretability work — it means the system found a shortcut through the low-variance geometry of final-layer space. This is the most practically significant finding from this project: **FVE must be paired with a qualitative check on the diversity and semantic content of generated text** before it can be interpreted as evidence of genuine verbalisation.

---

## Why Our Numbers Differ from the Anthropic Paper

The paper reports FVE of 0.6–0.8 on Claude models. Several factors explain the difference:

**Model scale.** Qwen2.5-0.5B has 494M parameters. Claude models have hundreds of billions. Larger models have higher-dimensional, more spread-out representation spaces. Our 896-dimensional hidden space has less room for diversity than Claude's, making activations easier to reconstruct from partial text information and inflating FVE relative to what the paper reports.

**Synthetic SFT.** The paper's SFT trains the AV on actual input contexts as supervision. We used synthetic summaries (first 100 characters of input text). This trains the AV to produce a narrow "Summary: educational-institutional-content" distribution regardless of actual input, which both causes partial mode collapse and reduces semantic faithfulness.

**Generation length.** 60 tokens versus approximately 200 in the paper. Longer text gives the AR significantly more signal to reconstruct from, which raises FVE honestly while simultaneously making the degenerate-short-sequence strategy less rewarding.

**AR capacity.** We use a frozen backbone with only the 1.6M-parameter MLP head trained. The paper fine-tunes the full AR, which sets a higher ceiling and makes degenerate shortcuts less effective.

**KL coefficient.** We used `kl_coeff = 0.2`. A stronger penalty would more effectively prevent the numeric-pattern collapse observed at layers 8 and 24.

---

## What Remains Uncertain

The most significant open question is whether the layer-16 verbalisations are genuinely informative or are drawing on statistical correlations in the C4 training distribution. The AV consistently generates educational and institutional summaries regardless of actual input — plausibly because these templates map to activation regions that also happen to be near many C4 layer-16 activations (the internet contains a lot of educational content, and Qwen was trained on it).

Distinguishing genuine verbalisation from statistical correlation requires a secondary evaluation: verify whether the generated text actually describes the original input using semantic similarity or human judgement. The Anthropic paper does this; we did not have the infrastructure within our compute budget. We cannot claim layer-16 results constitute genuine interpretable explanations — only that the reconstruction metric is satisfied.

---

## Reproducing Results

```bash
pip install -r requirements.txt
```

Each layer runs through the same four functions with the layer number substituted. For layer 16:

```python
extract_activations(layer_idx=16, output_dir="data/layer_16")
sft_av_lm(BASE_MODEL, summaries, output_dir="av_sft_l16", epochs=2)
train_ar_head(BASE_MODEL, data_path, sft_tokenizer, output_ar_head="ar_head_l16.bin")
train_nla(layer_idx=16, av_path="av_sft_l16", kl_coeff=0.2, output_dir="nla_trained_l16")
results = run_evaluation(av_path="nla_trained_l16/best_av_final",
                         ar_path="nla_trained_l16/best_ar_final.bin")
```

Evaluation results for all three layers are in [`results/`](results/). Model weights are not committed (too large) but reproducible with the above. Each full layer run takes approximately 3–4 hours on a free Colab T4 GPU.

---

## Summary

| | |
|---|---|
| Model | Qwen/Qwen2.5-0.5B |
| Layers tested | 8, 16, 24 |
| Best FVE (honest) | Layer 16: 0.906 with genuine text |
| Most interesting finding | FVE inflation from mode collapse at final layers; middle layers are the verbalisability sweet spot |
| Key limitation | No semantic faithfulness evaluation; SFT uses synthetic rather than real summaries |
