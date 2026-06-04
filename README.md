# Natural Language Autoencoders on Qwen2.5-0.5B

A reimplementation of [Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations](https://transformer-circuits.pub/2026/nla/index.html) (Fraser-Taliente et al., Anthropic, May 2026), applied to Qwen2.5-0.5B on a single free-tier T4 GPU.



## Overview

Modern language models are largely black boxes. We know they work, but we have little visibility into what is actually happening inside them. One way to probe this is to look at the residual stream — the running vector representation that accumulates information as a token passes through successive transformer layers. At any layer, this vector is a high-dimensional snapshot of "what the model knows" about the input so far.

The NLA paper introduces a method for turning a language model's internal activation vectors into natural language  and natural language back into activation vectors. Two components are trained jointly:

- **Activation Verbalizer (AV):** takes a residual-stream activation vector `h`, injected into the embedding of a special token, and generates a free-text description `z`.
- **Activation Reconstructor (AR):** reads `z` and predicts `ĥ`, an approximation of the original activation.

The Anthropic paper asks: can we translate these internal snapshots into plain English, then translate that English back into the original snapshot with low error? If yes, we have a form of unsupervised interpretability. The round-trip quality is measured by Fraction of Variance

Roundtrip quality is measured by **Fraction of Variance Explained (FVE)**:


FVE = 1 - MSE(h, ĥ) / Var[h]


FVE = 0 means the AR is no better than predicting the mean activation. 
FVE = 1 is perfect reconstruction. 
The Anthropic paper reports 0.6-0.8 FVE on Claude models. 
This reimplementation reaches **FVE = 0.9666** on Qwen2.5-0.5B



## Model and Layer

**Target model:** [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) - 494M parameters, 24 transformer layers, 896-dimensional residual stream.

I chose Qwen/Qwen2.5-0.5B	 for three reasons:

-Fits on free hardware - the AV, AR, and a frozen reference copy all run together on a single T4 GPU without running out of memory.
-Small but capable  at 494M parameters it is recent (2024) and strong for its size, making it a genuinely interesting target rather than a trivial one like a character level model.
-Better than GPT-2 - GPT-2 small has a 768-dimensional hidden state and has been studied extensively in interpretability research already. Qwen2.5-0.5B has a 896-dimensional hidden state and is less picked-over, so findings here are more novel.

The paper recommends middle-to-late layers, where representations are semantic rather than syntactic. 

**Activations were extracted from three layers** - 8, 16, and 24 - representing early, middle, and final representations respectively. 

**Layer 8** (of 24) was selected as the activation target which sits at the one-third point of the network and offered better reconstructibility within the available compute budget.

Experiments with layer 24 (the final layer) showed activations that were harder to reconstruct likely because final-layer representations are more entangled with the unembedding head. 
**Data:** 2,000 samples from the [AllenAI C4](https://huggingface.co/datasets/allenai/c4) English validation split, truncated to 128 tokens. 
The last-token hidden state at layer 8 serves as the activation target. All activations are L2-normalised to unit norm before saving, which stabilises FVE computation (global variance ≈ 0.68).



## Architecture and Training Pipeline

The pipeline has four stages. Full code is in the linked scripts.

Stage 1: Extracting Activations (data_pipeline.py)

I feed 2,000 examples from the C4 validation split through the frozen Qwen model and save the last-token hidden state at a chosen layer. The last-token position is used because causal attention means it has attended to every preceding token — it is the most information-dense position in the sequence.

One important decision: I L2-normalise each activation vector to unit norm before saving. This keeps the global variance global_var in a meaningful range (~0.5–0.7) regardless of which layer is being extracted. Without normalisation, deeper layers produce larger-magnitude vectors that inflate global_var to ~300, making FVE essentially meaningless as a metric (any small absolute MSE looks like perfect reconstruction when divided by 300).

The scale factor saved alongside activations is set to 1.0 because vectors are already unit norm. During injection, the activation is divided by this scale before being written into the embedding matrix — ensuring the injected vector has magnitude compatible with normal token embeddings.

Stage 2: SFT Warm-Start (generate_sft.py)

### Activation Verbalizer (AV)

A fine-tuned copy of Qwen2.5-0.5B. A special token `<|activation|>` is added to the vocabulary, with its embedding initialised to the mean of the existing vocabulary (preventing gradient explosions at the start of training). During both training and inference, the activation vector `h` replaces that token's embedding at position 0, and the model autoregressively generates up to 60 tokens.

The paper uses ~200 tokens per explanation on Claude-scale models. At 60 tokens on a T4, memory pressure becomes the binding constraint, this is one of the clearest reasons absolute FVE numbers are not directly comparable to the paper's.

### Activation Reconstructor (AR)

Also based on Qwen2.5-0.5B, but with the backbone **frozen**. Only a small trainable 2-layer MLP head is optimised:


[Frozen Qwen2.5-0.5B backbone] → mean-pooled last hidden state (896-dim)
                               → Linear(896, 896) → GELU → Linear(896, 896)
                               → ĥ (896-dim)


The frozen backbone reads `z` with full causal self-attention, giving a contextually rich representation for the MLP to invert. Mean-pooling over the full generated sequence (weighted by attention mask) is more robust than last-token pooling across variable sequence lengths.

The paper's AR is a truncated version of the target model (first `l` layers) with a learned affine map at the final token. Using a full frozen model with mean-pooling trades some mechanistic fidelity for training stability: freezing the backbone reduces GPU memory by roughly 40% and removes a major source of gradient instability in early training.

Stage 3: RL Training (train_nla.py)

## Training

### Supervised warm-start

Before RL, both AV and AR are initialised with supervised fine-tuning on a proxy task. The proxy target is:


"<|activation|> Summary: {first 100 characters of the input text}"


-The paper warms up the AV using Claude Opus 4.5-generated summaries high-quality prose with a characteristic style of short paragraphs and bold topic headings. That style persists through RL training, acting as a strong prior toward readable outputs. 
-However, since Claude API access was not available, the first 100 characters of the raw input text were used as a substitute. This weaker warm-start meant the AV never learned a consistent readable style ,a limitation that directly contributed to the steganographic collapse observed during RL training.

**SFT results (2 epochs):**
- AV cross-entropy loss: 3.99 → 2.41
- AR MSE: 0.1232 → 0.0242 → 0.0135
- Warm-start Val FVE: **0.3495**

### RL training (REINFORCE + KL penalty)

The AV acts as the policy. For each activation `h`, it samples a token sequence `z`. The AR reconstructs `ĥ` from `z`. 

The reward is:
reward = -log(MSE(ĥ, h) + ε)                              [reconstruction]
       + length penalty    (-5.0 if gen_length < 8)
       + repetition penalty (-4.0 × max(0, 0.6 − unique_token_ratio))
       + whitespace penalty (-8.0 if >30% of tokens are whitespace)


**Reward Function**
In the paper:
-Reconstruction reward: reward = -log(MSE(ĥ, h) + ε)
-KL divergence penalty to keep AV close to SFT reference
-No additional penalties needed at Claude scale

In our implementation:
-Same reconstruction reward and KL penalty as the paper
-Three additional penalties added due to small model collapse:

**Whitespace penalty (-8.0):** If more than 30% of the generated tokens are whitespace characters like tabs or newlines, a heavy penalty of -8.0 is applied. This was the most common failure mode - the AV would output almost nothing but blank space.

**Repetition penalty (-4.0):** If the AV keeps repeating the same token over and over - for example "the the the the..."  a penalty of -4.0 is applied. The penalty scales with how repetitive the output is, so partial repetition is also penalised, not just complete repetition.

**Length penalty (-5.0):** If the output is shorter than 8 tokens, a penalty of -5.0 is applied. This forces the AV to generate a minimum amount of content before stopping.

The AV is updated via REINFORCE: `∇J = E[∇log π(z|h) · R]`. 
A KL divergence penalty toward the frozen SFT reference AV prevents reward hacking. This mirrors the paper's GRPO + KL setup; REINFORCE replaces GRPO to avoid group-sampling overhead, which is prohibitive on a single T4. The AR head is updated simultaneously via supervised MSE regression on the generated sequences.

**Training log:**

| Epoch | Avg Reward | Train MSE | Val FVE   |
|-------|-----------|-----------|---------   |
|    1  |    5.478  | 0.013483  |  0.3495    |
|    2  |    6.992  | 0.001199  |  0.9701    |
|    3  |    7.243  | 0.000811  | **0.9920** |
|    4  |    7.361  | 0.000910  |  0.9920 (no improvement, early stop) |

Training halted at epoch 4 (patience = 2). Best checkpoint from epoch 3.

**Epoch 1 → 2** shows the biggest jump - Val FVE leaps from 0.35 to 0.97 in a single epoch. This is where the steganographic collapse occurred. The AV rapidly discovered that repeating numeric patterns give high reconstruction reward, and the AR co-adapted to decode them. Both models locked into this private encoding within a single training epoch.

**Epoch 2 → 3** shows a smaller improvement - FVE moves from 0.97 to 0.99. At this point the AV and AR are no longer learning anything fundamentally new. They are simply refining their shared numeric encoding to squeeze out marginal reconstruction gains.

**Epoch 4** shows no improvement over epoch 3. Early stopping activates after two epochs without improvement (patience = 2) and training halts. The best checkpoint from epoch 3 is saved and used for all evaluation.

Most importantly- the average reward keeps rising from 5.47 to 7.36 across all four epochs, even after FVE plateaus at 0.99 from epoch 3 onwards. This is a classic sign of reward hacking. The AV is finding ways to increase its reward score without genuinely improving reconstruction quality - further evidence that the reconstruction objective alone, without strong regularisation, does not produce meaningful outputs.

## Stage 4: Evaluation (evaluate.py)

Evaluation uses greedy (deterministic) generation rather than sampling, for reproducibility. For each of 200 held-out activations, the AV generates text, the AR reconstructs the activation, and per-sample FVE is computed. Three figures are produced automatically: FVE distribution histogram, training curve, and (for the layer sweep) FVE-by-layer plot.

## Results

I tested three layers — 8, 16, and 24 — representing early, middle, and final positions in the 24-layer network. The results tell a clear story about how the nature of representations changes with depth, and what that means for verbalisability.

**Layer Comparison**

|Layer |   Position     |  Mean FVE | Degenerate samples |             Observation                      |
|------|----------------|-----------|--------------------|----------------------------------------------|
|  8   |   Early (1/3)  |  0.9666   |       0/200        |Widest FVE spread — most informative          |
|  16  |   Middle (2/3) |  0.906    |       2/200        |Lowest FVE — hardest to reconstruct           |
|  24  |   Final        |  0.994    |       0/200        |Highest FVE — but most uniform and trivial    |


-The results show a U-shaped pattern across layers. 
-Layer 16 (middle) is the hardest to reconstruct with FVE = 0.906, and is the only layer with degenerate samples (2 out of 200). 
-Layer 24 (final) achieves the highest FVE of 0.994 but with zero spread — all 200 samples cluster at exactly FVE ≈ 1.0, meaning the AV found an extremely efficient 
encoding for final-layer activations.
-This U-shape is meaningful. Final-layer activations in Qwen2.5-0.5B appear to have low intrinsic dimensionality — they are geometrically simpler and easier to encode, 
even steganographically. 
-Middle-layer activations carry more complex, higher-dimensional semantic information that is harder to compress into 60 tokens even with numeric patterns. Early-layer activations fall in between.

**Layer 16: Real Verbalisation**

Global FVE: 0.906 | global_var: 0.606 | avg MSE: 0.057

|    Stat     |    Value    |
|-------------|-------------|
|FVE min	    |   −0.284    |
|FVE p25      |    0.894    |
|FVE median   |	0.942       |
|FVE p75      |	0.949       |
|FVE max      |	0.965       |

**FVE Distribution Layer 16**

Layer 16 is where the results are most meaningful. Here is a selection of the actual generated outputs:

``` "Summary: The 2018-19 school year has come to an end. While there have been some good practices and some worrisome issues, the 2017-18 school year has been a good one for many schools." — FVE 0.945 ``` 

``` "Summary: The 2010 presidential race between Barack Obama and John McCain was the first time in US history that a 24-year-old 4-term president was challenged..." — FVE 0.914 ```

``` "Summary: The 2010 Mid Term Budget of the Government of India 2010-11 (Himachal Pradesh, Uttarakhand, Jammu & Kashmir...)" — FVE 0.951 ```

``` "Summary: The 2010 MidCamp is the largest in-house training and development event in the country. As a result of the 2009 MidCamp, 2010 MidCamp will be a 3-day event." — FVE 0.953 ```

These outputs share a recognisable template — educational/institutional summaries, often involving the year 2010 — but they differ in topic, entity, and detail across inputs. The model is not generating the same text every time. Something about the activation content at layer 16 is influencing what gets produced.

A critical caveat: these outputs are not semantically faithful descriptions of the C4 inputs that produced the activations. The AV hallucinates specific details (the "Vikings", the "MidCamp event", Barack Obama) that may have nothing to do with the actual source text. This is expected — the SFT warm-start trained on synthetic summaries rather than real ones, so the AV learned a distribution of plausible-sounding summaries rather than a faithful description mechanism. The Anthropic paper addresses this with better SFT supervision; we did not have the data pipeline to replicate that within compute budget.

The failure cases at layer 16 are also informative:

``` "super fun fun fun fun fun fun fun..." — FVE 0.525 ```

``` "Summary: 1.1.1.1.1.1.1.1.1.1.1.1.1.1.1..." — FVE 0.849 ```

``` "'S''''''''''''',,''''''''..." — FVE 0.50 ```

These represent roughly 10–15% of layer 16 outputs — cases where the repetition penalty was not strong enough and the model collapsed to a degenerate fallback. The visible lower tail in the FVE distribution (down to −0.28 in the worst case) corresponds to these samples. The negative FVE means the AR's reconstruction for those inputs was actively worse than predicting the dataset mean — a signal that the generated text encoded nothing useful.

**Training Curve Layer 16**

The training curve shows FVE reaching 0.982 at epoch 1 and 0.998 at epoch 2 before the session was interrupted. The reported evaluation FVE (0.906) is lower than the training validation FVE because evaluation uses greedy decoding, which is more conservative than the stochastic sampling used during validation. This gap between training FVE and evaluation FVE is expected and healthy — it means the model is not purely overfitting to the stochastic sampling regime.


**Layer 8:** Early Layer — High Variance, Low Verbalisability
Global FVE: 0.967 | global_var: 0.684 | avg MSE: 0.023

FVE Distribution Layer 8:

Layer 8 achieves a higher FVE than layer 16 — but every single generated output looks like this:

```"0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0."```

```"| = 2 +0. | = 2 +0. | = 2 +0. | = 2 +0."```

```"0.0......................................................."```

These are mathematically structured numeric patterns. The AV has collapsed to generating a consistent template that it has learned maps to a vector reliably close to the target. Yet FVE is 0.967. How?

The answer lies in what layer 8 representations actually encode. Layer 8 is in the lower third of a 24-layer network. At this depth, the residual stream primarily captures syntactic and positional features: token identities, local dependency patterns, part-of-speech information, relative positions. This information is highly variable across different inputs — which is why global_var is at its highest here (0.684) — but it is also fundamentally non-semantic. There is no natural-language description of "this is the 47th token in a sentence beginning with a determiner followed by a noun phrase." These features are real and informative for the model's processing, but they resist verbalisation.

The AV cannot produce meaningful text capturing these features, so it finds a fixed output pattern that the AR maps to a vector in the expected neighbourhood of layer-8 activations. Because layer-8 representations cluster in a particular region of the unit sphere after L2 normalisation, this works well enough to achieve high FVE — but it is not verbalisation, it is metric gaming.

The uniqueness ratios confirm this: virtually every layer-8 output has unique_ratio = 0.05–0.12. The repetition penalty fires, but the reconstruction reward at layer 8 (very low MSE) is large enough to overcome it.

**Layer 24:** Final Layer — Low Variance, Trivial Reconstruction
Global FVE: 0.994 | global_var: 0.468 | avg MSE: 0.003

FVE Distribution Layer 24:

Layer 24 produces the highest FVE — 0.994, nearly perfect — but with completely degenerate text:

```"Summary 1000 1000 1000 1000000000000000000000000..."``` 

```"= 100 100 100 100 1000 100 1000 1000 1000..."```

```"Summary, 1, 1, 10, 100000000000000000000000000..."```

The FVE is not just high — it is also remarkably tight: the minimum is 0.946 and the maximum is 0.996, a range of only 0.05. This unimodal, narrow distribution is a completely different shape from layers 8 and 16.

This is the most interesting result of the three, and it reveals something real about final-layer representations. Layer 24 is the last transformer layer before the language model head. Its job is to transform the accumulated representation into something that directly predicts the next-token probability distribution. As a result, final-layer activations are geometrically clustered — many different inputs end up in the same neighbourhood of representation space, because they all need to produce reasonable next-token distributions. The global_var at layer 24 (0.468) is the lowest of the three layers tested, confirming this clustering.

When variance is this low, the FVE formula is easily inflated. Even an AR that always predicts a fixed vector near the cluster centre will achieve high FVE, because the distance from any individual activation to the cluster centre is small relative to the total spread. The AV exploits this: it outputs numeric sequences that the AR maps to a fixed cluster-centre prediction, achieving near-zero MSE on every input. The near-constant FVE across all 200 samples (tight distribution) is the smoking gun — real verbalisation would show variance in FVE, since some inputs are harder to describe than others.

This finding mirrors a well-known result in representation learning and probing studies. Final-layer representations in causal language models are "collapsed" toward output space — they have been optimised to predict the next token, not to carry general semantic information that can be read back out. The information you lose by passing through the final few layers is exactly the semantic richness that makes verbalisations meaningful.

**Layer Comparison and the Verbalisability Sweet Spot**


|Layer  |	global_var |	 FVE	  |    Text quality        |	               Interpretation                        |
|-------|------------|----------|------------------------|-------------------------------------------------------|
|  8	  |   0.684    |	0.967	  |Degenerate numeric      |	High variance but syntactic only — not verbalisable  |
|  16   |	  0.606	   |  0.906	  |Genuine diverse text	   |Semantic representations — verbalisable                |
|  24	  |   0.468    |	0.994   |	Degenerate numeric	   |Low-variance cluster — trivially reconstructable       |

The pattern is clear: middle layers are the sweet spot for natural language verbalisability. This aligns with extensive prior work in mechanistic interpretability (e.g., probing studies on BERT and GPT-2) that finds middle layers encode the richest semantic content. Early layers encode syntax and tokens; late layers encode output distributions. It is only in the middle that representations carry the kind of structured semantic content — topics, entities, discourse roles — that a language model can describe in words.

The FVE metric alone cannot distinguish between these cases. A high FVE at layer 24 does not mean the system is doing useful interpretability work — it means the system found a shortcut through the low-variance geometry of final-layer space. This is the most practically significant finding from this project: FVE must be paired with a qualitative check on the diversity and semantic content of generated text before it can be interpreted as evidence of genuine verbalisation.


**Four contributing factors:**

1. **KL coefficient too low (0.2).** was not strong enough to keep the AV close to its warm-start behaviour. A stronger penalty (0.5-1.0) would have anchored the AV closer to readable outputs.

2. **Weak warm-start prior.** The paper uses Claude-generated summaries to warm-start the AV, which gives it a strong prior toward readable prose. Here, raw text prefixes were used instead. The AV never learned a consistent readable style, leaving it free to drift during RL training.

3. **Short generation length.** At 60 tokens, numeric repetition is an easy and efficient encoding strategy. Longer generations (150+ tokens) would make pure numeric encoding harder to sustain and might push the AV toward more structured prose.

4. **Scale Diffference.** The paper trains on Claude-scale models where basic readability emerges naturally. At 0.5B parameters, Qwen2.5-0.5B has far less expressive capacity, making degenerate shortcuts more attractive under RL pressure.

This is arguably the most informative outcome: **the steganography failure mode is concretely demonstrated and its causal structure is clear.** 

The steganographic collapse demonstrates that reconstruction reward alone is insufficient to produce interpretable outputs. The KL penalty and warm-start quality are the load-bearing components of the paper's method. Without them, the AV and AR will always find a private encoding that maximises reward without being human-readable. This finding is consistent with the broader literature on reward hacking in RL fine-tuning.

**Potential corrective measures**:
-Increase KL coefficient from 0.2 to 0.5-1.0
-Replace raw text prefixes with Claude-generated summaries for warm-start
-Increase generation length from 60 to 150+ tokens

### What the numeric patterns reveal

The outputs are not random - the patterns map to interpretable activation structure:

- **All-zeros** (`"0. 0. 0. 0..."`) - highest FVE (0.978-0.981). Corresponds to unit-sphere activations with many near-zero dimensions.
- **All-ones** (`"1. 1. 1. 1..."`)  - mid-range FVE (~0.97). Likely encodes activations with strong uniform positive components.
- **Alternating `| = 2 +0.`**       - activations with structured geometry, possibly related to position encoding or layer-norm bookkeeping in Qwen's residual stream.
- **Worst reconstructions** (`"0.0......."` with FVE ≈ 0.90) period-runs rather than space-separated digits; the AR struggles with these, suggesting they encode less geometrically structured activations.

The AV has effectively learned a rudimentary floating-point notation. It encodes the dominant magnitude structure of the activation vector but none of its semantic content.

-This confirms that the AV is functioning as a lossy numeric compressor rather than a semantic explainer. 
-It captures geometric structure (magnitude, sparsity) but discards the information that would make the explanation useful to a human - what the model was actually processing, attending to, or predicting. This is precisely what the paper's warm-start and KL regularisation are designed to prevent.


## Why the FVE Exceeds the Paper's

FVE = 0.9666 is higher than the paper's 0.6–0.8. This is not a better result ,it is an unconstrained result.

| Factor                              |                 Effect on FVE                 |
|-------------------------------------|-----------------------------------------------|
| Steganographic encoding             | ↑ (maximises FVE without interpretability)    |
| L2-normalised unit-norm activations | ↑ (lower variance denominator)                |
| 60-token generation                 | ↑ (dense encoding beats prose at short length)|
| Weak KL penalty (0.2)               | ↑ (less constraint on output form)            |

The paper's 0.6–0.8 is *constrained* FVE — reconstruction quality while maintaining human-readable prose. That is the harder target. Reaching 0.97 without the prose constraint is not progress; it is a demonstration of what happens when the constraint is removed.

**Achieving constrained FVE comparable to the paper would require three changes:** 

-A stronger KL penalty (coefficient 0.5 or higher).
-Claude-generated summaries for warm-start instead of raw text prefixes.
-longer generation length (150+ tokens). 

These changes would anchor the AV toward readable prose during RL training and remove the conditions that made steganographic collapse the path of least resistance.


## Limitations and What Remained Out of Reach

**layer sweep.** Layer sweep was limited to three layers. Activations were extracted at layers 8, 16, and 24. A finer-grained sweep across all 24 layers would give a more complete picture of the U-shaped FVE pattern and identify exactly where the transition between easy and hard reconstruction occurs.

**No causal steering.** The paper shows that editing an NLA explanation and reconstructing via the AR produces a steering vector. Steganographic outputs are not meaningfully editable, so this experiment is not applicable here.

**GRPO vs. REINFORCE.** The paper uses GRPO, which normalises rewards group-wise across multiple samples of the same input, reducing gradient variance. REINFORCE with a global baseline has higher variance and likely contributed to the rapid collapse to steganographic solutions in epoch 2.



## Reproducing These Results

All code is in `nla_pipeline.ipynb` as four annotated self-contained cells:

1. **`data_pipeline`**  - extract Qwen2.5-0.5B layer-8 activations from C4 validation
2. **`generate_sft`**   - SFT warm-start for AV; supervised AR head training
3. **`train_nla`**      - REINFORCE RL training with KL penalty (5 epochs, early stopping)
4. **`evaluate`**       - FVE evaluation, distribution plot, training curve, failure mode analysis

**Requirements:** Python 3.10+, PyTorch 2.x, `transformers`, `datasets`, `bitsandbytes`, `matplotlib`. A free Colab T4 GPU suffices. Total runtime: ~4–5 hours end to end.

Trained checkpoints (`nla_trained_l8/best_av_final/`, `nla_trained_l8/best_ar_final.bin`) are not included due to size (~2 GB) but are fully reproducible by running the notebook.



## Conclusion

This experiment produced a concrete and reproducible demonstration of the steganography failure mode. When the KL regularisation is weak and the warm-start prior is 
uninformative, the AV and AR converge on a shared private numeric encoding — token sequences that achieve near-perfect reconstruction (FVE = 0.9666) while being 
completely opaque to any human reader.


The high FVE is not a success, it is evidence of what happens when the readability constraint is removed. The paper's 0.6–0.8 FVE is the harder and more meaningful  
target, achieved while maintaining human-readable prose outputs. This experiment shows that reconstruction fidelity alone is insufficient - the KL penalty and 
warm-start quality are the load-bearing components of the NLA framework.


Achieving constrained FVE comparable to the paper would require three changes: a stronger KL penalty (coefficient 0.5 or higher), Claude-generated summaries for 
warm-start instead of raw text prefixes, and longer generation length (150+ tokens). These changes would anchor the AV toward readable prose during RL training and 
remove the conditions that made steganographic collapse the path of least resistance.

The three-layer comparison revealed a U-shaped FVE pattern - middle layers are hardest to reconstruct even steganographically, suggesting they carry higher-dimensional semantic information than early or final layers.

The logical next step is a rerun with kl_coeff = 1.0-2.0 and Claude or GPT-4-generated warm-start summaries. 
FVE would likely drop to the 0.5-0.7 range - but the outputs would be genuinely interpretable and meaningful, which is the actual goal of the NLA framework.


|      Summary                  |                                                                                                    |
|-------------------------------|----------------------------------------------------------------------------------------------------|
| Model	                        |  Qwen/Qwen2.5-0.5B                                                                                 |
| Layers tested	                |  8, 16, 24                                                                                         |
| Best FVE (honest)	            |   Layer 16: 0.906 with genuine text                                                                |
| Most interesting finding	    |  FVE inflation from mode collapse at final layers; middle layers are the verbalisability sweet spot| 
| Key limitation	              |  No semantic faithfulness evaluation; SFT uses synthetic rather than real summaries                |



*Code: [`nla_pipeline.ipynb`](./nla_pipeline.ipynb) | Data: C4 validation (streaming) | Model: [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) | Compute: Google Colab T4 (free tier)*
