---
license: apache-2.0
base_model: google/gemma-4-31B-it
library_name: mlx
pipeline_tag: image-text-to-text
tags:
- mlx
- nvfp4
- quantized
- farabi
---

# Gemma 4 31B (it) — nvfp4 4-bit (shift + block scale) for the Farabi engine

A 4-bit `nvfp4` (group size 16) quantization of [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) for Apple Silicon
(MLX), about 17 GB. Text and image input.

**This repository is meant for the [Farabi engine](https://github.com/maximianuu/farabi).** `config.json` carries compensation values
(`farabi_norm_eps_exp`, `farabi_embed_shift`) that the engine applies when loading. Other MLX loaders will read the tensors but will not
apply them, and their output will be wrong.

## What was changed (Apache 2.0, section 4b)

Starting from the original weights:

1. Linear weights and the token embedding table are quantized to 4-bit `nvfp4`.
2. Each matrix is multiplied by a power of two before quantization so that its per-block scales stay in the normal range of the 8-bit
   scale format. Because every projection feeds an RMS norm, the factor cancels there; the affected norms get a correspondingly larger
   epsilon (exponents in `config.json`). Block scales are fitted by least squares instead of max-abs.

No calibration data and no fine-tuning were used. The tokenizer, chat template and processor files are unchanged.

## Why not the stock conversion

With the stock MLX `nvfp4` conversion of this model we observed malformed HTML tags in long generations (attributes invented inside
`<strong>`), which did not occur with an 8-bit conversion or with this one. In a token-level comparison against an 8-bit reference this
conversion measured a KL divergence of about 0.03.

## Use

```bash
hf download maximianuu/gemma-4-31b-it-nvfp4 --local-dir ~/.farabi/models/gemma-4-31b-it-nvfp4
hf download maximianuu/gemma-4-31b-it-assistant-nvfp4 --local-dir ~/.farabi/models/gemma-4-31b-it-assistant-nvfp4
python -m engine.server      # then request the model "gemma4-31b"
```

## License

Apache 2.0, same as the original model. "Gemma" is a name of its owner; this is an unofficial conversion.
