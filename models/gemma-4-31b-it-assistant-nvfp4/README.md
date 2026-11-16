---
license: apache-2.0
base_model: google/gemma-4-31B-it-assistant
library_name: mlx
tags:
- mlx
- nvfp4
- quantized
- farabi
---

# Gemma 4 31B assistant drafter — nvfp4 4-bit for the Farabi engine

A 4-bit `nvfp4` quantization (about 0.3 GB, down from 0.9 GB) of
[google/gemma-4-31B-it-assistant](https://huggingface.co/google/gemma-4-31B-it-assistant), the small draft model that proposes tokens for
Gemma 4 31B during speculative decoding in the [Farabi engine](https://github.com/maximianuu/farabi).

**What was changed (Apache 2.0, section 4b):** linear weights quantized to 4-bit `nvfp4` with power-of-two shifts and least-squares
block scales (`tools/gemma_draft.py` in the engine repository). No calibration data, no fine-tuning.

A draft model only affects speed: the main model verifies every proposed token.

License: Apache 2.0, same as the original model. Unofficial conversion.
