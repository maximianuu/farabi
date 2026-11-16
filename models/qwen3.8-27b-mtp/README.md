---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: mlx
tags:
- mlx
- farabi
---

# Qwen3.8-27B — multi-token-prediction head

The `mtp.*` tensors of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), copied unchanged (bf16) from the original
repository's last weight shard into a single file, `mtp.safetensors` (about 0.8 GB). The
[Farabi engine](https://github.com/maximianuu/farabi) uses this head as the draft model for speculative decoding.

**What was changed (Apache 2.0, section 4b):** nothing in the tensor values; they were only extracted into a separate file.

License: Apache 2.0, same as the original model. Unofficial extraction.
