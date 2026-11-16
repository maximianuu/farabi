# Third-party software, models and acknowledgements

## Libraries (not bundled; installed from `requirements.txt`)

| Library | License |
|---|---|
| [MLX](https://github.com/ml-explore/mlx) | MIT |
| [mlx-lm](https://github.com/ml-explore/mlx-lm) | MIT |
| [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) | MIT |
| [llguidance](https://github.com/guidance-ai/llguidance) | MIT |
| [transformers](https://github.com/huggingface/transformers), [tokenizers](https://github.com/huggingface/tokenizers), [safetensors](https://github.com/huggingface/safetensors) | Apache 2.0 |
| [NumPy](https://numpy.org) | BSD 3-Clause |
| [Pillow](https://python-pillow.org) | MIT-CMU |

## Models (separate downloads)

| Model | Source | License |
|---|---|---|
| Qwen 3.8 27B and its MTP head | `Qwen/Qwen3.8-27B` | Apache 2.0 |
| Gemma 4 31B and its assistant drafter | `google/gemma-4-31B-it`, `google/gemma-4-31B-it-assistant` | Apache 2.0 |
| bge-m3 | `BAAI/bge-m3`, MLX conversion `mlx-community/bge-m3-mlx-fp16` | MIT |

The quantized weights described in `models/` are modified versions of the Qwen and Gemma weights; each model card states what was changed.

## Acknowledgements

- The speculative-decoding loop for Qwen (draft with the model's own multi-token-prediction head, verify in one pass, roll back) follows
  the design of the MLX runner in [Ollama](https://github.com/ollama/ollama) (MIT). The code here is an independent implementation and
  uses neither Ollama's code nor its model files.
- Model architectures, the vision towers and the Gemma assistant drafter come from mlx-lm and mlx-vlm.
- The HTTP API mirrors a subset of the Ollama API so that existing clients work unchanged.

Farabi is not affiliated with or endorsed by Alibaba Cloud, Google, Apple, Ollama or Hugging Face.
