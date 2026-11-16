"""Image input.

The vision tower (a 27-block ViT + merger) and image preprocessing are taken from mlx-vlm (MIT);
the weights are in the main model's own files (`vision_tower.*`, bf16).
The language model and decoding stay in our core: here only the prompt's embeddings
(with image patches substituted in) and the position numbers are prepared.

Position: image tokens get a 3-dimensional position (time, row, column); the text after the image
continues from "largest position + 1". So with an image prompt the text
position differs from its index in the sequence by `shift`.
"""

from __future__ import annotations

import json
import os
from typing import List, Tuple

import mlx.core as mx
import numpy as np


_PROCESSOR: dict = {}       # model path -> (signature, image processor): stays in the process even after the model is unloaded


class VisionEncoder:
    def __init__(self, model_path: str):
        from mlx_vlm.models.qwen3_5.config import VisionConfig
        from mlx_vlm.models.qwen3_5.language import Qwen3_5RotaryEmbedding
        from mlx_vlm.models.qwen3_5.vision import VisionModel
        from mlx_vlm.utils import load_processor

        cfg = json.load(open(os.path.join(model_path, "config.json")))
        tc = cfg["text_config"]
        self.image_tok = cfg["image_token_id"]
        self.video_tok = cfg["video_token_id"]
        self.start_tok = cfg["vision_start_token_id"]
        self.merge = cfg["vision_config"]["spatial_merge_size"]

        self.tower = VisionModel(VisionConfig.from_dict(cfg["vision_config"]))
        mapping = json.load(open(os.path.join(model_path, "model.safetensors.index.json")))["weight_map"]
        on = "vision_tower."
        weights = {}
        for chunk in sorted({f for k, f in mapping.items() if k.startswith(on)}):
            for k, v in mx.load(os.path.join(model_path, chunk)).items():
                if k.startswith(on):
                    weights[k[len(on):]] = v
        self.tower.load_weights(list(weights.items()), strict=True)
        mx.eval(self.tower.parameters())
        self.tower.eval()

        from pathlib import Path

        # the processor (with another tokenizer inside) is built once per process: see core._PERSISTENT
        from .core import config_signature

        signature = config_signature(model_path)
        k = _PROCESSOR.get(model_path)
        if os.environ.get("FARABI_QWEN_PERSIST", "1") == "0" or k is None or k[0] != signature:
            k = (signature, load_processor(Path(model_path)))
            _PROCESSOR.clear()
            _PROCESSOR[model_path] = k
        self.processor = k[1]
        self.feature_cache = {}      # (image digest, grid) -> tower output
        rp = tc["rope_parameters"]
        self.mrope = Qwen3_5RotaryEmbedding(
            int(tc["head_dim"] * rp["partial_rotary_factor"]),
            max_position_embeddings=tc["max_position_embeddings"],
            base=rp["rope_theta"],
            mrope_section=rp["mrope_section"],
            rope_parameters=rp,
        )

    # -- position numbers (the one-line, image-only form of mlx-vlm LanguageModel.get_rope_index)
    def _location(self, ids: List[int], grid: List[List[int]]) -> Tuple[np.ndarray, int]:
        chunks = []
        st, following = 0, 0   # place in the sequence, next position number
        for t, h, w in grid:
            ed = ids.index(self.image_tok, st)
            gh, gw = h // self.merge, w // self.merge
            txt = ed - st
            chunks.append(np.tile(np.arange(txt) + following, (3, 1)))
            base_dir = following + txt
            ti = np.repeat(np.arange(t), gh * gw)
            hi = np.tile(np.repeat(np.arange(gh), gw), t)
            wi = np.tile(np.arange(gw), t * gh)
            chunks.append(np.stack([ti, hi, wi]) + base_dir)
            st = ed + t * gh * gw
            following = int(chunks[-1].max()) + 1
        if st < len(ids):
            chunks.append(np.tile(np.arange(len(ids) - st) + following, (3, 1)))
        location = np.concatenate(chunks, axis=1).astype(np.int32)
        assert location.shape[1] == len(ids), (location.shape, len(ids))
        return location, int(location.max()) + 1 - len(ids)

    def to_tokens(self, prompt: str, image_list: list) -> dict:
        """Turns the prompt into tokens (each image takes up as many slots as its patch count).

        Returned dict: ids, position (3,1,n), shift, places (the first token position
        of each image), chunks (for each image (patch slice, grid, token count)).
        The vision tower does NOT run here; `embed` runs it once the part to continue from the cache is known.
        """
        from mlx_vlm import prepare_inputs

        g = prepare_inputs(self.processor, images=image_list, prompts=prompt)
        ids = g["input_ids"][0].tolist()
        grid = g["image_grid_thw"].tolist()
        places, chunks, row, inter = [], [], 0, 0
        for t, h, w in grid:
            place = ids.index(self.image_tok, inter)
            n_tok = t * (h // self.merge) * (w // self.merge)
            places.append(place)
            chunks.append((row, row + t * h * w, [t, h, w], n_tok))
            row += t * h * w
            inter = place + n_tok
        location, shift = self._location(ids, grid)
        return {"ids": ids, "position": mx.array(location)[:, None, :], "offset": shift, "positions": places,
                "parts": chunks, "pixels": g["pixel_values"]}

    def embed(self, b: dict, digests: List[str], embed, begin: int = 0) -> mx.array:
        """Embeddings (1,n,H); only images after position `begin` go through the tower."""
        embedding = embed(mx.array(b["ids"])[None])
        kind = self.tower.patch_embed.proj.weight.dtype
        for place, (a, e, grid, n_tok), digest in zip(b["positions"], b["parts"], digests):
            if place < begin:
                continue
            entry_key = (digest, tuple(grid))
            oz = self.feature_cache.get(entry_key)
            if oz is None:
                oz, _ = self.tower(b["pixels"][a:e].astype(kind), mx.array([grid]))
                mx.eval(oz)
                self.feature_cache[entry_key] = oz
                while len(self.feature_cache) > 24:
                    self.feature_cache.pop(next(iter(self.feature_cache)))
            if oz.shape[0] != n_tok:
                raise ValueError(f"image tokens {n_tok} <-> features {oz.shape[0]}")
            embedding[0, place : place + n_tok] = oz.astype(embedding.dtype)
        mx.eval(embedding)
        return embedding

    def prepare(self, prompt: str, image_list: list, embed) -> Tuple[List[int], mx.array, mx.array, int]:
        """In one go: (tokens, embeddings (1,n,H), position (3,1,n), shift)."""
        b = self.to_tokens(prompt, image_list)
        embedding = self.embed(b, [str(i) + ":" + str(id(r)) for i, r in enumerate(image_list)], embed)
        return b["ids"], embedding, b["position"], b["offset"]
