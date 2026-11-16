"""JSON schema enforcement (llguidance) - together with speculative decoding.

The schema is applied by masking the tokens allowed at each step. Draft tokens are proposed
without a mask; the main model's verification rows are masked according to the schema state at that row
(if a draft violates the schema, it does not match the main model's choice and is rejected).
This way the output always conforms to the schema, and matching drafts still give a speedup.
"""

from __future__ import annotations

import json
from typing import List

import mlx.core as mx
import numpy as np

def close_schema(s, _top=True):
    """CLOSES every object that defines `properties` in the JSON schema without specifying `additionalProperties` (fields not in the schema cannot be written).

    JSON Schema's own default is open (extra fields allowed); Ollama's grammar, however, treats such an object as closed, and all
    callers (the batch pipeline, the client app) wrote their schemas for Ollama. Left open, after the last required field the model can
    open a new key and write free text into it without stopping (2026-11-12: 12,721 and 28,646 tokens in an image-judging prompt on Gemma; a Qwen
    classification prompt had a made-up extra field in 13/13 replies). Left untouched: free objects without `properties` ({"type": "object"}, "json" mode),
    objects with `additionalProperties` / `patternProperties` / `unevaluatedProperties` written explicitly, the top level of `allOf` members
    (closing them separately would make them mutually exclusive). The input is not modified."""
    if isinstance(s, list):
        return [close_schema(x) for x in s]
    if not isinstance(s, dict):
        return s
    y = {}
    for k, v in s.items():
        if k in ("properties", "$defs", "definitions", "patternProperties", "dependentSchemas") and isinstance(v, dict):
            y[k] = {name: close_schema(under) for name, under in v.items()}
        elif k in ("items", "additionalProperties", "additionalItems", "contains", "propertyNames", "unevaluatedProperties", "if", "then", "else"):
            y[k] = close_schema(v)
        elif k in ("anyOf", "oneOf", "prefixItems") and isinstance(v, list):
            y[k] = [close_schema(x) for x in v]
        elif k == "allOf" and isinstance(v, list):
            y[k] = [close_schema(x, False) for x in v]
        else:
            y[k] = v                               # type, required, enum, const, default, examples, not ...: as they are
    if (_top and isinstance(y.get("properties"), dict) and y["properties"]
            and not any(a in y for a in ("additionalProperties", "patternProperties", "unevaluatedProperties"))):
        y["additionalProperties"] = False
    return y


class Constraint:
    def __init__(self, hf_tokenizer, schema):
        import llguidance as llg
        import llguidance.hf

        # 2026-11-12 (0.4.2): the matcher's tokenizer is stored on the HF object ITSELF. It used to live in a dict keyed by id(object):
        # after a model was unloaded and the object died, its address could be given to another family's tokenizer, which would then be masked with the wrong dict (latent bug).
        t = getattr(hf_tokenizer, "_farabi_llg", None)
        if t is None:
            t = llguidance.hf.from_tokenizer(hf_tokenizer)
            try:
                hf_tokenizer._farabi_llg = t
            except Exception:
                pass                                  # if the attribute cannot be written, it is rebuilt every time (slow but correct)
        # 2026-11-12 (0.4.2): objects with `properties` are CLOSED by default (Ollama behaviour; see close_schema). FARABI_SCHEMA_OPEN=1 returns to the old
        # behaviour (JSON Schema's own default: open).
        import os

        if os.environ.get("FARABI_SCHEMA_OPEN") != "1":
            try:
                schema = close_schema(json.loads(schema) if isinstance(schema, str) else schema)
            except ValueError:
                pass                                  # unresolvable string: let the compiler report its own error
        txt = schema if isinstance(schema, str) else json.dumps(schema)
        # 2026-11-10 (measured in the image test): forcing a compact format ("{"a": "b"}" + no whitespace) pushes the model into json.dumps
        # mode and it writes non-ASCII letters as \\uXXXX escapes; since generic escapes were forbidden, the codes also came out corrupted
        # ("Çınar" -> "Ç\\u0013nar"). Like Ollama, we allow whitespace (at most 40 characters per gap, so there is no endless whitespace
        # loop) and enable generic escapes so that the model can write the correct code if it does emit an escape.
        grammar = llg.JsonCompiler(whitespace_flexible=True, whitespace_pattern=r"[\x20\x0A]{1,40}",
                                  json_allow_general_unicode_escapes=True).compile(txt)
        self.es = llg.LLMatcher(t, grammar)
        self.words = t.vocab_size
        self._error()

    def _error(self):
        h = self.es.get_error()
        if h:
            raise ValueError(f"schema matcher error: {h}")

    def _bitmask(self, row: int) -> np.ndarray:
        import llguidance.numpy as N

        return np.full((row, (self.words + 31) // 32), -1, dtype=np.int32), N

    def apply(self, logit: mx.array) -> mx.array:
        """(1,V) logits -> masked for the current state."""
        from mlx_vlm.structured import _apply_llguidance_mask

        bm, N = self._bitmask(1)
        N.fill_next_token_bitmask(self.es, bm, 0)
        return _apply_llguidance_mask(logit, mx.array(bm))

    def apply_rows(self, logit: mx.array, drafts: List[int]) -> mx.array:
        """(d+1,V) verification logits: row i, with the mask of the state after the first i drafts were consumed."""
        from mlx_vlm.structured import _apply_llguidance_mask

        n = len(drafts) + 1
        bm, N = self._bitmask(n)
        consumed = 0
        for i in range(n):
            N.fill_next_token_bitmask(self.es, bm, i)
            if i < n - 1:
                if self.es.is_stopped() or self.es.validate_tokens([drafts[i]]) != 1:
                    break                      # draft violates the schema: later rows are unreachable anyway
                self.es.consume_token(drafts[i])
                consumed += 1
        if consumed:
            self.es.rollback(consumed)
        self._error()
        return _apply_llguidance_mask(logit, mx.array(bm))

    def mask_needed_rows(self, drafts: List[int], candidates) -> dict:
        """Fast path (2026-11-11): if the UNMASKED candidates of the verification rows conform to the schema, there is no need to compute a mask.

        candidates: candidate token ids per row (row i = state after the first i drafts were consumed). If all candidates of a row are valid, that row's
        masked distribution is IDENTICAL to its unmasked one (the mask only removes invalid tokens). Returns: {row: bit mask (1, V/32)} - only for rows
        with at least one invalid candidate; the caller masks those rows and recomputes them. In free-text fields (inside a JSON string) almost no rows are returned.
        """
        needed = {}
        n = len(drafts) + 1
        consumed = 0
        for i in range(n):
            if self.es.is_stopped():
                break
            if any(self.es.validate_tokens([int(a)]) != 1 for a in candidates[i]):
                bm, N = self._bitmask(1)
                N.fill_next_token_bitmask(self.es, bm, 0)
                needed[i] = bm
            if i < n - 1:
                if self.es.validate_tokens([drafts[i]]) != 1:
                    break                      # draft violates the schema: rejected on this row, later ones are unreachable
                self.es.consume_token(drafts[i])
                consumed += 1
        if consumed:
            self.es.rollback(consumed)
        self._error()
        return needed

    def apply_mask(self, logit: mx.array, bm) -> mx.array:
        from mlx_vlm.structured import _apply_llguidance_mask

        return _apply_llguidance_mask(logit, mx.array(bm))

    def mandatory(self) -> List[int]:
        """The token sequence for which the schema leaves exactly ONE option in the current state (e.g. `{"` followed by a field name: `ay ni _ ol ay`). Empty list if none.

        llguidance's "fast-forward" tokens: since the text is forced, no model decision is needed; the caller feeds them into the cache
        in a single forward pass without spending a draft/verification round (engine/gemma.py generate). If the end of the sequence could merge with a longer token, the library does not return it (safe).
        """
        if self.es.is_stopped():
            return []
        return [int(x) for x in self.es.compute_ff_tokens()]

    def finished(self) -> bool:
        """Whether the schema is complete (only the end marker remains)."""
        return bool(self.es.is_stopped())

    def consume(self, tokens: List[int], finish=()) -> int:
        """Feeds tokens into the schema; stops at the end marker or when the schema completes. Returns: the number of tokens processed."""
        n = 0
        for b in tokens:
            if b in finish or self.es.is_stopped():
                break
            self.es.consume_token(b)
            n += 1
        self._error()
        return n

    def consume_and_cut(self, tokens: List[int], finish=()) -> int:
        """Feeds a draft round's output into the schema; if the schema completed in the MIDDLE of the round, what follows MUST BE DISCARDED. Returns: the number of tokens to keep.

        2026-11-13 (0.4.3): on Gemma's fast path (mask_needed_rows), verification rows AFTER the schema completes are not masked; if the main model
        also picked the same continuation as the draft (such as ` (` after `}`), those tokens were accepted and ended up in the reply: `{ "answer": "Ankara" } (`.
        In an overnight test, 156 of 1,676 schema-constrained Gemma replies (9.3%); 0 on Ollama's Gemma and on the Qwen path (apply_rows masks that row).
        If what comes after the closing is the end marker, it is left alone (normal ending, identical to the old behaviour)."""
        n = self.consume(tokens, finish)
        if 0 < n < len(tokens) and tokens[n] not in finish and self.es.is_stopped():
            return n
        return len(tokens)
