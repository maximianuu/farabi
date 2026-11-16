"""Chat format: Ollama-style messages -> the model's prompt; model output -> thinking / text / tool call.

The prompt is built with the model's own chat template (chat_template.jinja). Same rules as Ollama's
Qwen 3.8 renderer: multiple system messages are merged into a single leading block, images come
before the message text, and past assistant messages are written with an empty thinking block.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

# Thinking effort used when `think: true` arrives: "medium" (no directive; Ollama's behaviour) or
# "xhigh" (the template's "think carefully, verify assumptions" directive). Selected with FARABI_EFFORT.
DEFAULT_EFFORT = os.environ.get("FARABI_EFFORT", "medium")

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
TOOL_OPEN, TOOL_CLOSE = "<tool_call>", "</tool_call>"


def flatten(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], list, List[str]]:
    """Returns: (messages to pass to the template, images (PIL), image digests)."""
    system, remaining = [], []
    for m in messages:
        if m.get("role") in ("system", "developer"):
            if (m.get("content") or "").strip():
                system.append(m["content"].strip())
        else:
            remaining.append(m)
    outputs: List[Dict[str, Any]] = []
    if system:
        outputs.append({"role": "system", "content": "\n\n".join(system)})
    image_list, digests = [], []
    for m in remaining:
        role = m.get("role")
        txt = m.get("content") or ""
        if role == "user" and m.get("images"):
            from PIL import Image

            chunks = []
            for b64 in m["images"]:
                raw = base64.b64decode(b64)
                image_list.append(Image.open(io.BytesIO(raw)).convert("RGB"))
                digests.append(hashlib.sha1(raw).hexdigest())
                chunks.append({"type": "image"})
            chunks.append({"type": "text", "text": txt})
            outputs.append({"role": "user", "content": chunks})
        elif role == "assistant":
            d: Dict[str, Any] = {"role": "assistant", "content": txt}
            if m.get("tool_calls"):
                d["tool_calls"] = [
                    {"function": {"name": (c.get("function") or c).get("name", ""),
                                  "arguments": _to_dict((c.get("function") or c).get("arguments"))}}
                    for c in m["tool_calls"]
                ]
            outputs.append(d)
        elif role == "tool":
            outputs.append({"role": "tool", "content": txt})
        else:
            outputs.append({"role": "user", "content": txt})
    return outputs, image_list, digests


def _to_dict(a) -> dict:
    if isinstance(a, dict):
        return a
    if isinstance(a, str) and a.strip():
        try:
            v = json.loads(a)
            return v if isinstance(v, dict) else {}
        except Exception:
            return {}
    return {}


def build_prompt(tok, messages, tool_list, think: bool) -> str:
    kw: Dict[str, Any] = dict(add_generation_prompt=True, enable_thinking=bool(think), tokenize=False)
    # Thinking effort. If no effort is given, the template puts the "xhigh" directive (~40 tokens) at the start of the system block.
    # For Ollama `think: true` it does NOT add a directive (measured 2026-11-10: the same request was 42 tokens longer in the engine, and the engine
    # picked tools more often). A plain "on" therefore means "medium" (no directive) to match; if an effort name is given, that one applies.
    # Side benefit: thinking and non-thinking chats share the same system block (cache).
    if think:
        kw["reasoning_effort"] = think if isinstance(think, str) and think in ("low", "medium", "xhigh") else ("xhigh" if think == "high" else DEFAULT_EFFORT)
    if tool_list:
        kw["tools"] = tool_list
    return tok.apply_chat_template(messages, **kw)


# ---------------------------------------------------------------- output parsing


class Parser:
    """Splits streamed text into events: ("thinking", text) · ("text", text) · ("tool", {name, arguments})."""

    def __init__(self, think: bool, tool_list: Optional[List[dict]] = None):
        self.phase = "thinking" if think else "text"
        self.buffer = ""
        self.text_started = False
        self.schemas: Dict[str, Dict[str, Any]] = {}
        for a in tool_list or []:
            f = a.get("function") or a
            self.schemas[f.get("name", "")] = ((f.get("parameters") or {}).get("properties") or {})

    def feed(self, chunk: str) -> List[Tuple[str, Any]]:
        self.buffer += chunk
        return self._process(False)

    def wrap_up(self) -> List[Tuple[str, Any]]:
        return self._process(True)

    @staticmethod
    def _partial_tag(s: str, tags) -> int:
        """Length of a partial opening of a tag at the end of s, if any."""
        en = 0
        for e in tags:
            for k in range(min(len(e) - 1, len(s)), 0, -1):
                if s.endswith(e[:k]):
                    en = max(en, k)
                    break
        return en

    def _process(self, latest: bool) -> List[Tuple[str, Any]]:
        event: List[Tuple[str, Any]] = []
        while True:
            t = self.buffer
            if self.phase == "thinking":
                i = t.find(THINK_CLOSE)
                if i >= 0:
                    if t[:i].strip() or event:
                        event.append(("thinking", t[:i].rstrip("\n")))
                    self.buffer = t[i + len(THINK_CLOSE):]
                    self.phase = "text"
                    continue
                hold = 0 if latest else self._partial_tag(t, (THINK_CLOSE,))
                if len(t) - hold > 0:
                    event.append(("thinking", t[: len(t) - hold]))
                    self.buffer = t[len(t) - hold:]
                break
            if self.phase == "text":
                ia, id_ = t.find(TOOL_OPEN), t.find(THINK_OPEN)
                if id_ >= 0 and (ia < 0 or id_ < ia):
                    self._txt(event, t[:id_])
                    self.buffer = t[id_ + len(THINK_OPEN):]
                    self.phase = "thinking"
                    continue
                if ia >= 0:
                    self._txt(event, t[:ia].rstrip())
                    self.buffer = t[ia + len(TOOL_OPEN):]
                    self.phase = "tool"
                    continue
                hold = 0 if latest else self._partial_tag(t, (TOOL_OPEN, THINK_OPEN))
                # keep whitespace before a tool call out of the text: hold back trailing whitespace
                body = t[: len(t) - hold]
                if not latest:
                    trimmed = body.rstrip()
                    hold += len(body) - len(trimmed)
                    body = trimmed
                self._txt(event, body)
                self.buffer = t[len(t) - hold:] if hold else ""
                break
            # mode == "tool"
            i = t.find(TOOL_CLOSE)
            if i >= 0 or latest:
                body = t[:i] if i >= 0 else t
                call = self._parse_tool(body)
                if call:
                    event.append(("tool", call))
                self.buffer = t[i + len(TOOL_CLOSE):] if i >= 0 else ""
                self.phase = "text"
                if i >= 0:
                    continue
            break
        return [o for o in event if o[1]]

    def _txt(self, event, s: str):
        if not self.text_started:
            s = s.lstrip("\n")
            if not s:
                return
            self.text_started = True
        if s:
            event.append(("text", s))

    def _parse_tool(self, body: str) -> Optional[dict]:
        m = re.search(r"<function=([^>\n]+)>", body)
        if not m:
            return None
        name = m.group(1).strip()
        schema = self.schemas.get(name, {})
        arg: Dict[str, Any] = {}
        for pm in re.finditer(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", body[m.end():], re.S):
            entry_key, raw = pm.group(1).strip(), pm.group(2)
            arg[entry_key] = _coerce(raw, schema.get(entry_key))
        return {"function": {"name": name, "arguments": arg}}


def _coerce(raw: str, schema: Optional[dict]):
    kind = (schema or {}).get("type")
    if isinstance(kind, list):
        kind = next((x for x in kind if x != "null"), None)
    if kind == "string":
        return raw
    s = raw.strip()
    if kind in ("integer", "number", "boolean", "array", "object") or kind is None:
        try:
            v = json.loads(s)
            if kind is None and isinstance(v, str):
                return raw
            if kind == "integer" and isinstance(v, float) and v.is_integer():
                return int(v)
            return v
        except Exception:
            if kind == "boolean":
                return s.lower() in ("true", "1", "on", "yes")
            if kind in ("integer", "number"):
                try:
                    return int(s) if kind == "integer" else float(s)
                except Exception:
                    return raw
    return raw
