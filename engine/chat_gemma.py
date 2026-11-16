"""Gemma 4 chat format: Ollama-style messages -> prompt; model output -> thinking / text / tool call.

The prompt is built with the model's own template (chat_template.jinja, Google's 2026-07-09 version). Format:
  <|turn>role\\n ... <turn|>\\n                       turn
  <|channel>thought\\n ... <channel|>                 thinking (when off, the prompt ends with an empty thinking block)
  <|tool_call>call:name{key:value,...}<tool_call|>    tool call; strings go between <|"|>...<|"|>
  <|tool_response>response:name{...}<tool_response|>  tool response (after a call the model emits <|tool_response> and stops)
When thinking is on, the system block starts with <|think|> -> thinking and non-thinking chats are separate prefixes in the cache.
"""

from __future__ import annotations

import base64
import hashlib
import io
from typing import Any, Dict, List, Optional, Tuple

from .chat import _coerce, _to_dict

THINK_OPEN, THINK_CLOSE = "<|channel>thought\n", "<channel|>"
CHANNEL_OPEN = "<|channel>"
TOOL_OPEN, TOOL_CLOSE = "<|tool_call>", "<tool_call|>"
QUOTE = '<|"|>'
FINISH_TOKENS = ("<turn|>", "<eos>", "<|tool_response>")   # special tokens that end generation


def flatten(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], list, List[str]]:
    """Returns: (messages to pass to the template, images (PIL), image digests).

    System messages are merged into a single leading block; tool calls get sequential ids so that the template
    writes the tool responses that follow with the right tool name (Ollama messages may have no ids).
    """
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
    pending: List[Tuple[str, str]] = []      # (id, name): calls whose response is expected
    counter = 0
    for m in remaining:
        role = m.get("role")
        txt = m.get("content") or ""
        if role == "assistant":
            d: Dict[str, Any] = {"role": "assistant", "content": txt}
            if m.get("thinking"):
                d["reasoning"] = m["thinking"]
            pending = []
            if m.get("tool_calls"):
                d["tool_calls"] = []
                for c in m["tool_calls"]:
                    f = c.get("function") or c
                    ident = c.get("id") or f"c{counter}"
                    counter += 1
                    d["tool_calls"].append({"id": ident, "type": "function", "function": {"name": f.get("name", ""), "arguments": _to_dict(f.get("arguments"))}})
                    pending.append((ident, f.get("name", "")))
            outputs.append(d)
        elif role == "tool":
            d = {"role": "tool", "content": txt}
            name = m.get("tool_name") or m.get("name")
            ident = m.get("tool_call_id")
            if not ident and pending:
                es = next((b for b in pending if name and b[1] == name), pending[0])
                pending.remove(es)
                ident, name = es[0], name or es[1]
            if ident:
                d["tool_call_id"] = ident
            if name:
                d["name"] = name
            outputs.append(d)
        else:
            pending = []
            if m.get("images"):
                from PIL import Image

                chunks = []
                for b64 in m["images"]:
                    raw = base64.b64decode(b64)
                    image_list.append(Image.open(io.BytesIO(raw)).convert("RGB"))
                    digests.append(hashlib.sha1(raw).hexdigest())
                    chunks.append({"type": "image"})
                chunks.append({"type": "text", "text": txt})
                outputs.append({"role": "user", "content": chunks})
            else:
                outputs.append({"role": "user", "content": txt})
    return outputs, image_list, digests


def build_prompt(tok, messages, tool_list, think: bool) -> str:
    kw: Dict[str, Any] = dict(add_generation_prompt=True, enable_thinking=bool(think), tokenize=False)
    if tool_list:
        kw["tools"] = tool_list
    return tok.apply_chat_template(messages, **kw)


def starts_thinking(prompt: str) -> bool:
    """If thinking is on after a tool response, the template opens the thinking channel itself; the output starts from inside the thinking."""
    return prompt.endswith(THINK_OPEN)


# ---------------------------------------------------------------- tool arguments


class _Read:
    """Small recursive parser for the `{key:value,...}` body (strings are <|"|>...<|"|>)."""

    def __init__(self, s: str):
        self.s, self.i = s, 0

    def whitespace(self):
        while self.i < len(self.s) and self.s[self.i] in " \n\t\r":
            self.i += 1

    def value(self):
        self.whitespace()
        s, i = self.s, self.i
        if s.startswith(QUOTE, i):
            j = s.find(QUOTE, i + len(QUOTE))
            if j < 0:
                j = len(s)
            self.i = j + len(QUOTE)
            return s[i + len(QUOTE): j]
        if i < len(s) and s[i] == '"':                     # if the model uses plain quotes
            j = i + 1
            while j < len(s) and s[j] != '"':
                j += 2 if s[j] == "\\" else 1
            self.i = min(j + 1, len(s))
            return s[i + 1: j]
        if i < len(s) and s[i] == "{":
            return self.obj()
        if i < len(s) and s[i] == "[":
            self.i += 1
            arr = []
            while True:
                self.whitespace()
                if self.i >= len(s):
                    break
                if s[self.i] == "]":
                    self.i += 1
                    break
                arr.append(self.value())
                self.whitespace()
                if self.i < len(s) and s[self.i] == ",":
                    self.i += 1
            return arr
        j = i
        while j < len(s) and s[j] not in ",}]":
            j += 1
        self.i = j
        raw = s[i:j].strip()
        if raw == "true":
            return True
        if raw == "false":
            return False
        if raw in ("null", "None", ""):
            return None
        try:
            return int(raw)
        except ValueError:
            try:
                return float(raw)
            except ValueError:
                return raw

    def obj(self) -> dict:
        s = self.s
        self.whitespace()
        if self.i < len(s) and s[self.i] == "{":
            self.i += 1
        d: Dict[str, Any] = {}
        while True:
            self.whitespace()
            if self.i >= len(s):
                break
            if s[self.i] == "}":
                self.i += 1
                break
            if s.startswith(QUOTE, self.i) or s[self.i] == '"':
                entry_key = self.value()
            else:
                j = self.i
                while j < len(s) and s[j] not in ":,}":
                    j += 1
                entry_key, self.i = s[self.i:j].strip(), j
            self.whitespace()
            if self.i < len(s) and s[self.i] == ":":
                self.i += 1
                d[str(entry_key)] = self.value()
            elif entry_key:
                d[str(entry_key)] = None
            self.whitespace()
            if self.i < len(s) and s[self.i] == ",":
                self.i += 1
        return d


def parse_tool_call(body: str, schemas: Optional[Dict[str, Dict[str, Any]]] = None) -> Optional[dict]:
    """`call:name{...}` -> {"function": {"name", "arguments"}}; fixes types according to the schema."""
    g = body.strip()
    if g.startswith("call:"):
        g = g[5:]
    k = g.find("{")
    name = (g if k < 0 else g[:k]).strip()
    if not name:
        return None
    arg = _Read(g[k:]).obj() if k >= 0 else {}
    schema = (schemas or {}).get(name, {})
    for entry_key, v in list(arg.items()):
        kind = (schema.get(entry_key) or {}).get("type")
        if isinstance(kind, list):
            kind = next((x for x in kind if x != "null"), None)
        if kind == "string" and not isinstance(v, str) and v is not None:
            arg[entry_key] = str(v).lower() if isinstance(v, bool) else str(v)
        elif kind in ("integer", "number", "boolean", "array", "object") and isinstance(v, str):
            arg[entry_key] = _coerce(v, schema.get(entry_key))
    return {"function": {"name": name, "arguments": arg}}


# ---------------------------------------------------------------- output parsing


class Parser:
    """Splits streamed text into events: ("thinking", text) · ("text", text) · ("tool", {function: {name, arguments}})."""

    def __init__(self, think: bool, tool_list: Optional[List[dict]] = None, thinking_on: bool = False):
        del think                                  # Gemma opens thinking itself; only the start of 'thinking_on' is affected
        self.phase = "thinking" if thinking_on else "text"
        self.buffer = ""
        self.text_started = False
        self.think_started = False
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
                    self._think(event, t[:i].rstrip("\n"))
                    self.buffer = t[i + len(THINK_CLOSE):]
                    self.phase = "text"
                    continue
                hold = 0 if latest else self._partial_tag(t, (THINK_CLOSE,))
                self._think(event, t[: len(t) - hold])
                self.buffer = t[len(t) - hold:] if hold else ""
                break
            if self.phase == "text":
                ia, id_ = t.find(TOOL_OPEN), t.find(CHANNEL_OPEN)
                if id_ >= 0 and (ia < 0 or id_ < ia):
                    # the channel name runs to the end of the line ("thought\n"); if the whole name has not arrived, wait
                    j = t.find("\n", id_)
                    if j < 0 and not latest:
                        self._txt(event, t[:id_])
                        self.buffer = t[id_:]
                        break
                    self._txt(event, t[:id_])
                    self.buffer = t[j + 1:] if j >= 0 else ""
                    self.phase = "thinking"
                    continue
                if ia >= 0:
                    self._txt(event, t[:ia].rstrip())
                    self.buffer = t[ia + len(TOOL_OPEN):]
                    self.phase = "tool"
                    continue
                hold = 0 if latest else self._partial_tag(t, (TOOL_OPEN, CHANNEL_OPEN))
                body = t[: len(t) - hold]
                if not latest:                        # keep whitespace before a tool call out of the text
                    trimmed = body.rstrip()
                    hold += len(body) - len(trimmed)
                    body = trimmed
                self._txt(event, body)
                self.buffer = t[len(t) - hold:] if hold else ""
                break
            # mode == "tool"
            i = t.find(TOOL_CLOSE)
            if i >= 0 or latest:
                call = parse_tool_call(t[:i] if i >= 0 else t, self.schemas)
                if call:
                    event.append(("tool", call))
                self.buffer = t[i + len(TOOL_CLOSE):] if i >= 0 else ""
                self.phase = "text"
                if i >= 0:
                    continue
            break
        return [o for o in event if o[1]]

    def _think(self, event, s: str):
        if not self.think_started:
            s = s.lstrip("\n")
            if not s:
                return
            self.think_started = True
        if s:
            event.append(("thinking", s))

    def _txt(self, event, s: str):
        if not self.text_started:
            s = s.lstrip("\n")
            if not s:
                return
            self.text_started = True
        if s:
            event.append(("text", s))
