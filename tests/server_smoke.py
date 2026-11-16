"""Tests the server with the request formats sent by the client app.

Usage: tests/server_smoke.py [port]   (the server must be running)
"""
import base64, json, os, sys, time, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 11438
ROOT = f"http://127.0.0.1:{PORT}"
MODEL = "qwen3.8-27b"
elapsed, remaining = 0, 0


def walk(file_path):
    return json.load(urllib.request.urlopen(ROOT + file_path, timeout=30))


def send(file_path, body, streaming=False, timeout_secs=900):
    r = urllib.request.Request(ROOT + file_path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    f = urllib.request.urlopen(r, timeout=timeout_secs)
    if not streaming:
        return json.load(f)
    txt, think, tool_calls, latest, first = "", "", [], None, None
    t = time.time()
    for row in f:
        j = json.loads(row)
        if "error" in j:
            raise RuntimeError(j["error"])
        m = j.get("message", {})
        if first is None and (m.get("content") or m.get("thinking") or m.get("tool_calls")):
            first = time.time() - t
        txt += m.get("content", "")
        think += m.get("thinking", "")
        tool_calls += m.get("tool_calls", [])
        if j.get("done"):
            latest = j
    return {"text": txt, "thinking": think, "tool": tool_calls, "final": latest, "first": first, "duration": time.time() - t}


def check(name, condition, detail=""):
    global elapsed, remaining
    elapsed += bool(condition); remaining += not condition
    print(("✓ " if condition else "✗ ") + name + (f" — {detail}" if detail else ""), flush=True)


def speed(s):
    return s["eval_count"] / max(s["eval_duration"] / 1e9, 1e-9)


SYSTEM = "You are Farabi; a helper assistant running on the user's Mac. Answer briefly and clearly.\n" + "\n".join(
    f"Rule {i}: Do not make up facts in your answers; say when you do not know. Write file paths in full. ({i})" for i in range(1, 60))
TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Reads the contents of a file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Full file path"},
                                                        "line": {"type": "integer", "description": "Maximum lines"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "get_weather", "description": "Gives the weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
]

print("version:", walk("/api/version"), "| modeller:", [m["name"] for m in walk("/api/tags")["models"]])
s = send("/api/show", {"model": MODEL})
check("show: vision + tools + thinking", {"vision", "tools", "thinking"} <= set(s["capabilities"]))
t = time.time(); r = send("/api/generate", {"model": MODEL, "keep_alive": "10m"}); check("load", r.get("done_reason") == "load", f"{time.time()-t:.1f} s")
check("ps: loaded", [m["name"] for m in walk("/api/ps")["models"]] == [MODEL])

# 1) streaming plain chat, thinking off
m1 = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "What is the capital of Turkey? One sentence."}]
r = send("/api/chat", {"model": MODEL, "stream": True, "think": False, "messages": m1, "options": {"temperature": 0.5, "num_ctx": 131072}}, streaming=True)
check("stream: text", "Ankara" in r["text"] and not r["thinking"], f"{r['text'].strip()!r} · first chunk {r['first']:.2f} s · prompt {r['final']['prompt_eval_count']} · cache {r['final']['engine']['cache']}")
n1 = r["final"]["prompt_eval_count"]

# 2) continuation of the same chat → should continue from the cache
m2 = m1 + [{"role": "assistant", "content": r["text"]}, {"role": "user", "content": "And roughly what is its population? One sentence."}]
r2 = send("/api/chat", {"model": MODEL, "stream": True, "think": False, "messages": m2, "options": {"temperature": 0.5}}, streaming=True)
mo = r2["final"]["engine"]
check("continuation: continued from the cache", mo["cache"] in ("continued", "rewind") and mo["read"] < 80, f"cache {mo['cache']} · read {mo['read']}/{r2['final']['prompt_eval_count']} · first chunk {r2['first']:.2f} s · {r2['text'].strip()[:70]!r}")

# 3) NEW chat with the same system prompt → should fork from the root
m3 = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "What is the sum of 2 and 3? Just the number."}]
r3 = send("/api/chat", {"model": MODEL, "stream": True, "think": False, "messages": m3, "options": {"temperature": 0}}, streaming=True)
mo = r3["final"]["engine"]
check("new chat: the system block was not re-read", mo["cache"] == "fork" and mo["read"] < 80 and "5" in r3["text"], f"cache {mo['cache']} · read {mo['read']}/{r3['final']['prompt_eval_count']} · {r3['text'].strip()!r}")

# 4) back to the first chat → still in the cache
m4 = m2 + [{"role": "assistant", "content": r2["text"]}, {"role": "user", "content": "Thanks, that is all."}]
r4 = send("/api/chat", {"model": MODEL, "stream": True, "think": False, "messages": m4, "options": {"temperature": 0.5}}, streaming=True)
mo = r4["final"]["engine"]
check("back to the first chat: in cache", mo["read"] < 80, f"cache {mo['cache']} · read {mo['read']}/{r4['final']['prompt_eval_count']}")

# 5) thinking on
r = send("/api/chat", {"model": MODEL, "stream": True, "think": True, "messages": [{"role": "user", "content": "What is 17 × 23?"}], "options": {"temperature": 0.1}}, streaming=True)
check("thinking: streams separately and the answer is correct", bool(r["thinking"].strip()) and "391" in r["text"] and "</think>" not in r["text"], f"thinking {len(r['thinking'])} chars · {r['text'].strip()[:60]!r} · {speed(r['final']):.0f} tok/s")

# 6) tool call
ma = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Read the first 20 lines of the file /home/user/notes/plan.md."}]
r = send("/api/chat", {"model": MODEL, "stream": True, "think": False, "messages": ma, "tools": TOOLS, "options": {"temperature": 0.2}}, streaming=True)
c = (r["tool"] or [{}])[0].get("function", {})
check("tool call parsed", c.get("name") == "read_file" and c.get("arguments", {}).get("path") == "/home/user/notes/plan.md" and c.get("arguments", {}).get("line") == 20 and "<tool_call>" not in r["text"], f"{json.dumps(r['tool'], ensure_ascii=False)} · text {r['text']!r}")

# 7) tool result → answer (the tool step should continue from the cache)
mb = ma + [{"role": "assistant", "content": r["text"], "tool_calls": r["tool"]}, {"role": "tool", "content": "# Plan\n1. Finish the engine\n2. Connect it to FARABI\n3. Run the tests", "tool_name": "read_file"}]
r = send("/api/chat", {"model": MODEL, "stream": True, "think": False, "messages": mb, "tools": TOOLS, "options": {"temperature": 0.2}}, streaming=True)
mo = r["final"]["engine"]
check("tool result: answer + cache", "FARABI" in r["text"] and not r["tool"] and mo["read"] < 120, f"cache {mo['cache']} · read {mo['read']}/{r['final']['prompt_eval_count']} · {r['text'].strip()[:80]!r}")

# 8) JSON schema (the client app's JSON-query format, non-streaming)
schema = {"type": "object", "properties": {"query": {"type": "string"}, "news": {"type": "boolean"}}, "required": ["query", "news"]}
r = send("/api/chat", {"model": MODEL, "stream": False, "think": False, "format": schema, "keep_alive": "10m",
                         "messages": [{"role": "user", "content": "Chat:\nUser: what is the dollar exchange rate today\n\nWhat should be searched on the web to answer the user's LAST message? JSON only: {\"query\": \"…\", \"news\": true/false}\n\nReturn only JSON."}],
                         "options": {"temperature": 0, "num_predict": 120}})
try:
    o = json.loads(r["message"]["content"]); fine = isinstance(o.get("query"), str) and isinstance(o.get("news"), bool)
except Exception:
    o, fine = r["message"]["content"], False
check("JSON schema", fine, f"{o}")

# 9) image
image = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "test-1.png")
b64 = base64.b64encode(open(image, "rb").read()).decode()
mg = [{"role": "user", "content": "What is the order number in the image? Write only the number.", "images": [b64]}]
r = send("/api/chat", {"model": MODEL, "stream": True, "think": False, "messages": mg, "options": {"temperature": 0}}, streaming=True)
check("image reading", "TR-90315" in r["text"], f"{r['text'].strip()!r} · prompt {r['final']['prompt_eval_count']} · {r['duration']:.1f} s")
mg2 = mg + [{"role": "assistant", "content": r["text"]}, {"role": "user", "content": "What color is the circle in the image? One word."}]
r = send("/api/chat", {"model": MODEL, "stream": True, "think": False, "messages": mg2, "options": {"temperature": 0}}, streaming=True)
mo = r["final"]["engine"]
check("continuation of the image chat", "red" in r["text"].lower() and mo["read"] < 80, f"{r['text'].strip()!r} · cache {mo['cache']} · read {mo['read']}/{r['final']['prompt_eval_count']}")

# 10) speed on a long answer
r = send("/api/chat", {"model": MODEL, "stream": True, "think": False, "messages": [{"role": "user", "content": "Assume Apple has announced its new Mac Studio model with an M5 Ultra processor. Write an 8-paragraph, subheaded, fluent English tech news article about it."}], "options": {"temperature": 0, "num_predict": 500}}, streaming=True)
check("speed (500-token article)", speed(r["final"]) > 45, f"{speed(r['final']):.1f} tok/s · finish {r['final']['done_reason']} · accepted {r['final']['engine']['accepted']}/{r['final']['engine']['draft']}")

# 11) cancel: the client drops the connection, the server must stop and answer the next request right away
import socket
body = json.dumps({"model": MODEL, "stream": True, "think": False, "messages": [{"role": "user", "content": "Write the numbers from 1 to 3000 in words, separated by commas."}], "options": {"temperature": 0}}).encode()
sock = socket.create_connection(("127.0.0.1", PORT)); sock.sendall(b"POST /api/chat HTTP/1.0\r\nContent-Type: application/json\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
sock.recv(4096); time.sleep(1.5); sock.close()
t = time.time(); r = send("/api/chat", {"model": MODEL, "stream": False, "think": False, "messages": [{"role": "user", "content": "Say hello."}], "options": {"temperature": 0, "num_predict": 20}})
check("cancel: the server became idle right away", time.time() - t < 6, f"next request {time.time()-t:.1f} s · {r['message']['content'].strip()!r}")

print(f"\n{elapsed} passed, {remaining} failed")
