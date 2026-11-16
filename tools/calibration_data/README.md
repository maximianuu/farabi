# Calibration data (bring your own)

`tools/calibration.py` reads everything in this folder when the GPTQ pass (`tools/convert4.py`) builds its calibration set:

- `*.json` — chats: each file is a list of `{"role": ..., "content": ...}` messages.
- `system_prompt.txt` — optional long system prompt; one chat behind it is kept as a single long sequence.
- `*.txt`, `*.md`, `*.py`, `*.swift`, `*.c`, `*.rs`, `*.js` — prose and source code (up to four 9,000-character pieces per file).

The folder is empty on purpose. The more the text resembles what you will actually run through the model (languages, prompt length,
tool definitions, output formats), the better the quantized weights track the original model on your workload. Around 50-60k tokens
were enough for the published weights. Do not put anything here that you also use to measure fidelity.
