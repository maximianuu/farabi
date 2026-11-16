# Farabi (macOS app)

A small native macOS front-end for the Farabi inference engine: a plain chat window, a model picker, and an optional
background watchdog that keeps the engine running when the app is closed.

## What it does
- **Plain chat:** only the messages you type are sent to the model. No system prompt, no tools, no connectors, no
  generation options (the model's own defaults apply).
- **Model picker (toolbar):** lists the engine's models with size, quantization and disk size; the model in memory gets a
  green dot and its memory footprint. Load a model into memory or unload it again from the menu or the eject button.
- **Saved chats (sidebar):** each chat is a JSON file in `~/Library/Application Support/Farabi/chats/`. Deleting a chat
  moves its file to the Trash. Nothing leaves your Mac.
- **Settings:** interface language (English by default, Turkish selectable) and the *Keep running in the background* switch.
  When on, the app installs a launchd job (`io.github.maximianuu.farabi.watchdog`) that runs `Farabi --watchdog` at login; the
  watchdog starts the engine if it is not running and restarts it if it dies. When off, quitting the app stops the engine
  (unless it is busy generating).
- Logs: `~/Library/Logs/farabi-engine.log` (engine requests), `farabi-engine-process.log` (engine stdout/stderr started by
  the app), `farabi-watchdog.log`.

## How it finds the engine
The engine is a Python package started as `python -m engine.server` with the engine folder as working directory, and it
listens on `127.0.0.1:11438`. Defaults:

| UserDefaults key | Default |
| --- | --- |
| `engineFolder` | `~/farabi` (the cloned repository root) |
| `enginePython` | `~/farabi/.venv/bin/python` |

Override them with `defaults write io.github.maximianuu.Farabi engineFolder /path/to/farabi` (same for `enginePython`; `enginePort` if the engine runs on another port).
The app talks to the engine over its Ollama-compatible HTTP API; `GET /` provides `version`, `busy` and `network_lock`.

## Build
Requires macOS 26+, Apple Silicon, Xcode and [XcodeGen](https://github.com/yonaskolb/XcodeGen).

```sh
xcodegen generate
xcodebuild -project Farabi.xcodeproj -scheme Farabi -configuration Release \
  -derivedDataPath /tmp/farabi-build CODE_SIGN_IDENTITY=- CODE_SIGNING_REQUIRED=NO build
```

The app is ad-hoc signed (no developer account needed). Keep the derived-data folder outside the project directory.
`tools/install.sh` does the same and copies `Farabi.app` to `/Applications`; pass `--agent` to also reload the watchdog
launch agent. The icon is generated with `swift tools/icon.swift Sources/Assets.xcassets/AppIcon.appiconset`.

## Testing
- **Headless:** `Farabi.app/Contents/MacOS/Farabi --test-chat qwen3.8-27b "your question"` sends one request to the running
  engine and prints the reply and generation statistics.
- **Live test hook** (verify the UI without mouse automation): run
  `defaults write io.github.maximianuu.Farabi test "<png>|<command>"`, then post the distributed notification
  `io.github.maximianuu.Farabi.test`. The app saves a window screenshot to `<png>` and a JSON state dump to `<png>.json`
  (use `-` as the png to skip the screenshot). Commands: empty (screenshot only), `settings`, `settings-close`,
  `model:<name>`, `ask:<text>`, `new`, `chat:<n>`, `chat-delete:<n>`, `think:1|0`, `lang:tr|en`, `background:1|0`, `unload`.

## Files
- `Sources/main.swift`: entry point (`--watchdog`, `--test-chat`, otherwise the UI)
- `Sources/Engine.swift`: engine API client, process start/stop, `Watchdog`, `LaunchAgent`
- `Sources/App.swift`: SwiftUI views, chat store, app state, test hook; `tx(tr, en)` picks the UI language
- `tools/icon.swift`, `tools/install.sh`, `project.yml` (XcodeGen), `Sources/Assets.xcassets`
