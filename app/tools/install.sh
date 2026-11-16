#!/bin/zsh
# Builds Farabi and copies it to /Applications.
# Usage: tools/install.sh [--agent]   (--agent also reloads the watchdog launch agent after installing)
set -e
cd "$(dirname "$0")/.."
DD="${TMPDIR:-/tmp}/farabi-build"      # derived data lives outside the project folder
LABEL=io.github.maximianuu.farabi.watchdog
DOMAIN=gui/$(id -u)
PLIST=$HOME/Library/LaunchAgents/$LABEL.plist

xcodegen generate >/dev/null
xcodebuild -project Farabi.xcodeproj -scheme Farabi -configuration Release -destination 'platform=macOS' \
  -derivedDataPath "$DD" CODE_SIGN_IDENTITY=- CODE_SIGNING_REQUIRED=NO build > "$DD.log" 2>&1 \
  || { grep -E "error:" "$DD.log" | head -20; echo "BUILD FAILED (see $DD.log)"; exit 1; }
APP="$DD/Build/Products/Release/Farabi.app"
codesign --verify --deep --strict "$APP"

if [ "$1" = "--agent" ]; then
  # Unload the watchdog first: it runs the app's own binary, which must not be replaced while running.
  launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
fi
osascript -e 'tell application "Farabi" to quit' >/dev/null 2>&1 || true
for i in {1..30}; do pgrep -x Farabi >/dev/null || break; sleep 0.3; done
ditto "$APP" /Applications/Farabi.app
echo "installed: /Applications/Farabi.app"
if [ "$1" = "--agent" ]; then
  if [ -f "$PLIST" ]; then
    launchctl bootstrap "$DOMAIN" "$PLIST" && echo "watchdog reloaded"
  else
    echo "no watchdog plist yet: enable it in Farabi > Settings"
  fi
fi
