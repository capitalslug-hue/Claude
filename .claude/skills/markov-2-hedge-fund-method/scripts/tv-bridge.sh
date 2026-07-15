#!/bin/sh
# Bootstrap + launch the TradingView MCP bridge (tradesdontlie/tradingview-mcp).
#
# Runs on the LOCAL machine only — the bridge talks to TradingView Desktop via
# Chrome DevTools Protocol on localhost:${CDP_PORT:-9222}, which a remote/cloud
# Claude Code session can never reach. On first launch it clones the bridge
# into .tradingview-mcp/ (gitignored) and installs its dependencies; after
# that it just starts the server. Requires: git, Node.js 18+.
set -e

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
DEST="$ROOT/.tradingview-mcp"

if [ ! -f "$DEST/src/server.js" ]; then
    echo "tv-bridge: first run — cloning tradingview-mcp into $DEST" >&2
    git clone --depth 1 https://github.com/tradesdontlie/tradingview-mcp.git "$DEST" >&2
    (cd "$DEST" && npm install --silent) >&2
fi

exec node "$DEST/src/server.js"
