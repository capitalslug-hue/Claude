# Connecting Claude Code to your TradingView

The TradingView MCP bridge ([tradesdontlie/tradingview-mcp](https://github.com/tradesdontlie/tradingview-mcp))
lets Claude read and drive **TradingView Desktop running on your own computer**
via Chrome DevTools Protocol on `localhost:9222`. Nothing talks to
TradingView's servers; everything stays on your machine.

**Hard requirement:** this only works when Claude Code runs **locally** on the
same computer as TradingView Desktop. A remote / cloud / web Claude session can
never reach your machine's localhost — no prompt can change that.

## One-time setup (~5 minutes)

1. **TradingView Desktop** — install the desktop app from your TradingView
   account page. The browser version will NOT work.
2. **Node.js 18+** and **git** installed locally.
3. **Clone this repo** and open it in local Claude Code (CLI or desktop app).
   The repo's `.mcp.json` already defines a `tradingview` server; approve it
   when Claude Code asks. On first launch it auto-clones the bridge into
   `.tradingview-mcp/` (gitignored) and installs its dependencies — the
   wrapper doing this is
   `.claude/skills/markov-2-hedge-fund-method/scripts/tv-bridge.sh`.
   - **Windows without Git Bash:** clone the bridge manually
     (`git clone https://github.com/tradesdontlie/tradingview-mcp.git .tradingview-mcp`
     then `cd .tradingview-mcp && npm install`) and change `.mcp.json` to
     `"command": "node", "args": [".tradingview-mcp/src/server.js"]`.

## Every session

1. **Quit TradingView completely**, then relaunch it with debugging enabled:
   - **macOS:** `open -a "TradingView" --args --remote-debugging-port=9222`
   - **Windows:** `"C:\...\TradingView.exe" --remote-debugging-port=9222`
     (find the path via your Start-menu shortcut's properties)
   - **Linux:** `tradingview --remote-debugging-port=9222` (snap) or the
     flatpak/AppImage equivalent
   - If it won't connect, make sure no old TradingView process is still
     running and nothing else holds port 9222 (`lsof -i :9222`).
2. Open this repo in local Claude Code and ask Claude to run
   **`tv_health_check`** — success is `cdp_connected: true`.

## What this unlocks

- Claude sees your **actual charts, watchlists, and indicators** — including
  the `markov2.pine` indicator from this skill (`assets/markov2.pine`): paste
  it once into the Pine Editor, and Claude can then read its regime table and
  signals off your live charts.
- Chart control and Pine Script workflows via the bridge's ~78 tools.
- Pairs with `/markov-2-hedge-fund-method` for the heavy math: the skill's
  Python engine still does the honest stride-sampled matrices and walk-forward
  backtests; the bridge supplies your live TradingView context.

## Security note

`--remote-debugging-port` exposes the TradingView app to any process on your
machine via localhost while it's running. Launch with the flag only when
you're actively using the bridge, and quit/relaunch normally afterwards.
