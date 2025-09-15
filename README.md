# volume_price_trade

A minimal-but-serious ML backtesting framework focused on intraday **Volume Profile + VPA + ICT** behaviors over S&P 500 1-minute bars in GCS. Uses a proper **src/ package layout**.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
bash scripts/run_checks.sh
```

Then step through the Codex prompts in `codex/prompts/` (00 → 07). Codex acts as orchestrator only.
