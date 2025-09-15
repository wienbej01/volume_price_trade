#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Ensure env + logs dir
mkdir -p logs
set -a; source ./env.sh; set +a

# Args
PROMPT_FILE="${1:?Usage: ./run_prompt_safely.sh codex/prompts/00_persona.md [logs/run_XX.log]}"
LOG_FILE="${2:-logs/run_$(date -u +%Y%m%d_%H%M%SZ).log}"

# Pre-create the log so 'tail' won't 404
: > "$LOG_FILE"
echo "=== NEW RUN === $(date -u +%FT%TZ) prompt=$(readlink -f "$PROMPT_FILE")" >> "$LOG_FILE"

# Launch fully detached, no stdout/stderr to TTY
nohup python -u scripts/glm_orchestrator_v3.py "$PROMPT_FILE" \
  --quiet --max-tokens 700 --loops 12 --log "$LOG_FILE" \
  >/dev/null 2>&1 < /dev/null & disown

# Print only the path (for scripting/automation)
echo "$LOG_FILE"
