#!/usr/bin/env bash
set -euo pipefail

# ========= USER CONFIG =========
OR_KEY="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"
PROJECT_DIR="${HOME}/volume_price_trade"   # adjust if the folder lives elsewhere
TITLE_HEADER="codex-cli"
REFERER_HEADER="http://localhost"

# ========= PRECHECKS =========
if [ ! -d "${PROJECT_DIR}" ]; then
  echo "❌ Project folder not found: ${PROJECT_DIR}"
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "❌ curl not found. Install curl and re-run."
  exit 1
fi

cd "${PROJECT_DIR}"

# ========= .env (safe write/update, don’t print key) =========
touch .env
# Ensure the three core lines exist (create if missing)
grep -q '^OPENROUTER_API_KEY=' .env || echo 'OPENROUTER_API_KEY=' >> .env
grep -q '^OPENAI_API_KEY=' .env      || echo 'OPENAI_API_KEY=${OPENROUTER_API_KEY}' >> .env
grep -q '^OPENAI_BASE_URL=' .env     || echo 'OPENAI_BASE_URL=https://openrouter.ai/api/v1' >> .env
# Optional attribution envs
grep -q '^HTTP_REFERER=' .env || echo 'HTTP_REFERER=http://localhost' >> .env
grep -q '^X_TITLE=' .env      || echo 'X_TITLE=codex-cli' >> .env

# Inject/replace the OpenRouter key (won’t echo to terminal)
sed -i "s|^OPENROUTER_API_KEY=.*$|OPENROUTER_API_KEY=${OR_KEY}|" .env
# Ensure OPENAI_API_KEY mirrors OPENROUTER_API_KEY (keeps other lines intact)
sed -i 's|^OPENAI_API_KEY=.*$|OPENAI_API_KEY=${OPENROUTER_API_KEY}|' .env
# Ensure base URL points to OpenRouter
sed -i 's|^OPENAI_BASE_URL=.*$|OPENAI_BASE_URL=https://openrouter.ai/api/v1|' .env

chmod 600 .env

# Helper to load env in shells that don’t auto-export
cat > env.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
set -a; source .env; set +a
EOF
chmod +x env.sh

# ========= Codex config (idempotent, backup if exists) =========
mkdir -p codex
if [ -f codex/config.yaml ]; then
  cp codex/config.yaml "codex/config.yaml.bak.$(date +%s)"
fi

# Note: headers are literal values here (no env expansion needed by Codex).
cat > codex/config.yaml <<'EOF'
llm:
  provider: openrouter
  base_url: https://openrouter.ai/api/v1
  api_key_env: OPENROUTER_API_KEY
  model: z-ai/glm-4.5
  headers:
    HTTP-Referer: http://localhost
    X-Title: codex-cli
  reasoning:
    enabled: true
EOF

# Optional profile
if [ ! -f codex/profile.md ]; then
cat > codex/profile.md <<'EOF'
## Role
Head of Quantitative Strategy Development (CLI).
Use OpenRouter `z-ai/glm-4.5` with reasoning enabled for complex tasks.

## Guidance
- Be concise; produce complete, tested changes.
- Prefer deterministic steps; summarize assumptions.
EOF
fi

# ========= Smoke test to OpenRouter via cURL =========
echo "→ Running OpenRouter cURL smoke test…"
set +e
RESP="$(bash -c 'set -a; source ./.env; set +a; \
  curl -sS https://openrouter.ai/api/v1/chat/completions \
    -H "Authorization: Bearer ${OPENROUTER_API_KEY}" \
    -H "Content-Type: application/json" \
    -H "HTTP-Referer: '"${REFERER_HEADER}"'" \
    -H "X-Title: '"${TITLE_HEADER}"'" \
    -d '"'"'{
      "model": "z-ai/glm-4.5",
      "messages": [{"role":"user","content":"Say OK from GLM-4.5 in one short line."}]
    }'"'"' \
  | head -c 400')"
STATUS=$?
set -e

if [ $STATUS -ne 0 ] || [[ -z "$RESP" ]]; then
  echo "❌ OpenRouter smoke test failed. Check your network or API key."
  exit 1
else
  echo "✅ OpenRouter reachable. Sample (truncated):"
  echo "$RESP"
  echo
fi

# ========= Launch Codex (best-effort; adapt to your CLI) =========
if ! command -v codex >/dev/null 2>&1; then
  echo "⚠️  Codex CLI not detected on PATH. Setup complete; run Codex after you install it."
  echo "Next time: cd ${PROJECT_DIR} && source ./env.sh && <your codex command>"
  exit 0
fi

echo "→ Launching Codex against z-ai/glm-4.5 (reasoning enabled)…"
source ./env.sh || true

# Try common subcommands; adjust if your Codex uses a different verb
if codex ask --help >/dev/null 2>&1; then
  codex ask "Confirm you are z-ai/glm-4.5 via OpenRouter. Print exactly: OK-GLM45."
elif codex chat --help >/dev/null 2>&1; then
  codex chat -m "Confirm you are z-ai/glm-4.5 via OpenRouter. Print exactly: OK-GLM45."
elif codex run --help >/dev/null 2>&1; then
  codex run :llm:ping -- "Confirm you are z-ai/glm-4.5 via OpenRouter. Print exactly: OK-GLM45."
else
  echo "ℹ️  Starting interactive Codex (no subcommand detected)…"
  codex || true
fi

echo "🎉 Done."
echo "Project: ${PROJECT_DIR}"
echo "Reuse later: cd ${PROJECT_DIR} && source ./env.sh && <your codex command>"
