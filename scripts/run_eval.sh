#!/usr/bin/env bash
# Usage: scripts/run_eval.sh <agent_import_path> [dataset] [runs] [n_concurrent] [extra_harbor_flags...]
# dataset: terminal-bench@2.0 (default) | swebench-verified

set -euo pipefail

AGENT_IMPORT_PATH="${1:?Usage: $0 <agent_import_path> [dataset] [runs] [n_concurrent] [extra_harbor_flags...]}"
DATASET="${2:-terminal-bench@2.0}"
RUNS="${3:-2}"
N_CONCURRENT="${4:-1}"
shift "$(( $# < 4 ? $# : 4 ))"
EXTRA_FLAGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Load .env — walk up from REPO_DIR until found.
_ENV_FILE=""
_search="$REPO_DIR"
for _i in 1 2 3; do
    if [[ -f "$_search/.env" ]]; then
        _ENV_FILE="$_search/.env"
        break
    fi
    _search="$(dirname "$_search")"
done
if [[ -n "$_ENV_FILE" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$_ENV_FILE"
    set +a
fi

MODEL="${HARBOR_MODEL:-anthropic/claude-opus-4-6}"
API_BASE="${HARBOR_API_BASE:-}"

echo "agent:       $AGENT_IMPORT_PATH"
echo "dataset:     $DATASET"
echo "model:       $MODEL"
echo "concurrent:  $N_CONCURRENT"
echo "runs:        $RUNS"
echo ""

# Prefer an outer wall-clock timeout when GNU timeout is available.
TIMEOUT_CMD=()
if command -v timeout >/dev/null 2>&1; then
    TIMEOUT_CMD=(timeout --signal=TERM --kill-after=60 2h)
elif command -v gtimeout >/dev/null 2>&1; then
    TIMEOUT_CMD=(gtimeout --signal=TERM --kill-after=60 2h)
fi

CMD=(
    harbor run
    --agent-import-path "$AGENT_IMPORT_PATH"
    -d "$DATASET"
    -m "$MODEL"
    -e docker
    -n "$N_CONCURRENT"
    --n-attempts "$RUNS"
)
if [[ -n "$API_BASE" ]]; then
    CMD+=(--ak "api_base=$API_BASE")
fi
if [[ ${#EXTRA_FLAGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_FLAGS[@]}")
fi

if [[ ${#TIMEOUT_CMD[@]} -gt 0 ]]; then
    "${TIMEOUT_CMD[@]}" "${CMD[@]}"
else
    "${CMD[@]}"
fi
