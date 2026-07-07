#!/usr/bin/env bash
# Run evolution end-to-end, then show results.
#
# Usage:
#   ./run.sh            # text classification (default)
#   ./run.sh terminal   # agent task (Terminal-Bench / SWE-bench; needs Docker)
set -euo pipefail
cd "$(dirname "$0")"

TASK="${1:-text}"

if [[ "$TASK" == "terminal" ]]; then
    python main.py evolve --task terminal
    echo "Results: logs/frontier_val.json and logs/evolution_summary.jsonl"
else
    python main.py evolve
    python main.py benchmark --results
fi
