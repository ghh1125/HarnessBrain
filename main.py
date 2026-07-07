"""HarnessBrain — entry point.

Subcommands:
  evolve      Run the evolution loop (LLM proposes → benchmark evaluates)
  benchmark   Run or display benchmark results (text classification task)

Task selection:
  --task text      Text classification (Symptom2Disease, LawBench, USPTO)  [default]
  --task terminal  Terminal-Bench 2.0 / SWE-bench (Docker-based, harbor eval)

Examples:
  python main.py evolve --dataset USPTO --iterations 20
  python main.py evolve --task terminal --iterations 5 --trials 2
  python main.py benchmark --dataset USPTO
  python main.py benchmark --test --results
  python main.py benchmark --frontier --test
"""

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))


def _add_evolve_args(p: argparse.ArgumentParser):
    """Combined evolve args for both text and terminal tasks."""
    p.add_argument("--iterations", type=int, default=None,
                   help="Number of evolution iterations (uses the built-in per-task default if unset)")
    p.add_argument("--propose-timeout", type=int, default=2400,
                   help="Timeout for proposer (seconds)")
    p.add_argument("--run-name", type=str, default=None,
                   help="Custom run name (auto-generated if not set)")
    p.add_argument("--fresh", action="store_true",
                   help="Clear previous agents and logs before starting")
    p.add_argument("--skip-baseline", action="store_true",
                   help="Skip Phase 0 baseline evaluation")
    # Text-task args
    p.add_argument("--datasets", nargs="+", default=None,
                   help="[text] Datasets to evolve on (default: all from config.yaml)")
    p.add_argument("--model", default=None, help="[text] Classifier model override")
    # Terminal-task args
    p.add_argument("--trials", type=int, default=2,
                   help="[terminal] Trials per task during evolution")
    p.add_argument("--skip-smoke", action="store_true",
                   help="[terminal] Skip smoke tests")
    p.add_argument("--full-eval", action="store_true",
                   help="[terminal] Run optional 5-trial winner eval on full dataset")
    p.add_argument("--concurrent", type=int, default=1,
                   help="[terminal] Max concurrent harbor trials")


def _add_benchmark_args(p: argparse.ArgumentParser):
    p.add_argument("--memory", type=str, help="Filter to one memory system by name or path")
    p.add_argument("--dataset", type=str, help="Filter to one dataset")
    p.add_argument("--model", type=str, help="Filter by model name (used with --frontier)")
    p.add_argument("--test", action="store_true", help="Run test evaluation (default: val)")
    p.add_argument("--frontier", action="store_true", help="Print frontier summary and write JSON")
    p.add_argument("--results", action="store_true", help="Print results table only, no new jobs")
    p.add_argument("--pareto", action="store_true", help="Show only Pareto-optimal systems")
    p.add_argument("--mode", choices=["online", "offline"], default=None)
    p.add_argument("--num-epochs", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--logs-dir", type=str, default=None, help="Override logs directory path")


def main():
    parser = argparse.ArgumentParser(
        description="HarnessBrain — memory system evolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    evolve_p = sub.add_parser("evolve", help="Run the evolution loop")
    evolve_p.add_argument(
        "--task",
        choices=["text", "terminal"],
        default="text",
        help="Task type: 'text' for text classification, 'terminal' for Terminal-Bench/SWE-bench",
    )
    _add_evolve_args(evolve_p)

    bench_p = sub.add_parser("benchmark", help="Run or display benchmark results")
    _add_benchmark_args(bench_p)

    args = parser.parse_args()

    if args.subcommand == "evolve":
        import yaml
        cfg = yaml.safe_load((REPO_ROOT / "config.yaml").read_text())

        # Unified entry point: src.evolve.run_evolve dispatches internally by
        # args.task (text → classification loop, terminal → agent loop) and
        # selects the matching component-memory taxonomy at runtime.
        import signal
        import src.evolve as _ev
        signal.signal(signal.SIGINT, _ev._handle_signal)
        signal.signal(signal.SIGTERM, _ev._handle_signal)

        if args.task == "terminal":
            if args.iterations is None:
                args.iterations = 10
        else:
            if args.iterations is None:
                args.iterations = 20
            if args.model is None:
                args.model = None
            if args.datasets is None:
                args.datasets = cfg["datasets"]

        _ev.run_evolve(args)

    elif args.subcommand == "benchmark":
        from src.benchmark import main_async, load_config
        cfg = load_config()
        if args.mode is None:
            args.mode = cfg["inner_loop"].get("mode", "offline")
        if args.num_epochs is None:
            args.num_epochs = cfg["inner_loop"].get("num_epochs", 1)
        if args.temperature is None:
            args.temperature = cfg["inner_loop"].get("temperature")
        asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
