"""trajectory_behavior.py — Rule-based behavior feature extraction from trajectory.json.

Extracts quantitative features from agent trajectories using string matching only
(no LLM calls). Features describe *what the agent did* before submitting, used to
attribute prompt_template evidence when newly-passing tasks show verification behavior.

AGENT-ONLY: this module reads agent trajectory.json files, which only exist for the
terminal/agent task. The text-classification loop never produces trajectories, so the
producer hook in component_evidence (_record_trajectory_behavior_attribution) is a
silent no-op there (it requires episode["job_dir"], which only the agent loop sets).

Public API
----------
extract_features(traj: dict) -> dict
    Extract features from a loaded trajectory dict.

extract_features_from_file(path: Path) -> dict | None
    Load and extract; returns None on any read/parse error.

find_task_trajectory(job_dir: Path, task_name: str) -> Path | None
    Locate trajectory.json for a task by name prefix inside a job directory.

aggregate_behavior_signal(task_features: dict[str, dict]) -> dict
    Aggregate per-task features into a single signal dict for attribution.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Keep `memory.encoding.trajectory_behavior` and `src.memory.encoding.trajectory_behavior`
# as one module object, matching the rest of the package.
if __name__.startswith("src.memory."):
    sys.modules[__name__.replace("src.", "", 1)] = sys.modules[__name__]
elif __name__.startswith("memory."):
    sys.modules[f"src.{__name__}"] = sys.modules[__name__]

# ── Command pattern sets ──────────────────────────────────────────────────────

# Commands that read or verify state (read-only intent)
_VERIFY_PATTERNS: tuple[str, ...] = (
    "cat ",
    "head ",
    "tail ",
    "ls ",
    "wc ",
    "grep ",
    "diff ",
    "python3 -c",
    "python -c",
    "echo $",
    "echo \"$",
    "echo '",
    "test ",
    "[ ",
    "[[ ",
    "stat ",
    "file ",
    "hexdump",
    "xxd ",
    "od ",
    "find ",
    "du ",
    "md5sum",
    "sha",
    "assert",
    "print(",
    "printf",
)

# Commands that write or modify state
_WRITE_PATTERNS: tuple[str, ...] = (
    "cat >",
    "echo >",
    " >> ",
    "tee ",
    "sed -i",
    "> /",
    "cp ",
    "mv ",
    "mkdir ",
    "touch ",
    "wget ",
    "curl ",
    "pip install",
    "apt-get",
    "npm install",
)

# Commands that clean up temporary files
_CLEANUP_PATTERNS: tuple[str, ...] = (
    "rm ",
    "rm -",
    "unlink ",
    "rmdir ",
)


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(traj: dict) -> dict:
    """Extract quantitative behavior features from a trajectory dict.

    Returns a flat dict with the following keys:

    total_steps             — total step entries in the trajectory
    total_tool_calls        — total tool calls across all steps
    task_complete_count     — number of mark_task_complete calls
    double_submit           — bool: agent submitted more than once
    first_submit_call_frac  — position of first submit / total calls (0.0–1.0)
    verify_cmds_pre_submit  — read/check commands in last 5 calls before first submit
    write_cmds_pre_submit   — write/modify commands in last 5 calls before first submit
    cleanup_pre_submit      — bool: rm/unlink in last 3 calls before first submit
    pre_submit_verify_ratio — verify_cmds_pre_submit / cmds in last 5 window
    """
    steps = traj.get("steps") or []

    # Flatten all tool calls across steps in order
    all_calls: list[tuple[str, str]] = []  # (function_name, keystrokes)
    for step in steps:
        for tc in step.get("tool_calls") or []:
            fn = tc.get("function_name") or ""
            ks = tc.get("arguments", {}).get("keystrokes") or ""
            all_calls.append((fn, ks))

    submit_indices = [i for i, (fn, _) in enumerate(all_calls) if fn == "mark_task_complete"]
    first_submit = submit_indices[0] if submit_indices else len(all_calls)

    pre_submit = all_calls[:first_submit]
    window = pre_submit[-5:] if len(pre_submit) >= 5 else pre_submit
    last3 = pre_submit[-3:] if len(pre_submit) >= 3 else pre_submit

    verify_count = sum(
        1 for fn, ks in window
        if fn == "bash_command" and any(p in ks for p in _VERIFY_PATTERNS)
    )
    write_count = sum(
        1 for fn, ks in window
        if fn == "bash_command" and any(p in ks for p in _WRITE_PATTERNS)
    )
    cleanup = any(
        any(p in ks for p in _CLEANUP_PATTERNS)
        for fn, ks in last3
        if fn == "bash_command"
    )

    return {
        "total_steps": len(steps),
        "total_tool_calls": len(all_calls),
        "task_complete_count": len(submit_indices),
        "double_submit": len(submit_indices) >= 2,
        "first_submit_call_frac": round(first_submit / max(len(all_calls), 1), 3),
        "verify_cmds_pre_submit": verify_count,
        "write_cmds_pre_submit": write_count,
        "cleanup_pre_submit": cleanup,
        "pre_submit_verify_ratio": round(verify_count / max(len(window), 1), 3),
    }


def extract_features_from_file(path: Path) -> dict | None:
    """Load trajectory.json and extract features; returns None on any error."""
    try:
        traj = json.loads(path.read_text(encoding="utf-8"))
        return extract_features(traj)
    except Exception:
        return None


def find_task_trajectory(job_dir: Path, task_name: str) -> Path | None:
    """Return the trajectory.json path for task_name inside job_dir.

    Task directories are named ``{task_name}__{random_suffix}``, so we glob
    by prefix.
    """
    job_dir = Path(job_dir)
    if not job_dir.is_dir():
        return None
    for candidate in job_dir.glob(f"{task_name}__*/agent/trajectory.json"):
        return candidate
    # Also try without the double-underscore suffix (exact match)
    exact = job_dir / task_name / "agent" / "trajectory.json"
    return exact if exact.exists() else None


# ── Aggregation ───────────────────────────────────────────────────────────────

# A task is considered to show "verification behavior" if it has at least one
# verify command before submitting, OR the agent submitted twice (second submit
# implies the harness prompted a re-check).
_VERIFY_THRESHOLD_PER_TASK = 1


def aggregate_behavior_signal(task_features: dict[str, dict]) -> dict:
    """Aggregate per-task feature dicts into a single behavior signal.

    Returns:
        tasks_with_verify  — count of tasks showing verification behavior
        tasks_with_cleanup — count of tasks that cleaned up before submit
        tasks_double_submit — count of tasks where agent submitted twice
        mean_verify_ratio  — mean pre_submit_verify_ratio across tasks
        verify_task_names  — list of task names that showed verify behavior
        total_tasks        — total tasks analyzed
    """
    verify_tasks: list[str] = []
    cleanup_count = 0
    double_submit_count = 0
    ratio_sum = 0.0

    for task_name, feat in task_features.items():
        if (feat.get("verify_cmds_pre_submit", 0) >= _VERIFY_THRESHOLD_PER_TASK
                or feat.get("double_submit", False)):
            verify_tasks.append(task_name)
        if feat.get("cleanup_pre_submit", False):
            cleanup_count += 1
        if feat.get("double_submit", False):
            double_submit_count += 1
        ratio_sum += feat.get("pre_submit_verify_ratio", 0.0)

    n = max(len(task_features), 1)
    return {
        "tasks_with_verify": len(verify_tasks),
        "tasks_with_cleanup": cleanup_count,
        "tasks_double_submit": double_submit_count,
        "mean_verify_ratio": round(ratio_sum / n, 3),
        "verify_task_names": verify_tasks,
        "total_tasks": len(task_features),
    }
