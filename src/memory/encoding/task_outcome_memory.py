import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional


COMPONENT_MEMORY_DIR = Path(
    os.environ.get(
        "COMPONENT_MEMORY_DIR",
        str(Path(__file__).parent.parent.parent.parent / "workspace" / "component_memory"),
    )
)


FAILURE_COMPONENTS = {
    "agent_timeout": "agent_loop",
    "budget_waste": "agent_loop",
    "tool_argument_parse": "tool_parsing",
    "marker_corruption": "command_execution",
    "shell_quoting": "command_execution",
    "agent_runtime_error": "error_handling",
    "verifier_failure": None,
    "infra_failure": None,
    "missing_result": None,
    "corrupt_result": None,
    "unknown_failure": None,
}


FAILURE_GUIDANCE = {
    "agent_timeout": (
        "Agent execution timed out. Prefer bounded command execution, explicit "
        "progress checks, and early submission once a task-specific success signal is found."
    ),
    "budget_waste": (
        "The agent likely spent too much budget after finding a plausible path. Add "
        "convergence checks and stop conditions for solved tasks."
    ),
    "tool_argument_parse": (
        "Tool call arguments were malformed. Strengthen structured tool-call formatting "
        "and validation before execution."
    ),
    "marker_corruption": (
        "Terminal protocol or marker handling failed. Make command execution robust to "
        "interactive output, prompts, and terminal control sequences."
    ),
    "shell_quoting": (
        "Shell quoting damaged generated files or commands. Prefer Python-based file "
        "writes and avoid fragile heredoc or history-expansion patterns."
    ),
    "agent_runtime_error": (
        "The harness raised a runtime error. Add localized exception handling and safe "
        "fallbacks around the failing control path."
    ),
    "verifier_failure": (
        "The task ran but did not satisfy the verifier. Improve task understanding, "
        "spec inspection, and final validation before submission."
    ),
    "infra_failure": (
        "Environment setup failed before agent logic could be evaluated. Do not treat "
        "this as negative harness evidence."
    ),
    "missing_result": (
        "The trial produced no result file. Treat as execution reliability evidence "
        "unless logs show pure infrastructure failure."
    ),
    "corrupt_result": (
        "The result file was unreadable. Treat as benchmark plumbing reliability evidence."
    ),
    "unknown_failure": (
        "The failure lacks a clear signature. Inspect the raw trial logs before drawing "
        "component-level conclusions."
    ),
}


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _stringify(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_stringify(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(_stringify(v) for v in value)
    return str(value)


def _reward_from_result(result: dict) -> float:
    verifier = result.get("verifier_result") or {}
    rewards = verifier.get("rewards") or {}
    reward = rewards.get("reward")
    try:
        return float(reward) if reward is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def classify_trial_result(result: Optional[dict], log_text: str = "") -> dict:
    if result is None:
        category = "missing_result"
        return {
            "reward": 0.0,
            "outcome": "fail",
            "failure_category": category,
            "failure_component": FAILURE_COMPONENTS[category],
            "exception_type": None,
        }

    reward = _reward_from_result(result)

    _ar = result.get("agent_result") or {}
    n_episodes = (_ar.get("metadata") or {}).get("n_episodes")
    exception_info = result.get("exception_info") or {}
    exception_type = (
        result.get("exception_type")
        or result.get("error_type")
        or exception_info.get("exception_type")
        or exception_info.get("error_type")
    )
    text = " ".join([
        str(exception_type or ""),
        _stringify(result.get("exception")),
        _stringify(result.get("error")),
        _stringify(result.get("message")),
        _stringify(result.get("traceback")),
        _stringify(exception_info.get("exception_message")),
        _stringify(exception_info.get("exception_traceback")),
        _stringify(exception_info.get("message")),
        _stringify(exception_info.get("traceback")),
        log_text,
    ]).lower()

    if reward > 0:
        return {
            "reward": reward,
            "outcome": "pass",
            "failure_category": None,
            "failure_component": None,
            "exception_type": exception_type,
        }

    category = None
    etype = str(exception_type or "").lower()
    if "failed to parse tool arguments" in text or (
        "tool" in text and "invalid json" in text
    ):
        category = "tool_argument_parse"
    elif "failed to send non-blocking keys" in text or "marker" in text:
        category = "marker_corruption"
    elif "event not found" in text or "history expansion" in text or "heredoc" in text:
        category = "shell_quoting"
    elif "agenttimeouterror" in etype or "agenttimeout" in text:
        category = "agent_timeout"

    elif "artificially limited" in text and n_episodes is not None and n_episodes >= 30:
        category = "agent_timeout"

    elif "parser warnings" in text and (
        "no valid json" in text or "extra text" in text
    ):
        category = "tool_argument_parse"
    elif (
        "environmentstarttimeouterror" in etype
        or "agentsetuptimeouterror" in etype
        or "_setup_environment" in text
        or "_setup_agent" in text
        or "environment start" in text
        or "docker image" in text
        or "pull access denied" in text
        or "403 forbidden" in text
        or ("container" in text and "setup" in text)
        or "asciinema" in text
        or ("tmux" in text and "setup" in text)
    ):
        category = "infra_failure"
    elif "runtimeerror" in etype:
        category = "agent_runtime_error"
    elif result.get("verifier_result") is not None:
        category = "verifier_failure"
    else:
        category = "unknown_failure"

    return {
        "reward": reward,
        "outcome": "fail",
        "failure_category": category,
        "failure_component": FAILURE_COMPONENTS.get(category),
        "exception_type": exception_type,
    }


def _majority(values: list):
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None
    return Counter(filtered).most_common(1)[0][0]


def _read_trial_result(trial_dir: Path) -> tuple[Optional[dict], str, Optional[str]]:
    result_path = trial_dir / "result.json"
    log_chunks = []
    for log_name in ("exception.txt", "trial.log", "agent.log", "run.log"):
        p = trial_dir / log_name
        if p.exists():
            try:
                log_chunks.append(
                    p.read_text(encoding="utf-8", errors="ignore")[-6000:]
                )
            except OSError:
                pass
    log_text = "\n".join(log_chunks)[-12000:]

    if not result_path.exists():
        return None, log_text, "missing_result"
    try:
        return json.loads(result_path.read_text(encoding="utf-8")), log_text, None
    except (json.JSONDecodeError, OSError):
        return None, log_text, "corrupt_result"


def summarize_task_outcomes(task_outcomes: dict) -> dict:
    category_counts: Counter = Counter()
    component_counts: Counter = Counter()
    passes = 0
    partial_passes = 0
    failures = 0
    infra_failures = 0

    for item in task_outcomes.values():
        rate = float(item.get("pass_rate", 0) or 0)
        if rate >= 1.0:
            passes += 1
        elif rate > 0:
            partial_passes += 1
        else:
            failures += 1

        category = item.get("failure_category")
        component = item.get("failure_component")
        if category:
            category_counts[category] += 1
            if category == "infra_failure":
                infra_failures += 1
        if component:
            component_counts[component] += 1

    return {
        "n_tasks": len(task_outcomes),
        "passes": passes,
        "partial_passes": partial_passes,
        "failures": failures,
        "infra_failures": infra_failures,
        "failure_categories": dict(category_counts),
        "failure_components": dict(component_counts),
    }


def collect_task_outcomes(
    job_dir: Path,
    task_rewards: Optional[dict] = None,
    expected_trials: Optional[int] = None,
) -> dict:
    job_dir = Path(job_dir)
    task_trials: dict[str, list[dict]] = {}
    if not job_dir.exists():
        return {"tasks": {}, "summary": summarize_task_outcomes({})}

    for trial_dir in sorted(job_dir.iterdir()):
        if not trial_dir.is_dir() or "__" not in trial_dir.name:
            continue
        task = trial_dir.name.rsplit("__", 1)[0]
        result, log_text, read_error = _read_trial_result(trial_dir)
        if read_error == "corrupt_result":
            classified = {
                "reward": 0.0,
                "outcome": "fail",
                "failure_category": "corrupt_result",
                "failure_component": FAILURE_COMPONENTS["corrupt_result"],
                "exception_type": None,
            }
        else:
            classified = classify_trial_result(result, log_text)
        classified["trial_id"] = trial_dir.name
        task_trials.setdefault(task, []).append(classified)

    tasks = {}
    for task, trials in sorted(task_trials.items()):
        rewards = [float(t.get("reward", 0) or 0) for t in trials]
        if task_rewards and task in task_rewards:
            rewards = [float(x) for x in task_rewards.get(task, [])]
        pass_count = sum(1 for r in rewards if r > 0)
        trial_count = len(rewards) or len(trials)
        categories = [
            t.get("failure_category") for t in trials if t.get("reward", 0) <= 0
        ]
        components = [
            t.get("failure_component") for t in trials if t.get("reward", 0) <= 0
        ]
        exception_types = [
            t.get("exception_type") for t in trials if t.get("exception_type")
        ]
        pass_rate = pass_count / max(trial_count, 1)
        if pass_rate >= 1.0:
            outcome = "pass"
        elif pass_rate > 0:
            outcome = "partial_pass"
        else:
            outcome = "fail"

        tasks[task] = {
            "outcome": outcome,
            "pass_rate": round(pass_rate, 4),
            "rewards": rewards,
            "trial_count": trial_count,
            "expected_trials": expected_trials,
            "failure_category": _majority(categories),
            "failure_component": _majority(components),
            "exception_types": sorted(set(exception_types)),
            "trial_failures": [
                {
                    "trial_id": t.get("trial_id"),
                    "failure_category": t.get("failure_category"),
                    "failure_component": t.get("failure_component"),
                    "exception_type": t.get("exception_type"),
                }
                for t in trials
                if t.get("failure_category")
            ],
        }

    return {
        "tasks": tasks,
        "summary": summarize_task_outcomes(tasks),
    }


class TaskOutcomeMemory:
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or COMPONENT_MEMORY_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "task_outcomes.json"

    def _blank(self) -> dict:
        return {
            "last_updated": "",
            "episodes": {},
            "tasks": {},
        }

    def load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return self._blank()

    def save(self, state: dict) -> None:
        state["last_updated"] = _timestamp()
        self.path.write_text(
            json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _comparison_episode(self, state: dict, episode: dict) -> Optional[dict]:
        if episode.get("compare_to_previous") is False:
            return None
        parent = episode.get("parent_candidate")
        episodes = state.get("episodes", {})
        if parent and parent in episodes:
            return episodes[parent]
        if not episodes:
            return None
        return list(episodes.values())[-1]

    def update_episode(self, episode: dict) -> dict:
        task_outcomes = episode.get("task_outcomes") or {}
        if not task_outcomes:
            return {
                "task_outcome_summary": summarize_task_outcomes({}),
                "task_new_passes": [],
                "task_regressions": [],
                "task_persistent_failures": [],
                "new_pass_attribution": {},
                "task_failure_category_deltas": {
                    "improved_categories": {},
                    "regressed_categories": {},
                    "persistent_categories": {},
                },
            }

        state = self.load()
        previous = self._comparison_episode(state, episode)
        previous_tasks = (previous or {}).get("tasks", {})

        new_passes = []
        regressions = []
        persistent_failures = []
        improved_categories: Counter = Counter()
        regressed_categories: Counter = Counter()
        persistent_categories: Counter = Counter()
        new_pass_attribution: dict[str, dict] = {}

        for task, current in sorted(task_outcomes.items()):
            current_rate = float(current.get("pass_rate", 0) or 0)
            current_category = current.get("failure_category")
            previous_item = previous_tasks.get(task, {})
            previous_rate = float(previous_item.get("pass_rate", 0) or 0)
            previous_category = previous_item.get("failure_category")

            if previous is not None and task in previous_tasks:
                if current_rate > previous_rate:
                    new_passes.append(task)
                    if previous_category:
                        improved_categories[previous_category] += 1
                        component = FAILURE_COMPONENTS.get(previous_category)
                        if component:
                            bucket = new_pass_attribution.setdefault(component, {
                                "tasks": [],
                                "source_failure_category": previous_category,
                                "count": 0,
                            })
                            bucket["tasks"].append(task)
                            bucket["count"] = len(bucket["tasks"])
                elif current_rate < previous_rate:
                    regressions.append(task)
                    if current_category:
                        regressed_categories[current_category] += 1
                elif current_rate <= 0:
                    persistent_failures.append(task)
                    if current_category:
                        persistent_categories[current_category] += 1

            task_state = state.setdefault("tasks", {}).setdefault(task, {
                "ever_solved": False,
                "solved_by": [],
                "history": [],
            })
            if current_rate > 0:
                task_state["ever_solved"] = True
                if episode.get("episode_id") not in task_state["solved_by"]:
                    task_state["solved_by"].append(episode.get("episode_id"))
            task_state["latest_episode"] = episode.get("episode_id")
            task_state["latest_pass_rate"] = current_rate
            task_state["latest_failure_category"] = current_category
            task_state["latest_failure_component"] = current.get("failure_component")
            task_state.setdefault("history", []).append({
                "episode_id": episode.get("episode_id"),
                "iteration": episode.get("iteration", 0),
                "pass_rate": current_rate,
                "failure_category": current_category,
                "failure_component": current.get("failure_component"),
            })
            task_state["history"] = task_state["history"][-20:]

        summary = summarize_task_outcomes(task_outcomes)
        deltas = {
            "improved_categories": dict(improved_categories),
            "regressed_categories": dict(regressed_categories),
            "persistent_categories": dict(persistent_categories),
        }
        enriched = {
            "task_outcome_summary": summary,
            "task_new_passes": new_passes,
            "task_regressions": regressions,
            "task_persistent_failures": persistent_failures,
            "new_pass_attribution": new_pass_attribution,
            "task_failure_category_deltas": deltas,
        }

        episode_id = episode.get("episode_id", "")
        state.setdefault("episodes", {})[episode_id] = {
            "episode_id": episode_id,
            "iteration": episode.get("iteration", 0),
            "parent_candidate": episode.get("parent_candidate"),
            "score": episode.get("score"),
            "tasks": task_outcomes,
            **enriched,
        }
        self.save(state)
        return enriched

    def guidance_summary(self, recent_window: int = 5) -> dict:
        state = self.load()
        episodes = list(state.get("episodes", {}).values())
        latest = episodes[-1] if episodes else {}
        latest_summary = latest.get("task_outcome_summary", {})

        recent_new_passes = []
        recent_regressions = []
        for ep in episodes[-recent_window:]:
            if ep.get("task_new_passes"):
                recent_new_passes.append({
                    "episode_id": ep.get("episode_id"),
                    "iteration": ep.get("iteration"),
                    "tasks": ep.get("task_new_passes", [])[:10],
                    "new_pass_attribution": ep.get("new_pass_attribution", {}),
                    "improved_categories": ep.get("task_failure_category_deltas", {}).get(
                        "improved_categories", {}
                    ),
                })
            if ep.get("task_regressions"):
                recent_regressions.append({
                    "episode_id": ep.get("episode_id"),
                    "iteration": ep.get("iteration"),
                    "tasks": ep.get("task_regressions", [])[:10],
                    "regressed_categories": ep.get("task_failure_category_deltas", {}).get(
                        "regressed_categories", {}
                    ),
                })

        task_states = state.get("tasks", {})
        unsolved_by_category: dict[str, list[str]] = {}
        for task, item in task_states.items():
            if item.get("ever_solved"):
                continue
            category = item.get("latest_failure_category") or "unknown_failure"
            unsolved_by_category.setdefault(category, []).append(task)

        persistent_failure_categories = latest_summary.get("failure_categories", {})
        top_failure_categories = sorted(
            persistent_failure_categories.items(), key=lambda x: (-x[1], x[0])
        )

        category_guidance = []
        for category, count in top_failure_categories[:6]:
            category_guidance.append({
                "failure_category": category,
                "count": count,
                "suggested_component": FAILURE_COMPONENTS.get(category),
                "guidance": FAILURE_GUIDANCE.get(
                    category, FAILURE_GUIDANCE["unknown_failure"]
                ),
                "treat_as_harness_evidence": category != "infra_failure",
            })

        return {
            "latest_episode": latest.get("episode_id"),
            "tracked_task_count": len(task_states),
            "solved_task_count": sum(
                1 for item in task_states.values() if item.get("ever_solved")
            ),
            "latest_summary": latest_summary,
            "persistent_failure_categories": persistent_failure_categories,
            "category_guidance": category_guidance,
            "recent_new_passes": recent_new_passes,
            "recent_regressions": recent_regressions,
            "unsolved_by_category": {
                key: value[:10] for key, value in sorted(unsolved_by_category.items())
            },
        }
