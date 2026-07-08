import os
import difflib
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

COMPONENT_MEMORY_DIR = Path(os.environ.get("COMPONENT_MEMORY_DIR", str(Path(__file__).parent.parent.parent.parent / "workspace" / "component_memory")))

if __name__.startswith("src.memory."):
    sys.modules[__name__.replace("src.", "", 1)] = sys.modules[__name__]
elif __name__.startswith("memory."):
    sys.modules[f"src.{__name__}"] = sys.modules[__name__]
EVO_DIR = Path(__file__).parent.parent.parent / "workspace" / "evo"
AGENTS_DIR = Path(__file__).parent.parent.parent / "agents"


_PARENT_NAME_MAP: dict[str, str] = {}


def configure(task: str) -> None:
    global AGENTS_DIR, _PARENT_NAME_MAP
    if task == "terminal":
        AGENTS_DIR = Path(__file__).parent.parent.parent / "harness_agents"
        _PARENT_NAME_MAP = {
            "kira-baseline": "baseline_kira",
            "terminus2-baseline": "baseline_terminus2",
            "no_memory": "baseline_kira",
        }
    else:
        AGENTS_DIR = Path(__file__).parent.parent.parent / "agents"
        _PARENT_NAME_MAP = {}


def _resolve_parent_name(name: str) -> str:
    return _PARENT_NAME_MAP.get(name, name)


def _find_agent_file(name: str) -> list[str]:
    source = get_agent_source(name)
    return source.splitlines(keepends=True) if source else []


def get_agent_source(name: str) -> str:
    resolved = _resolve_parent_name(name)
    for candidate in ([name, resolved] if resolved != name else [name]):
        for d in [EVO_DIR, AGENTS_DIR]:
            p = d / f"{candidate}.py"
            if p.exists():
                return p.read_text(encoding="utf-8")
    return ""


def _compute_diff(episode_id: str, parent_candidate: str) -> str:
    current_lines = _find_agent_file(episode_id)
    if not current_lines:
        return ""
    parent_lines = _find_agent_file(parent_candidate) if parent_candidate else []
    diff = difflib.unified_diff(
        parent_lines,
        current_lines,
        fromfile=f"{parent_candidate}.py" if parent_candidate else "(none)",
        tofile=f"{episode_id}.py",
        lineterm="",
    )
    return "".join(diff)


class EpisodeRecorder:
    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or COMPONENT_MEMORY_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_path = self.output_dir / "episodes.jsonl"

        try:
            from memory.config_loader import get_memory_config
            self._regression_threshold = get_memory_config().get(
                "evidence_settings", {}
            ).get("regression_threshold", -3.0)
        except Exception:
            self._regression_threshold = -3.0


    _RELIABILITY_STATUSES = {"timeout", "crash", "invalid", "oversized_context"}

    def _determine_status(
        self,
        score: float,
        score_delta: float,
        failure_type: str | None = None,
    ) -> str:
        if failure_type and failure_type in self._RELIABILITY_STATUSES:
            return failure_type
        if score == 0:
            return "bug"
        if score_delta < self._regression_threshold:
            return "regression"
        return "success"

    def record(self, candidate_info: dict) -> dict:
        score = float(candidate_info.get("score", 0))
        score_delta = float(candidate_info.get("score_delta", 0))
        failure_type = candidate_info.get("failure_type") or None
        episode_id = candidate_info.get("episode_id", "")
        parent = candidate_info.get("parent_candidate") or ""
        token_cost = candidate_info.get("token_cost", 0)

        score_delta_vs_parent = candidate_info.get("score_delta_vs_parent")
        if score_delta_vs_parent is None and parent:
            parent_episode = next(
                (ep for ep in reversed(self.get_all_episodes())
                 if ep.get("episode_id") == parent),
                None,
            )
            if parent_episode is not None:
                score_delta_vs_parent = score - float(parent_episode.get("score", 0))

        score_delta_vs_best = candidate_info.get("score_delta_vs_best")
        if score_delta_vs_best is None:
            previous = self.get_all_episodes()
            if previous:
                best_score = max(float(ep.get("score", 0)) for ep in previous)
                score_delta_vs_best = score - best_score
            else:
                score_delta_vs_best = score_delta

        harness_hash = ""
        harness_path = candidate_info.get("harness_code_path", "")
        if harness_path:
            p = Path(harness_path)
            if p.exists():
                harness_hash = hashlib.md5(p.read_bytes()).hexdigest()[:8]

        episode = {
            "episode_id": episode_id,
            "iteration": candidate_info.get("iteration", 0),
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "parent_candidate": parent or None,
            "score": score,
            "score_delta": score_delta,
            "score_delta_vs_parent": score_delta_vs_parent,
            "score_delta_vs_best": score_delta_vs_best,
            "token_cost": token_cost,
            "history_bytes": token_cost,
            "history_bytes_proxy": token_cost,
            "token_cost_proxy": token_cost,
            "status": self._determine_status(score, score_delta, failure_type),
            "failure_type": failure_type,
            "harness_code_hash": harness_hash,
            "diff_from_parent": _compute_diff(episode_id, parent),
            "proposer_reasoning": (candidate_info.get("proposer_output") or "")[:2000],
        }


        if candidate_info.get("history_context_mode") is not None:
            episode["history_context_mode"] = candidate_info["history_context_mode"]
        if candidate_info.get("prompt_chars") is not None:
            episode["prompt_chars"] = candidate_info["prompt_chars"]


        reading_mode = candidate_info.get("reading_mode")
        if reading_mode is not None and reading_mode != "unknown":
            episode["reading_mode"] = reading_mode
        context_chars = candidate_info.get("context_chars")
        if context_chars:
            episode["context_chars"] = context_chars


        for key in (
            "evidence_maturity",
            "governance_mode",
            "candidate_role",
            "assigned_memory_ids",
            "assigned_avoid_memory_ids",
            "role_context_mode",
            "per_task_results",
            "task_outcomes",
            "task_outcome_summary",
            "task_new_passes",
            "task_regressions",
            "task_persistent_failures",
            "new_pass_attribution",
            "task_failure_category_deltas",
            "job_dir",
        ):
            if candidate_info.get(key) is not None:
                episode[key] = candidate_info[key]

        with self.episodes_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(episode) + "\n")

        return episode

    def get_all_episodes(self) -> list:
        if not self.episodes_path.exists():
            return []
        episodes = []
        for line in self.episodes_path.read_text(encoding="utf-8").strip().split("\n"):
            line = line.strip()
            if line:
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return episodes

    def _with_compat_fields(self, episode: dict) -> dict:
        migrated = dict(episode)
        token_cost = migrated.get("token_cost", 0)
        migrated.setdefault("score_delta_vs_parent", None)
        migrated.setdefault("score_delta_vs_best", migrated.get("score_delta", 0))
        migrated.setdefault("history_bytes", token_cost)
        migrated.setdefault("history_bytes_proxy", migrated.get("history_bytes", token_cost))
        migrated.setdefault("token_cost_proxy", token_cost)
        return migrated

    def migrate_existing_records(self) -> int:
        episodes = self.get_all_episodes()
        if not episodes:
            return 0

        migrated = [self._with_compat_fields(ep) for ep in episodes]
        changed = migrated != episodes
        if changed:
            with self.episodes_path.open("w", encoding="utf-8") as f:
                for ep in migrated:
                    f.write(json.dumps(ep) + "\n")
        return sum(1 for before, after in zip(episodes, migrated) if before != after)

    def get_latest_episode(self) -> dict:
        eps = self.get_all_episodes()
        return eps[-1] if eps else {}

    def get_best_episode(self) -> dict:
        eps = self.get_all_episodes()
        return max(eps, key=lambda e: e.get("score", 0)) if eps else {}


if __name__ == "__main__":
    recorder = EpisodeRecorder()
    migrated_count = recorder.migrate_existing_records()
    if migrated_count:
        print(f"Migrated existing episodes with compatibility fields: {migrated_count}")
        print()
    episodes = recorder.get_all_episodes()
    print(f"Total episodes: {len(episodes)}")
    print()
    latest = recorder.get_latest_episode()
    if latest:
        print("Latest episode:")
        print(json.dumps(latest, indent=2, ensure_ascii=False))
    print()
    best = recorder.get_best_episode()
    if best:
        print("Best episode:")
        print(json.dumps(best, indent=2, ensure_ascii=False))
