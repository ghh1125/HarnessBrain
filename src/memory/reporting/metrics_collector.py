"""MetricsCollector — collects structured experiment metrics after each run.

Reads from:
  workspace/memory/episodes.jsonl
  workspace/memory/component_evidence.json
  workspace/memory/search_guidance.json
  workspace/evo/iter_*/*/proposal_plan.json

Writes (append) to:
  workspace/memory/metrics.jsonl
"""

from __future__ import annotations
import os
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

COMPONENT_MEMORY_DIR = Path(os.environ.get("COMPONENT_MEMORY_DIR", str(Path(__file__).parent.parent.parent.parent / "workspace" / "component_memory")))
EVO_DIR = Path(__file__).parent.parent.parent / "workspace" / "evo"
LOGS_BASE = Path(__file__).parent.parent.parent / "logs"

_NULL = None


class MetricsCollector:
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        evo_dir: Optional[Path] = None,
    ):
        self.output_dir = output_dir or COMPONENT_MEMORY_DIR
        self.evo_dir = evo_dir or EVO_DIR
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self._evidence_path = self.output_dir / "component_evidence.json"
        self._guidance_path = self.output_dir / "search_guidance.json"
        self._episodes_path = self.output_dir / "episodes.jsonl"

    # ── Data loaders ──────────────────────────────────────────

    def _load_episodes(self) -> list[dict]:
        if not self._episodes_path.exists():
            return []
        episodes = []
        for line in self._episodes_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return episodes

    def _load_component_evidence(self) -> dict:
        if not self._evidence_path.exists():
            return {}
        try:
            return json.loads(self._evidence_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _load_search_guidance(self) -> dict:
        if not self._guidance_path.exists():
            return {}
        try:
            return json.loads(self._guidance_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _load_proposal_plans(self) -> dict[str, dict]:
        """Returns mapping episode_id -> plan dict."""
        plans: dict[str, dict] = {}
        for pf in sorted(self.evo_dir.glob("iter_*/*/proposal_plan.json")):
            try:
                record = json.loads(pf.read_text(encoding="utf-8"))
                cid = record.get("candidate_id") or pf.parent.name
                plans[cid] = record.get("plan", {})
            except Exception:
                continue
        return plans

    def _get_fewshot_baseline(self, logs_dir: Optional[Path] = None) -> Optional[float]:
        """Find fewshot_all val accuracy from logs directory."""
        search_roots = []
        if logs_dir and logs_dir.exists():
            search_roots.append(logs_dir)
        search_roots.append(LOGS_BASE)

        for root in search_roots:
            for val_file in root.rglob("val.json"):
                parts = val_file.parts
                if "fewshot_all" in parts:
                    try:
                        data = json.loads(val_file.read_text(encoding="utf-8"))
                        acc = data.get("accuracy")
                        if acc is not None:
                            return float(acc) * 100
                    except Exception:
                        continue
        return _NULL

    def _load_frontier(
        self,
        logs_dir: Optional[Path],
        filename: str,
        score_key: str,
    ) -> dict:
        """Load a val/test frontier file and return its best entry plus pareto rows."""
        path = (logs_dir or LOGS_BASE) / filename
        if not path.exists():
            return {
                "best_system": _NULL,
                "best_score": _NULL,
                "best_ctx_len": _NULL,
                "pareto": [],
            }

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {
                "best_system": _NULL,
                "best_score": _NULL,
                "best_ctx_len": _NULL,
                "pareto": [],
            }

        pareto = []
        for entry in data.get("_pareto", []):
            if not isinstance(entry, dict):
                continue
            score = entry.get(score_key, entry.get("accuracy"))
            if score is None:
                continue
            pareto.append(
                {
                    "system": entry.get("system"),
                    "score": round(float(score), 1),
                    "ctx_len": entry.get("ctx_len"),
                }
            )

        dataset_best = []
        for dataset, record in data.items():
            if str(dataset).startswith("_") or not isinstance(record, dict):
                continue
            score = record.get(score_key, record.get("accuracy"))
            system = record.get("best_system", record.get("system"))
            if system is None or score is None:
                continue
            dataset_best.append(
                {
                    "system": system,
                    "score": round(float(score), 1),
                    "ctx_len": record.get("ctx_len"),
                }
            )

        candidates = pareto or dataset_best
        best = max(candidates, key=lambda r: float(r.get("score") or 0), default=None)
        return {
            "best_system": best.get("system") if best else _NULL,
            "best_score": best.get("score") if best else _NULL,
            "best_ctx_len": best.get("ctx_len") if best else _NULL,
            "pareto": pareto,
        }

    def _extract_test_system_from_launcher_log(
        self, log_file: Path, text: str
    ) -> Optional[str]:
        """Extract the memory system name from a benchmark test launcher log."""
        match = re.search(r"--memory\s+(?:\S*/)?agents/([^/\s]+)\.py", text)
        if match:
            return match.group(1)

        # Fallback for launcher names such as:
        # 00_test_Symptom2Disease_mem_agent_i6_3_gpt-oss-120b.log
        # 00_test_Symptom2Disease_mem_agent_i6_3_claude-3-5-sonnet.log
        name_match = re.search(r"_test_[^_]+_(mem_agent_i\d+_\d+)_", log_file.name)
        if name_match:
            return name_match.group(1)
        return None

    def _result_json_matches_launcher(self, result_path: Path, log_file: Path) -> bool:
        """Return True when a result JSON appears to belong to this launcher run."""
        try:
            if not result_path.exists():
                return False
            # The benchmark writes result JSON immediately before the launcher log
            # is closed. If the global results/ file was overwritten by a later
            # ablation group, its mtime will drift away from this log.
            return abs(result_path.stat().st_mtime - log_file.stat().st_mtime) <= 300
        except Exception:
            return False

    def _empty_current_test_frontier(self, source: str = "current_run_missing") -> dict:
        return {
            "best_system": _NULL,
            "best_score": _NULL,
            "best_ctx_len": _NULL,
            "best_correct": _NULL,
            "best_total": _NULL,
            "pareto": [],
            "source": source,
        }

    def _current_run_test_results_from_files(self, logs_dir: Path) -> list[dict]:
        """Read current-run mem_agent test results from the global results dir.

        benchmark.py stores test outputs under text_classification/results, not
        under the per-run logs dir. Because mem_agent_i* names are reused across
        ablation groups, stale global test.json files must be ignored.
        """
        frontier_val = logs_dir / "frontier_val.json"
        freshness_cutoff = (
            frontier_val.stat().st_mtime - 60
            if frontier_val.exists()
            else logs_dir.stat().st_mtime - 60
        )
        results_root = Path(__file__).parent.parent.parent / "results"
        candidates: list[dict] = []

        for val_file in logs_dir.glob("*/mem_agent_*/*/val.json"):
            try:
                dataset = val_file.parts[-4]
                system = val_file.parts[-3]
                model = val_file.parts[-2]
            except Exception:
                continue

            result_path = results_root / dataset / system / model / "test.json"
            try:
                if (
                    not result_path.exists()
                    or result_path.stat().st_mtime < freshness_cutoff
                ):
                    continue
                result = json.loads(result_path.read_text(encoding="utf-8"))
                acc = result.get("accuracy")
                if acc is None:
                    continue
                candidates.append(
                    {
                        "system": system,
                        "score": round(float(acc) * 100, 1),
                        "ctx_len": result.get("memory_context_chars"),
                        "correct": result.get("correct"),
                        "total": result.get("total"),
                    }
                )
            except Exception:
                continue

        return candidates

    def _load_current_run_test_frontier(self, logs_dir: Optional[Path]) -> dict:
        """Load test frontier restricted to candidates tested in this run.

        The benchmark frontier file is global and may retain old systems from
        previous runs. For ablation reporting we only want current-run generated
        agents, so this method derives the candidate set from .launcher test logs.
        """
        if not logs_dir or not logs_dir.exists():
            return self._empty_current_test_frontier()

        fallback = self._load_frontier(logs_dir, "frontier.json", "test_accuracy")
        launcher_dir = logs_dir / ".launcher"
        frontier_by_system = {
            row.get("system"): row
            for row in fallback.get("pareto", [])
            if isinstance(row, dict) and row.get("system")
        }
        candidates: list[dict] = []
        candidates_source = "current_run_launcher"

        test_logs = sorted(launcher_dir.glob("*test*.log")) if launcher_dir.exists() else []
        for log_file in test_logs:
            try:
                text = log_file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            system = self._extract_test_system_from_launcher_log(log_file, text)
            if not system or not system.startswith("mem_agent_"):
                continue

            score = _NULL
            ctx_len = _NULL
            correct = _NULL
            total = _NULL

            result_match = re.search(r"Saved test results to\s+(.+?test\.json)", text)
            if result_match:
                result_path = Path(result_match.group(1).strip())
                if self._result_json_matches_launcher(result_path, log_file):
                    try:
                        result = json.loads(result_path.read_text(encoding="utf-8"))
                        acc = result.get("accuracy")
                        if acc is not None:
                            score = round(float(acc) * 100, 1)
                        ctx_len = result.get("memory_context_chars")
                        correct = result.get("correct")
                        total = result.get("total")
                    except Exception:
                        pass

            if score is None:
                done_match = re.search(r"Done:.*?\btest=([0-9.]+)%", text)
                if done_match:
                    score = round(float(done_match.group(1)), 1)

            if ctx_len is None and system in frontier_by_system:
                ctx_len = frontier_by_system[system].get("ctx_len")

            if score is None:
                continue

            candidates.append(
                {
                    "system": system,
                    "score": score,
                    "ctx_len": ctx_len,
                    "correct": correct,
                    "total": total,
                }
            )

        if not candidates:
            candidates = self._current_run_test_results_from_files(logs_dir)
            candidates_source = "current_run_results"

        if not candidates:
            return self._empty_current_test_frontier()

        candidates.sort(
            key=lambda row: (
                float(row.get("score") or 0),
                -int(row.get("ctx_len") or 10**12),
            ),
            reverse=True,
        )
        best = candidates[0]
        return {
            "best_system": best.get("system"),
            "best_score": best.get("score"),
            "best_ctx_len": best.get("ctx_len"),
            "best_correct": best.get("correct"),
            "best_total": best.get("total"),
            "pareto": candidates,
            "source": candidates_source,
        }

    def _proposer_tokens(self) -> dict:
        """Aggregate proposer LLM usage from iter_*/proposer_usage.jsonl.

        This is the clean proposer-only comparison token metric. It excludes
        classifier/evaluation tokens and excludes byte/char proxy costs.
        """
        sources = {
            "provider": {"prompt": 0, "completion": 0, "total": 0, "calls": 0},
            "estimated": {"prompt": 0, "completion": 0, "total": 0, "calls": 0},
            "cached": {"prompt": 0, "completion": 0, "total": 0, "calls": 0},
        }
        retry = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}
        rows_seen = 0
        models = set()

        for usage_file in sorted(self.evo_dir.glob("iter_*/proposer_usage.jsonl")):
            for line in usage_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                rows_seen += 1
                if row.get("model"):
                    models.add(str(row["model"]))
                source = row.get("usage_source") or "estimated"
                if source not in sources:
                    source = "estimated"
                prompt = int(row.get("prompt_tokens") or 0)
                completion = int(row.get("completion_tokens") or 0)
                total = int(row.get("total_tokens") or prompt + completion)

                sources[source]["prompt"] += prompt
                sources[source]["completion"] += completion
                sources[source]["total"] += total
                sources[source]["calls"] += 1

                try:
                    attempt = int(row.get("attempt") or 1)
                except Exception:
                    attempt = 1
                if attempt > 1:
                    retry["prompt"] += prompt
                    retry["completion"] += completion
                    retry["total"] += total
                    retry["calls"] += 1

        return {
            "model": ", ".join(sorted(models)) if models else _NULL,
            "provider_prompt_tokens": sources["provider"]["prompt"],
            "provider_completion_tokens": sources["provider"]["completion"],
            "provider_total_tokens": sources["provider"]["total"],
            "provider_calls": sources["provider"]["calls"],
            "estimated_prompt_tokens": sources["estimated"]["prompt"],
            "estimated_completion_tokens": sources["estimated"]["completion"],
            "estimated_total_tokens": sources["estimated"]["total"],
            "estimated_calls": sources["estimated"]["calls"],
            "cached_prompt_tokens": sources["cached"]["prompt"],
            "cached_completion_tokens": sources["cached"]["completion"],
            "cached_total_tokens": sources["cached"]["total"],
            "cached_calls": sources["cached"]["calls"],
            "retry_prompt_tokens": retry["prompt"],
            "retry_completion_tokens": retry["completion"],
            "retry_total_tokens": retry["total"],
            "retry_calls": retry["calls"],
            "calls": rows_seen,
            "usage_source_breakdown": {
                "provider": sources["provider"]["calls"],
                "estimated": sources["estimated"]["calls"],
                "cached": sources["cached"]["calls"],
            },
            "has_provider_usage": sources["provider"]["calls"] > 0,
            "has_mixed_usage_sources": sum(
                1 for s in sources.values() if s["calls"] > 0
            )
            > 1,
        }

    def _harness_execution_tokens(self, logs_dir: Optional[Path]) -> dict:
        """Aggregate classifier/evaluation LLM usage from val.json/test.json.

        Provider tokens are real API usage. Cached tokens are reported
        separately because they do not represent new provider calls in this run.
        """
        totals = {
            "provider": {"prompt": 0, "completion": 0, "total": 0, "calls": 0},
            "estimated": {"prompt": 0, "completion": 0, "total": 0, "calls": 0},
            "cached": {"prompt": 0, "completion": 0, "total": 0, "calls": 0},
        }
        result_files = 0
        roots = [logs_dir] if logs_dir and logs_dir.exists() else []

        for root in roots:
            for result_file in list(root.rglob("val.json")) + list(root.rglob("test.json")):
                try:
                    data = json.loads(result_file.read_text(encoding="utf-8"))
                except Exception:
                    continue
                result_files += 1
                summary = data.get("classifier_usage_source_summary")
                if isinstance(summary, dict):
                    for source in ("provider", "estimated", "cached"):
                        values = summary.get(source, {}) or {}
                        prompt = int(values.get("prompt_tokens") or 0)
                        completion = int(values.get("completion_tokens") or 0)
                        total = int(values.get("total_tokens") or prompt + completion)
                        calls = int(values.get("calls") or 0)
                        totals[source]["prompt"] += prompt
                        totals[source]["completion"] += completion
                        totals[source]["total"] += total
                        totals[source]["calls"] += calls
                else:
                    # Backward-compatible fallback: old llm_* fields lacked
                    # source attribution, so they are estimates, not provider API usage.
                    prompt = int(data.get("llm_input_tokens") or 0)
                    completion = int(data.get("llm_output_tokens") or 0)
                    total = int(data.get("llm_total_tokens") or prompt + completion)
                    calls = int(data.get("llm_calls") or 0)
                    totals["estimated"]["prompt"] += prompt
                    totals["estimated"]["completion"] += completion
                    totals["estimated"]["total"] += total
                    totals["estimated"]["calls"] += calls

        models = set()
        roots = [logs_dir] if logs_dir and logs_dir.exists() else []
        for root in roots:
            for result_file in list(root.rglob("val.json")) + list(root.rglob("test.json")):
                parts = result_file.parts
                if len(parts) >= 2:
                    # logs/<run>/<dataset>/<memory>/<model_short>/<split>.json
                    models.add(parts[-2])

        return {
            "model": ", ".join(sorted(models)) if models else _NULL,
            "classifier_provider_prompt_tokens": totals["provider"]["prompt"],
            "classifier_provider_completion_tokens": totals["provider"]["completion"],
            "classifier_provider_total_tokens": totals["provider"]["total"],
            "classifier_provider_calls": totals["provider"]["calls"],
            "classifier_estimated_prompt_tokens": totals["estimated"]["prompt"],
            "classifier_estimated_completion_tokens": totals["estimated"]["completion"],
            "classifier_estimated_total_tokens": totals["estimated"]["total"],
            "classifier_estimated_calls": totals["estimated"]["calls"],
            "classifier_cached_prompt_tokens": totals["cached"]["prompt"],
            "classifier_cached_completion_tokens": totals["cached"]["completion"],
            "classifier_cached_total_tokens": totals["cached"]["total"],
            "classifier_cached_calls": totals["cached"]["calls"],
            "classifier_calls": (
                totals["provider"]["calls"]
                + totals["estimated"]["calls"]
                + totals["cached"]["calls"]
            ),
            "result_files": result_files,
            "usage_source_breakdown": {
                "provider": totals["provider"]["calls"],
                "estimated": totals["estimated"]["calls"],
                "cached": totals["cached"]["calls"],
            },
            "has_provider_usage": totals["provider"]["calls"] > 0,
            "has_mixed_usage_sources": sum(
                1 for s in totals.values() if s["calls"] > 0
            )
            > 1,
        }

    def _total_api_tokens(self, proposer_tokens: dict, harness_tokens: dict) -> dict:
        proposer_provider = int(proposer_tokens.get("provider_total_tokens") or 0)
        proposer_estimated = int(proposer_tokens.get("estimated_total_tokens") or 0)
        proposer_cached = int(proposer_tokens.get("cached_total_tokens") or 0)
        harness_provider = int(
            harness_tokens.get("classifier_provider_total_tokens") or 0
        )
        harness_estimated = int(
            harness_tokens.get("classifier_estimated_total_tokens") or 0
        )
        harness_cached = int(
            harness_tokens.get("classifier_cached_total_tokens") or 0
        )
        provider_total = proposer_provider + harness_provider
        estimated_total = proposer_estimated + harness_estimated
        cached_total = proposer_cached + harness_cached
        return {
            "provider_total_tokens": provider_total,
            "estimated_total_tokens": estimated_total,
            "cached_total_tokens": cached_total,
            "has_provider_usage": provider_total > 0,
            "has_mixed_usage_sources": sum(
                1 for value in (provider_total, estimated_total, cached_total) if value > 0
            )
            > 1,
        }

    def _iteration_count(self, episodes: list[dict]) -> int:
        iterations = {
            int(ep.get("iteration"))
            for ep in episodes
            if ep.get("iteration") is not None
        }
        if iterations:
            return len(iterations)

        evo_iters = [p for p in self.evo_dir.glob("iter_*") if p.is_dir()]
        return len(evo_iters)

    def _real_api_token_averages(
        self,
        episodes: list[dict],
        proposer_tokens: dict,
        harness_tokens: dict,
    ) -> dict:
        def _avg(total: object, denom: object) -> Optional[float]:
            try:
                total_f = float(total or 0)
                denom_f = float(denom or 0)
            except Exception:
                return _NULL
            if denom_f <= 0:
                return _NULL
            return round(total_f / denom_f, 1)

        iterations = self._iteration_count(episodes)
        proposer_total = proposer_tokens.get("provider_total_tokens", 0)
        proposer_calls = proposer_tokens.get("provider_calls", 0)
        classifier_total = harness_tokens.get("classifier_provider_total_tokens", 0)
        classifier_calls = harness_tokens.get("classifier_provider_calls", 0)

        return {
            "iterations": iterations,
            "proposer_provider_calls": proposer_calls,
            "classifier_provider_calls": classifier_calls,
            "proposer_provider_avg_per_iter": _avg(proposer_total, iterations),
            "proposer_provider_avg_per_call": _avg(proposer_total, proposer_calls),
            "classifier_provider_avg_per_iter": _avg(classifier_total, iterations),
            "classifier_provider_avg_per_call": _avg(classifier_total, classifier_calls),
        }

    def _adaptive_policy(self, episodes: list[dict], evidence: dict, guidance: dict) -> dict:
        def _dist(key: str) -> dict:
            counts: dict[str, int] = {}
            for ep in episodes:
                value = ep.get(key)
                if value:
                    counts[str(value)] = counts.get(str(value), 0) + 1
            return counts

        maturity = evidence.get("evidence_maturity") or guidance.get("evidence_maturity") or {}
        utility = guidance.get("memory_utility_summary", {}) or {}
        return {
            "evidence_maturity": maturity.get("maturity"),
            "has_positive_anchor": maturity.get("has_positive_anchor", False),
            "has_effective_strategy": maturity.get("has_effective_strategy", False),
            "has_confirmed_direction": maturity.get("has_confirmed_direction", False),
            "positive_anchor_count": len(evidence.get("positive_anchors", [])),
            "effective_strategy_count": len(evidence.get("strategy_evidence", [])),
            "component_strategy_count": evidence.get("component_strategy_count", 0),
            "interaction_strategy_count": evidence.get("interaction_strategy_count", 0),
            "candidate_strategy_count": evidence.get("candidate_strategy_count", 0),
            "usable_strategy_count": utility.get("usable_strategy_count", 0),
            "historical_effective_count": utility.get(
                "historical_effective_count",
                len(evidence.get("strategy_evidence", [])),
            ),
            "stale_strategy_count": utility.get("stale_strategy_count", 0),
            "cooling_down_count": utility.get("cooling_down_count", 0),
            "reactivated_memory_count": utility.get("reactivated_memory_count", 0),
            "utility_pruned_strategy_count": utility.get(
                "utility_pruned_strategy_count", 0
            ),
            "avg_memory_utility_score": utility.get("avg_utility_score", 0.0),
            "clear_positive_count": evidence.get("clear_positive_count", 0),
            "clear_regression_count": evidence.get("clear_regression_count", 0),
            "ambiguous_positive_count": evidence.get("ambiguous_positive_count", 0),
            "ambiguous_regression_count": evidence.get("ambiguous_regression_count", 0),
            "avoid_only_guidance": maturity.get("avoid_only_guidance", False),
            "recommended_reading_strategy": maturity.get("recommended_reading_strategy"),
            "recommended_governance_mode": maturity.get("recommended_governance_mode"),
            "reading_strategy_distribution": _dist("reading_mode"),
            "governance_mode_distribution": _dist("governance_mode"),
            "candidate_role_distribution": _dist("candidate_role"),
            "latest_guidance_governance_mode": (
                guidance.get("governance", {}) or {}
            ).get("mode"),
        }

    def _avoid_memory_stats(self) -> dict:
        try:
            from memory.updating.memory_utility import MemoryUtilityTracker

            tracker = MemoryUtilityTracker(self.output_dir)
            states = tracker.load_states()
            items = states.get("memory_items", {})
            avoid_items = {
                k: v for k, v in items.items()
                if v.get("memory_type") == "avoid"
            }
            return {
                "active_avoid_count": sum(
                    1 for v in avoid_items.values()
                    if v.get("current_status") == "active_avoid"
                ),
                "contradicted_count": sum(
                    1 for v in avoid_items.values()
                    if v.get("current_status") == "contradicted"
                ),
                "reactivated_hypothesis_count": sum(
                    1 for v in avoid_items.values()
                    if v.get("current_status") == "reactivated_hypothesis"
                ),
                "strengthened_count": sum(
                    1 for v in avoid_items.values()
                    if v.get("current_status") == "active_avoid"
                    and v.get("strengthened_count", 0) > 0
                ),
                "avg_avoid_strength": (
                    round(
                        sum(float(v.get("avoid_strength", 1.0) or 1.0)
                            for v in avoid_items.values()) / len(avoid_items),
                        3,
                    )
                    if avoid_items else _NULL
                ),
                "retest_pool_size": len([
                    v for v in avoid_items.values()
                    if v.get("current_status") == "active_avoid"
                    and float(v.get("avoid_strength", 1.0) or 1.0) > 0.3
                ]),
            }
        except Exception:
            return {}

    def _task_outcome_stats(self) -> dict:
        try:
            path = self.output_dir / "task_outcomes.json"
            if not path.exists():
                return {}
            from memory.encoding.task_outcome_memory import TaskOutcomeMemory

            tracker = TaskOutcomeMemory(self.output_dir)
            guidance = tracker.guidance_summary()
            latest = guidance.get("latest_summary", {}) or {}
            return {
                "tracked_task_count": guidance.get("tracked_task_count", 0),
                "solved_task_count": guidance.get("solved_task_count", 0),
                "latest_episode": guidance.get("latest_episode"),
                "latest_passes": latest.get("passes", 0),
                "latest_partial_passes": latest.get("partial_passes", 0),
                "latest_failures": latest.get("failures", 0),
                "persistent_failure_categories": guidance.get(
                    "persistent_failure_categories", {}
                ),
                "recent_new_pass_episode_count": len(
                    guidance.get("recent_new_passes", [])
                ),
                "recent_regression_episode_count": len(
                    guidance.get("recent_regressions", [])
                ),
            }
        except Exception:
            return {}

    # ── Metric computations ───────────────────────────────────

    def _build_score_curve(self, episodes: list[dict]) -> dict:
        from collections import defaultdict
        by_iter: dict = defaultdict(list)
        for e in episodes:
            if e.get("score") is not None:
                by_iter[e["iteration"]].append(float(e["score"]))
        return {i: max(scores) for i, scores in by_iter.items()}

    def _get_best_iter(self, score_curve: dict) -> Optional[int]:
        if not score_curve:
            return None
        best_score = max(score_curve.values())
        for i in sorted(score_curve.keys()):
            if score_curve[i] >= best_score:
                return i
        return None

    def _get_token_to_best(self, best_iter: Optional[int]) -> Optional[int]:
        if best_iter is None:
            return None
        try:
            total = 0
            for usage_file in sorted(self.evo_dir.glob("iter_*/proposer_usage.jsonl")):
                for line in usage_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    u = json.loads(line)
                    if int(u.get("iteration", 0)) <= best_iter:
                        total += int(u.get("total_tokens", 0))
            return total if total > 0 else None
        except Exception:
            return None

    def _role_scores(self, episodes: list[dict]) -> dict:
        from collections import defaultdict
        role_scores: dict = defaultdict(list)
        for e in episodes:
            role = e.get("candidate_role") or "unknown"
            score = e.get("score")
            if score is not None:
                role_scores[role].append(float(score))
        return {
            role: {
                "best": max(scores),
                "avg": round(sum(scores) / len(scores), 1),
                "count": len(scores),
            }
            for role, scores in role_scores.items()
        }

    def _search_efficiency(
        self, episodes: list[dict], baseline: Optional[float]
    ) -> dict:
        if not episodes:
            return {
                "evaluations_to_target": _NULL,
                "best_score_at_budget": {},
                "token_cost_to_best": _NULL,
                "area_under_curve": _NULL,
            }

        scores = [float(ep.get("score", 0)) for ep in episodes]

        # evaluations_to_target
        evals_to_target: Optional[int] = _NULL
        if baseline is not None:
            for i, s in enumerate(scores):
                if s >= baseline:
                    evals_to_target = i + 1
                    break

        # best_score_at_budget
        best_at_budget: dict[str, Optional[float]] = {}
        for budget in [5, 10, 15, 20]:
            subset = scores[:budget]
            best_at_budget[str(budget)] = round(max(subset), 1) if subset else _NULL

        # token_cost_to_best
        best_score = max(scores) if scores else 0.0
        best_idx = next((i for i, s in enumerate(scores) if s == best_score), None)
        token_cost_to_best: Optional[int] = _NULL
        if best_idx is not None:
            token_cost_to_best = sum(
                int(ep.get("token_cost", 0)) for ep in episodes[: best_idx + 1]
            )

        # area_under_curve (best-so-far trapz, normalized to [0,1])
        auc: Optional[float] = _NULL
        if len(scores) >= 2:
            bsf: list[float] = []
            current = 0.0
            for s in scores:
                current = max(current, s)
                bsf.append(current)
            trapz = sum((bsf[i] + bsf[i + 1]) / 2 for i in range(len(bsf) - 1))
            max_trapz = 100.0 * (len(bsf) - 1)
            auc = round(trapz / max_trapz, 4) if max_trapz > 0 else _NULL
        elif len(scores) == 1:
            auc = round(scores[0] / 100, 4)

        score_curve = self._build_score_curve(episodes)
        best_iter = self._get_best_iter(score_curve)
        token_to_best = self._get_token_to_best(best_iter)

        return {
            "evaluations_to_target": evals_to_target,
            "best_score_at_budget": best_at_budget,
            "token_cost_to_best": token_cost_to_best,
            "area_under_curve": auc,
            "score_curve": score_curve,
            "best_iter": best_iter,
            "token_to_best": token_to_best,
        }

    def _memory_effectiveness(
        self,
        episodes: list[dict],
        evidence: dict,
        guidance: dict,
        plans: dict[str, dict],
    ) -> dict:
        avoid_directions: set[str] = set()
        for av in guidance.get("avoid", []):
            d = str(av.get("direction", "")).lower().strip()
            if d:
                avoid_directions.add(d)

        # repeated_failure_rate: among regression episodes, how many had
        # change_family already in avoid list
        regression_eps = [
            ep for ep in episodes if ep.get("status") == "regression"
        ]
        repeated_failures = 0
        regression_with_plan = 0
        for ep in regression_eps:
            eid = ep.get("episode_id", "")
            plan = plans.get(eid, {})
            cf = str(plan.get("change_family", "")).lower().strip()
            if cf:
                regression_with_plan += 1
                if cf in avoid_directions:
                    repeated_failures += 1
        repeated_failure_rate: Optional[float] = (
            round(repeated_failures / regression_with_plan, 3)
            if regression_with_plan > 0
            else _NULL
        )

        # promising_direction_reuse_rate: promising families reused after first seen
        promising_families: list[dict] = []
        for comp_data in evidence.get("components", {}).values():
            for fam in comp_data.get("change_families", []):
                if fam.get("status") == "promising":
                    promising_families.append(fam)

        reused = sum(
            1
            for fam in promising_families
            if len(fam.get("supporting_episodes", [])) > 1
        )
        promising_direction_reuse_rate: Optional[float] = (
            round(reused / len(promising_families), 3)
            if promising_families
            else _NULL
        )

        # evidence_hit_rate: best candidate's change_family in high_priority
        best_eps = max(episodes, key=lambda e: float(e.get("score", 0)), default=None)
        evidence_hit_rate: Optional[float] = _NULL
        if best_eps:
            best_plan = plans.get(best_eps.get("episode_id", ""), {})
            best_cf = str(best_plan.get("change_family", "")).lower().strip()
            if best_cf:
                hp_directions = [
                    str(hp.get("direction", "")).lower().strip()
                    for hp in guidance.get("high_priority", [])
                ]
                evidence_hit_rate = 1.0 if best_cf in hp_directions else 0.0

        return {
            "repeated_failure_rate": repeated_failure_rate,
            "promising_direction_reuse_rate": promising_direction_reuse_rate,
            "evidence_hit_rate": evidence_hit_rate,
        }

    def _attribution(
        self,
        episodes: list[dict],
        evidence: dict,
        plans: dict[str, dict],
    ) -> dict:
        audit = evidence.get("attribution_audit", [])

        single_ratio: Optional[float] = _NULL
        ambiguous_ratio: Optional[float] = _NULL
        if audit:
            n = len(audit)
            single_count = sum(1 for r in audit if r.get("attribution_type") == "clear")
            ambiguous_count = sum(
                1 for r in audit if r.get("attribution_type") == "ambiguous"
            )
            single_ratio = round(single_count / n, 3)
            ambiguous_ratio = round(ambiguous_count / n, 3)

        # proposal_plan_compliance_rate
        plan_compliance: Optional[float] = _NULL
        valid_plan_rate: Optional[float] = _NULL
        if plans:
            from memory.steering.proposal_plan import ProposalPlanManager

            try:
                mgr = ProposalPlanManager(
                    output_dir=self.output_dir, evo_dir=self.evo_dir
                )
                stats = mgr.compute_session_compliance()
                declared = stats.get("declared_compliance", stats)
                plan_compliance = declared.get("compliance_rate")
            except Exception:
                plan_compliance = _NULL

            total_eps = len(episodes)
            valid_plan_rate = (
                round(len(plans) / total_eps, 3) if total_eps > 0 else _NULL
            )

        return {
            "single_component_edit_ratio": single_ratio,
            "ambiguous_ratio": ambiguous_ratio,
            "proposal_plan_compliance_rate": plan_compliance,
            "valid_plan_rate": valid_plan_rate,
        }

    def _stability(self, episodes: list[dict]) -> dict:
        if not episodes:
            return {
                "valid_candidate_rate": _NULL,
                "regression_rate": _NULL,
                "bug_rate": _NULL,
                "score_variance": _NULL,
            }
        n = len(episodes)
        statuses = [ep.get("status", "") for ep in episodes]
        scores = [float(ep.get("score", 0)) for ep in episodes]

        valid_rate = round(sum(1 for s in statuses if s == "success") / n, 3)
        regression_rate = round(sum(1 for s in statuses if s == "regression") / n, 3)
        bug_rate = round(sum(1 for s in statuses if s == "bug") / n, 3)

        variance: Optional[float] = _NULL
        if len(scores) >= 2:
            mean = sum(scores) / len(scores)
            var = sum((s - mean) ** 2 for s in scores) / (len(scores) - 1)
            variance = round(math.sqrt(var), 2)

        return {
            "valid_candidate_rate": valid_rate,
            "regression_rate": regression_rate,
            "bug_rate": bug_rate,
            "score_variance": variance,
        }

    # ── Public API ────────────────────────────────────────────

    def collect(
        self,
        session_id: str,
        logs_dir: Optional[Path] = None,
        config_override: Optional[dict] = None,
    ) -> dict:
        """Collect all metrics for this session. Returns the metrics dict."""
        episodes = self._load_episodes()
        evidence = self._load_component_evidence()
        guidance = self._load_search_guidance()
        plans = self._load_proposal_plans()
        baseline = self._get_fewshot_baseline(logs_dir)
        frontier = {
            "val": self._load_frontier(logs_dir, "frontier_val.json", "val_accuracy"),
            "test": self._load_current_run_test_frontier(logs_dir),
        }
        proposer_tokens = self._proposer_tokens()
        harness_execution_tokens = self._harness_execution_tokens(logs_dir)
        total_api_tokens = self._total_api_tokens(
            proposer_tokens, harness_execution_tokens
        )
        real_api_token_averages = self._real_api_token_averages(
            episodes, proposer_tokens, harness_execution_tokens
        )

        # Load memory config for metadata
        mem_cfg: dict = {}
        if config_override:
            mem_cfg = config_override
        else:
            try:
                from memory.config_loader import get_memory_config
                mem_cfg = get_memory_config()
            except Exception:
                pass

        return {
            "session_id": session_id,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "total_episodes": len(episodes),
            "fewshot_all_baseline": round(baseline, 1) if baseline is not None else _NULL,
            "config": {
                "enabled": mem_cfg.get("enabled", False),
                "evidence_quality": mem_cfg.get("evidence_quality", False),
                "memory_efficiency": mem_cfg.get("memory_efficiency", False),
                "search_governance": mem_cfg.get("search_governance", False),
            },
            "search_efficiency": self._search_efficiency(episodes, baseline),
            "memory_effectiveness": self._memory_effectiveness(
                episodes, evidence, guidance, plans
            ),
            "attribution": self._attribution(episodes, evidence, plans),
            "stability": self._stability(episodes),
            "frontier": frontier,
            "proposer_tokens": proposer_tokens,
            "harness_execution_tokens": harness_execution_tokens,
            "total_api_tokens": total_api_tokens,
            "real_api_token_averages": real_api_token_averages,
            "real_api_tokens": {
                "proposer_provider_total": proposer_tokens.get(
                    "provider_total_tokens", 0
                ),
                "proposer_provider_calls": proposer_tokens.get("provider_calls", 0),
                "classifier_provider_total": harness_execution_tokens.get(
                    "classifier_provider_total_tokens", 0
                ),
                "classifier_provider_calls": harness_execution_tokens.get(
                    "classifier_provider_calls", 0
                ),
                "total_provider_tokens": total_api_tokens.get(
                    "provider_total_tokens", 0
                ),
                "proposer_cached_total": proposer_tokens.get(
                    "cached_total_tokens", 0
                ),
                "classifier_cached_total": harness_execution_tokens.get(
                    "classifier_cached_total_tokens", 0
                ),
                "proposer_estimated_total": proposer_tokens.get(
                    "estimated_total_tokens", 0
                ),
                "classifier_estimated_total": harness_execution_tokens.get(
                    "classifier_estimated_total_tokens", 0
                ),
            },
            "adaptive_policy": self._adaptive_policy(episodes, evidence, guidance),
            "avoid_memory": self._avoid_memory_stats(),
            "task_outcome_memory": self._task_outcome_stats(),
            "role_scores": self._role_scores(episodes),
        }

    def save(self, metrics: dict) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    def load_all(self) -> list[dict]:
        """Load all saved metrics records."""
        if not self.metrics_path.exists():
            return []
        records = []
        for line in self.metrics_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    # ── Report generation ────────────────────────────────────

    def generate_comparison_report(self, session_ids: list[str]) -> str:
        all_records = self.load_all()
        by_session: dict[str, dict] = {}
        for rec in all_records:
            sid = rec.get("session_id", "")
            if sid in session_ids:
                by_session[sid] = rec

        ordered_sids = [s for s in session_ids if s in by_session]
        if not ordered_sids:
            return "No matching sessions found in metrics.jsonl"

        def _fmt(val: object, suffix: str = "") -> str:
            if val is None:
                return "—"
            if isinstance(val, float):
                return f"{val:.1f}{suffix}"
            return f"{val}{suffix}"

        def _pct(val: object) -> str:
            if val is None:
                return "—"
            return f"{float(val) * 100:.0f}%"

        def _num(val: object) -> str:
            if val is None:
                return "—"
            try:
                return f"{int(val):,}"
            except Exception:
                return str(val)

        header_labels = {s: s.replace("ablation_", "").replace("_", " ").title() for s in ordered_sids}

        rows: list[tuple[str, list[str]]] = []

        def _row(label: str, extractor) -> tuple[str, list[str]]:
            return (label, [extractor(by_session[s]) for s in ordered_sids])

        def _g(d: dict, *keys: str):
            val = d
            for k in keys:
                if not isinstance(val, dict):
                    return None
                val = val.get(k)
            return val

        rows.append(_row(
            "Best Val Frontier (%)",
            lambda r: _fmt(_g(r, "frontier", "val", "best_score"), "%"),
        ))
        rows.append(_row(
            "Best Val System",
            lambda r: _fmt(_g(r, "frontier", "val", "best_system")),
        ))
        rows.append(_row(
            "Best Test Frontier (%)",
            lambda r: _fmt(_g(r, "frontier", "test", "best_score"), "%"),
        ))
        rows.append(_row(
            "Best Test System",
            lambda r: _fmt(_g(r, "frontier", "test", "best_system")),
        ))
        rows.append(_row(
            "Best Test Ctx Len",
            lambda r: _num(_g(r, "frontier", "test", "best_ctx_len")),
        ))
        rows.append(("", []))
        rows.append(_row(
            "Evidence Maturity",
            lambda r: _fmt(_g(r, "adaptive_policy", "evidence_maturity")),
        ))
        rows.append(_row(
            "Positive Anchors",
            lambda r: _num(_g(r, "adaptive_policy", "positive_anchor_count")),
        ))
        rows.append(_row(
            "Effective Strategies",
            lambda r: _num(_g(r, "adaptive_policy", "effective_strategy_count")),
        ))
        rows.append(_row(
            "Usable / Stale Strategies",
            lambda r: (
                f"{_num(_g(r, 'adaptive_policy', 'usable_strategy_count'))} / "
                f"{_num(_g(r, 'adaptive_policy', 'stale_strategy_count'))}"
            ),
        ))
        rows.append(_row(
            "Avg Memory Utility",
            lambda r: _fmt(_g(r, "adaptive_policy", "avg_memory_utility_score")),
        ))
        rows.append(_row(
            "Strategy Scope C/I/Cand",
            lambda r: (
                f"{_num(_g(r, 'adaptive_policy', 'component_strategy_count'))} / "
                f"{_num(_g(r, 'adaptive_policy', 'interaction_strategy_count'))} / "
                f"{_num(_g(r, 'adaptive_policy', 'candidate_strategy_count'))}"
            ),
        ))
        rows.append(_row(
            "Clear Positive / Regression",
            lambda r: (
                f"{_num(_g(r, 'adaptive_policy', 'clear_positive_count'))} / "
                f"{_num(_g(r, 'adaptive_policy', 'clear_regression_count'))}"
            ),
        ))
        rows.append(_row(
            "Ambiguous Positive / Regression",
            lambda r: (
                f"{_num(_g(r, 'adaptive_policy', 'ambiguous_positive_count'))} / "
                f"{_num(_g(r, 'adaptive_policy', 'ambiguous_regression_count'))}"
            ),
        ))
        rows.append(_row(
            "Reading Strategy Dist.",
            lambda r: _fmt(_g(r, "adaptive_policy", "reading_strategy_distribution")),
        ))
        rows.append(_row(
            "Governance Mode Dist.",
            lambda r: _fmt(_g(r, "adaptive_policy", "governance_mode_distribution")),
        ))
        rows.append(("", []))
        rows.append(_row(
            "PROPOSER_MODEL",
            lambda r: _fmt(_g(r, "proposer_tokens", "model")),
        ))
        rows.append(_row(
            "PROPOSER_MODEL Prompt Tokens",
            lambda r: _num(
                (_g(r, "proposer_tokens", "provider_prompt_tokens") or 0)
                + (_g(r, "proposer_tokens", "estimated_prompt_tokens") or 0)
            ) if r.get("proposer_tokens") else "—",
        ))
        rows.append(_row(
            "PROPOSER_MODEL Completion Tokens",
            lambda r: _num(
                (_g(r, "proposer_tokens", "provider_completion_tokens") or 0)
                + (_g(r, "proposer_tokens", "estimated_completion_tokens") or 0)
            ) if r.get("proposer_tokens") else "—",
        ))
        rows.append(_row(
            "PROPOSER_MODEL Total Tokens",
            lambda r: _num(
                (_g(r, "proposer_tokens", "provider_total_tokens") or 0)
                + (_g(r, "proposer_tokens", "estimated_total_tokens") or 0)
            ) if r.get("proposer_tokens") else "—",
        ))
        rows.append(_row(
            "PROPOSER_MODEL Provider Tokens",
            lambda r: _num(_g(r, "proposer_tokens", "provider_total_tokens")),
        ))
        rows.append(_row(
            "PROPOSER_MODEL Avg Provider / Iter",
            lambda r: _fmt(
                _g(r, "real_api_token_averages", "proposer_provider_avg_per_iter")
            ),
        ))
        rows.append(_row(
            "PROPOSER_MODEL Avg Provider / Call",
            lambda r: _fmt(
                _g(r, "real_api_token_averages", "proposer_provider_avg_per_call")
            ),
        ))
        rows.append(_row(
            "PROPOSER_MODEL Estimated Tokens",
            lambda r: _num(_g(r, "proposer_tokens", "estimated_total_tokens")),
        ))
        rows.append(_row(
            "PROPOSER_MODEL Retry Tokens",
            lambda r: _num(_g(r, "proposer_tokens", "retry_total_tokens")),
        ))
        rows.append(_row(
            "PROPOSER_MODEL Calls",
            lambda r: _num(_g(r, "proposer_tokens", "calls")),
        ))
        rows.append(_row(
            "CLASSIFIER_MODEL",
            lambda r: _fmt(_g(r, "harness_execution_tokens", "model")),
        ))
        rows.append(_row(
            "CLASSIFIER_MODEL Provider Tokens",
            lambda r: _num(_g(r, "harness_execution_tokens", "classifier_provider_total_tokens")),
        ))
        rows.append(_row(
            "CLASSIFIER_MODEL Avg Provider / Iter",
            lambda r: _fmt(
                _g(r, "real_api_token_averages", "classifier_provider_avg_per_iter")
            ),
        ))
        rows.append(_row(
            "CLASSIFIER_MODEL Avg Provider / Call",
            lambda r: _fmt(
                _g(r, "real_api_token_averages", "classifier_provider_avg_per_call")
            ),
        ))
        rows.append(_row(
            "CLASSIFIER_MODEL Estimated Tokens",
            lambda r: _num(_g(r, "harness_execution_tokens", "classifier_estimated_total_tokens")),
        ))
        rows.append(_row(
            "CLASSIFIER_MODEL Cached Tokens",
            lambda r: _num(_g(r, "harness_execution_tokens", "classifier_cached_total_tokens")),
        ))
        rows.append(_row(
            "CLASSIFIER_MODEL Calls",
            lambda r: _num(_g(r, "harness_execution_tokens", "classifier_calls")),
        ))
        rows.append(_row(
            "Total Provider API Tokens",
            lambda r: _num(_g(r, "total_api_tokens", "provider_total_tokens")),
        ))
        rows.append(_row(
            "Total Estimated Tokens",
            lambda r: _num(_g(r, "total_api_tokens", "estimated_total_tokens")),
        ))
        rows.append(_row(
            "Total Cached Tokens",
            lambda r: _num(_g(r, "total_api_tokens", "cached_total_tokens")),
        ))
        rows.append(("", []))
        rows.append(_row(
            "Best Val Candidate (%)",
            lambda r: _fmt(
                max(
                    (v for v in r.get("search_efficiency", {}).get("best_score_at_budget", {}).values() if v is not None),
                    default=None,
                ),
                "%",
            ),
        ))
        rows.append(_row(
            "Eval to Target",
            lambda r: _fmt(_g(r, "search_efficiency", "evaluations_to_target")),
        ))
        rows.append(_row(
            "Area Under Curve",
            lambda r: _fmt(_g(r, "search_efficiency", "area_under_curve")),
        ))
        rows.append(_row(
            "Token Cost to Best (proxy)",
            lambda r: _fmt(_g(r, "search_efficiency", "token_cost_to_best")),
        ))
        rows.append(("", []))  # separator
        rows.append(_row(
            "Repeated Failure Rate",
            lambda r: _pct(_g(r, "memory_effectiveness", "repeated_failure_rate")),
        ))
        rows.append(_row(
            "Promising Reuse Rate",
            lambda r: _pct(_g(r, "memory_effectiveness", "promising_direction_reuse_rate")),
        ))
        rows.append(_row(
            "Evidence Hit Rate",
            lambda r: _pct(_g(r, "memory_effectiveness", "evidence_hit_rate")),
        ))
        rows.append(("", []))
        rows.append(_row(
            "Single Edit Ratio",
            lambda r: _pct(_g(r, "attribution", "single_component_edit_ratio")),
        ))
        rows.append(_row(
            "Ambiguous Ratio",
            lambda r: _pct(_g(r, "attribution", "ambiguous_ratio")),
        ))
        rows.append(_row(
            "Plan Compliance Rate",
            lambda r: _pct(_g(r, "attribution", "proposal_plan_compliance_rate")),
        ))
        rows.append(_row(
            "Valid Plan Rate",
            lambda r: _pct(_g(r, "attribution", "valid_plan_rate")),
        ))
        rows.append(("", []))
        rows.append(_row(
            "Valid Candidate Rate",
            lambda r: _pct(_g(r, "stability", "valid_candidate_rate")),
        ))
        rows.append(_row(
            "Regression Rate",
            lambda r: _pct(_g(r, "stability", "regression_rate")),
        ))
        rows.append(_row(
            "Bug Rate",
            lambda r: _pct(_g(r, "stability", "bug_rate")),
        ))
        rows.append(_row(
            "Score Variance (std)",
            lambda r: _fmt(_g(r, "stability", "score_variance")),
        ))

        # Build markdown table
        col_w = 22
        hdr_w = 26
        n_cols = len(ordered_sids)
        header_vals = [header_labels[s] for s in ordered_sids]
        lines: list[str] = []
        lines.append(
            f"| {'Metric':<{hdr_w}} | " + " | ".join(f"{h:<{col_w}}" for h in header_vals) + " |"
        )
        lines.append(
            f"|{'─' * (hdr_w + 2)}|" + "|".join(f"{'─' * (col_w + 2)}" for _ in ordered_sids) + "|"
        )
        for label, vals in rows:
            if not label:
                lines.append(
                    f"|{' ':>{hdr_w + 2}}|" + "|".join(f"{' ':>{col_w + 2}}" for _ in ordered_sids) + "|"
                )
                continue
            lines.append(
                f"| {label:<{hdr_w}} | " + " | ".join(f"{v:<{col_w}}" for v in vals) + " |"
            )

        return "\n".join(lines)


if __name__ == "__main__":
    import sys

    collector = MetricsCollector()
    records = collector.load_all()
    print(f"metrics.jsonl has {len(records)} record(s)")
    print()

    if records:
        latest = records[-1]
        print(f"Latest session: {latest.get('session_id')}  ({latest.get('timestamp')})")
        print(f"  total_episodes      : {latest.get('total_episodes')}")
        print(f"  fewshot_all_baseline: {latest.get('fewshot_all_baseline')}")

        se = latest.get("search_efficiency", {})
        print(f"\n  search_efficiency:")
        print(f"    evaluations_to_target : {se.get('evaluations_to_target')}")
        print(f"    best_score_at_budget  : {se.get('best_score_at_budget')}")
        print(f"    token_cost_to_best    : {se.get('token_cost_to_best')}")
        print(f"    area_under_curve      : {se.get('area_under_curve')}")

        me = latest.get("memory_effectiveness", {})
        print(f"\n  memory_effectiveness:")
        print(f"    repeated_failure_rate        : {me.get('repeated_failure_rate')}")
        print(f"    promising_direction_reuse_rate: {me.get('promising_direction_reuse_rate')}")
        print(f"    evidence_hit_rate             : {me.get('evidence_hit_rate')}")

        at = latest.get("attribution", {})
        print(f"\n  attribution:")
        print(f"    single_component_edit_ratio  : {at.get('single_component_edit_ratio')}")
        print(f"    ambiguous_ratio              : {at.get('ambiguous_ratio')}")
        print(f"    proposal_plan_compliance_rate: {at.get('proposal_plan_compliance_rate')}")
        print(f"    valid_plan_rate              : {at.get('valid_plan_rate')}")

        st = latest.get("stability", {})
        print(f"\n  stability:")
        print(f"    valid_candidate_rate: {st.get('valid_candidate_rate')}")
        print(f"    regression_rate     : {st.get('regression_rate')}")
        print(f"    bug_rate            : {st.get('bug_rate')}")
        print(f"    score_variance      : {st.get('score_variance')}")

        fr = latest.get("frontier", {})
        print(f"\n  frontier:")
        print(f"    best_val_system : {fr.get('val', {}).get('best_system')}")
        print(f"    best_val_score  : {fr.get('val', {}).get('best_score')}")
        print(f"    best_test_system: {fr.get('test', {}).get('best_system')}")
        print(f"    best_test_score : {fr.get('test', {}).get('best_score')}")

        pt = latest.get("proposer_tokens", {})
        print(f"\n  proposer_tokens:")
        print(f"    model            : {pt.get('model')}")
        print(f"    prompt_tokens    : {(pt.get('provider_prompt_tokens', 0) or 0) + (pt.get('estimated_prompt_tokens', 0) or 0)}")
        print(f"    completion_tokens: {(pt.get('provider_completion_tokens', 0) or 0) + (pt.get('estimated_completion_tokens', 0) or 0)}")
        print(f"    total_tokens     : {(pt.get('provider_total_tokens', 0) or 0) + (pt.get('estimated_total_tokens', 0) or 0)}")
        print(f"    retry_tokens     : {pt.get('retry_total_tokens')}")

        ht = latest.get("harness_execution_tokens", {})
        print(f"\n  harness_execution_tokens:")
        print(f"    model                     : {ht.get('model')}")
        print(f"    classifier_provider_total : {ht.get('classifier_provider_total_tokens')}")
        print(f"    classifier_estimated_total: {ht.get('classifier_estimated_total_tokens')}")
        print(f"    classifier_cached_total   : {ht.get('classifier_cached_total_tokens')}")

        tt = latest.get("total_api_tokens", {})
        print(f"\n  total_api_tokens:")
        print(f"    provider_total : {tt.get('provider_total_tokens')}")
        print(f"    estimated_total: {tt.get('estimated_total_tokens')}")
        print(f"    cached_total   : {tt.get('cached_total_tokens')}")

        ap = latest.get("adaptive_policy", {})
        print(f"\n  adaptive_policy:")
        print(f"    evidence_maturity       : {ap.get('evidence_maturity')}")
        print(f"    effective_strategy_count: {ap.get('effective_strategy_count')}")
        print(f"    positive_anchor_count   : {ap.get('positive_anchor_count')}")
        print(f"    clear pos/reg           : {ap.get('clear_positive_count')} / {ap.get('clear_regression_count')}")
        print(f"    ambiguous pos/reg       : {ap.get('ambiguous_positive_count')} / {ap.get('ambiguous_regression_count')}")
        print(f"    reading_strategy_dist   : {ap.get('reading_strategy_distribution')}")
        print(f"    governance_mode_dist    : {ap.get('governance_mode_distribution')}")

        print()
        session_ids = [r.get("session_id") for r in records]
        print(collector.generate_comparison_report(session_ids))
    else:
        print("No metrics yet. Run evolution first to populate workspace/memory/")
        if "--test" in sys.argv:
            # Inject a synthetic record for format testing
            synthetic = {
                "session_id": "synthetic_test",
                "timestamp": "2026-04-29T00:00:00",
                "total_episodes": 10,
                "fewshot_all_baseline": 85.0,
                "config": {"enabled": True, "modules": {"episode_recorder": True}},
                "search_efficiency": {
                    "evaluations_to_target": 4,
                    "best_score_at_budget": {"5": 82.0, "10": 88.0, "15": 88.0, "20": 88.0},
                    "token_cost_to_best": 30000,
                    "area_under_curve": 0.74,
                },
                "memory_effectiveness": {
                    "repeated_failure_rate": 0.33,
                    "promising_direction_reuse_rate": 0.5,
                    "evidence_hit_rate": 1.0,
                },
                "attribution": {
                    "single_component_edit_ratio": 0.6,
                    "ambiguous_ratio": 0.4,
                    "proposal_plan_compliance_rate": 0.8,
                    "valid_plan_rate": 0.9,
                },
                "stability": {
                    "valid_candidate_rate": 0.8,
                    "regression_rate": 0.15,
                    "bug_rate": 0.05,
                    "score_variance": 5.2,
                },
            }
            print("\n--- Synthetic record format test ---")
            print(json.dumps(synthetic, indent=2, ensure_ascii=False))
            print("\n--- Comparison report (single session) ---")
            # Temporarily write and read back
            collector.save(synthetic)
            print(collector.generate_comparison_report(["synthetic_test"]))
            # Clean up synthetic record
            lines = collector.metrics_path.read_text(encoding="utf-8").splitlines()
            remaining = [l for l in lines if "synthetic_test" not in l]
            collector.metrics_path.write_text("\n".join(remaining) + ("\n" if remaining else ""), encoding="utf-8")
