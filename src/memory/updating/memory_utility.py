
from __future__ import annotations
import os
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

COMPONENT_MEMORY_DIR = Path(os.environ.get("COMPONENT_MEMORY_DIR", str(Path(__file__).parent.parent.parent.parent / "workspace" / "component_memory")))

USABLE_STATUSES = {"usable", "reactivated"}


def strategy_memory_id(episode_id: str | None) -> str:
    return f"strategy:{episode_id or 'unknown'}"


def anchor_memory_id(episode_id: str | None) -> str:
    return f"anchor:{episode_id or 'unknown'}"


def component_memory_id(component: str | None, family_id: str | None) -> str:
    return f"component:{component or 'unknown'}:{family_id or 'unknown'}"


def avoid_memory_id(component: str | None, family_id: str | None) -> str:
    return f"avoid:{component or 'unknown'}:{family_id or 'unknown'}"


class MemoryUtilityTracker:

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or COMPONENT_MEMORY_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.utility_path = self.output_dir / "memory_utility.json"



    def _blank(self) -> dict[str, Any]:
        return {
            "last_updated": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "memory_items": {},
        }

    def _load(self) -> dict[str, Any]:
        if not self.utility_path.exists():
            return self._blank()
        try:
            return json.loads(self.utility_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._blank()

    def _save(self, data: dict[str, Any]) -> None:
        data["last_updated"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self.utility_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def load_states(self) -> dict[str, Any]:
        return self._load()

    def save_states(self, states: dict[str, Any]) -> None:
        self._save(states)

    def _default_item(
        self,
        memory_id: str,
        memory_type: str,
        historical_status: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        item = {
            "memory_id": memory_id,
            "memory_type": memory_type,
            "historical_status": historical_status,
            "current_status": "usable",
            "utility_score": 0.0,
            "use_count": 0,
            "success_after_use": 0,
            "failure_after_use": 0,
            "recent_failures": 0,
            "last_used_iter": None,
            "last_success_iter": None,
            "usable": True,

            "reuse_count": 0,
            "reuse_success": 0,
            "reuse_failure": 0,
            "metadata": metadata or {},
        }
        if memory_type == "avoid":
            item.update({
                "current_status": "active_avoid",
                "avoid_strength": 1.0,
                "retest_count": 0,
                "contradicted_count": 0,
                "strengthened_count": 0,
                "last_retest_iter": None,
                "usable": False,
            })
            item.update(self._avoid_metadata_fields(metadata or {}))
        return item

    def _avoid_metadata_fields(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "component": metadata.get("component"),
            "family_id": metadata.get("family_id"),
            "description": metadata.get("description"),
        }

    def _normalize_avoid_item(self, item: dict[str, Any]) -> dict[str, Any]:
        metadata = item.get("metadata", {}) or {}
        item.setdefault("historical_status", "regression")
        item.setdefault("current_status", "active_avoid")
        item.setdefault("avoid_strength", 1.0)
        item.setdefault("retest_count", 0)
        item.setdefault("contradicted_count", 0)
        item.setdefault("strengthened_count", 0)
        item.setdefault("last_retest_iter", None)
        item.setdefault("component", metadata.get("component"))
        item.setdefault("family_id", metadata.get("family_id"))
        item.setdefault("description", metadata.get("description"))
        item["usable"] = False
        return item



    def register_memory(
        self,
        memory_id: str,
        memory_type: str,
        historical_status: str = "effective",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = self._load()
        items = data.setdefault("memory_items", {})
        item = items.get(memory_id)
        if item is None:
            item = self._default_item(
                memory_id, memory_type, historical_status, metadata
            )
            items[memory_id] = item
        else:
            item.setdefault("metadata", {}).update(metadata or {})
            item["memory_type"] = item.get("memory_type") or memory_type
            item["historical_status"] = item.get("historical_status") or historical_status
            self._normalize_item(item)
        if memory_type == "avoid" or item.get("memory_type") == "avoid":
            item.update(self._avoid_metadata_fields(item.get("metadata", {}) or {}))
            self._normalize_avoid_item(item)
        self._save(data)
        return dict(item)

    def register_strategy(self, strategy: dict[str, Any]) -> dict[str, Any]:
        episode_id = strategy.get("episode_id") or strategy.get("candidate_id")
        return self.register_memory(
            strategy_memory_id(episode_id),
            "strategy",
            "effective",
            {
                "episode_id": episode_id,
                "score": strategy.get("score"),
                "score_delta": strategy.get("score_delta"),
                "evidence_scope": strategy.get("evidence_scope"),
                "components_changed": strategy.get("components_changed", []),
            },
        )

    def register_anchor(self, anchor: dict[str, Any]) -> dict[str, Any]:
        episode_id = anchor.get("episode_id") or anchor.get("candidate_id")
        return self.register_memory(
            anchor_memory_id(episode_id),
            "anchor",
            "effective",
            {
                "episode_id": episode_id,
                "score": anchor.get("score"),
                "score_delta": anchor.get("score_delta"),
                "components_changed": anchor.get("components_changed", []),
            },
        )

    def register_from_evidence(self, evidence: dict[str, Any]) -> None:
        for strategy in evidence.get("strategy_evidence", []):
            self.register_strategy(strategy)
        for anchor in evidence.get("positive_anchors", []):
            self.register_anchor(anchor)
        for component, comp_data in evidence.get("components", {}).items():
            for family in comp_data.get("change_families", []):
                self.register_memory(
                    component_memory_id(component, family.get("family_id")),
                    "component",
                    "effective" if family.get("avg_score_delta", 0) > 0 else "observed",
                    {
                        "component": component,
                        "family_id": family.get("family_id"),
                        "description": family.get("description"),
                        "avg_score_delta": family.get("avg_score_delta"),
                    },
                )
            for family in comp_data.get("regression_families", []):
                self.register_memory(
                    avoid_memory_id(component, family.get("family_id")),
                    "avoid",
                    "regression",
                    {
                        "component": component,
                        "family_id": family.get("family_id"),
                        "description": family.get("description"),
                        "avg_score_delta": family.get("avg_score_delta"),
                    },
                )



    def _normalize_item(self, item: dict[str, Any]) -> dict[str, Any]:
        item.setdefault("current_status", "usable")
        item.setdefault("utility_score", 0.0)
        item.setdefault("use_count", 0)
        item.setdefault("success_after_use", 0)
        item.setdefault("failure_after_use", 0)
        item.setdefault("recent_failures", 0)
        item.setdefault("last_used_iter", None)
        item.setdefault("last_success_iter", None)
        item.setdefault("reuse_count", item.get("use_count", 0))
        item.setdefault("reuse_success", item.get("success_after_use", 0))
        item.setdefault("reuse_failure", item.get("failure_after_use", 0))
        item.setdefault("usable", item.get("current_status") in USABLE_STATUSES)
        return item

    def update_after_episode(self, episode: dict[str, Any]) -> None:
        memory_ids = episode.get("assigned_memory_ids") or []
        if not memory_ids:
            return

        data = self._load()
        items = data.setdefault("memory_items", {})
        iteration = episode.get("iteration")
        score_delta = float(episode.get("score_delta", 0) or 0)
        success = score_delta > 0

        for memory_id in memory_ids:
            item = items.get(memory_id)
            if item is None:
                item = self._default_item(
                    memory_id,
                    memory_id.split(":", 1)[0],
                    "observed",
                    {},
                )
                items[memory_id] = item
            self._normalize_item(item)
            if item.get("memory_type") == "avoid":
                self._normalize_avoid_item(item)
                continue
            previous_status = item.get("current_status", "usable")
            item["use_count"] += 1
            item["last_used_iter"] = iteration
            if item.get("memory_type") == "strategy":
                item["reuse_count"] = item["use_count"]

            if success:
                item["utility_score"] = round(float(item["utility_score"]) + 1.0, 3)
                item["success_after_use"] += 1
                item["last_success_iter"] = iteration
                item["recent_failures"] = 0
                if item.get("memory_type") == "strategy":
                    item["reuse_success"] = item["success_after_use"]
                item["current_status"] = (
                    "reactivated"
                    if previous_status in {"stale", "cooling_down"}
                    else "usable"
                )
                item["usable"] = True
            else:
                item["utility_score"] = round(float(item["utility_score"]) - 1.0, 3)
                item["failure_after_use"] += 1
                item["recent_failures"] += 1
                if item.get("memory_type") == "strategy":
                    item["reuse_failure"] = item["failure_after_use"]
                if item["recent_failures"] >= 2 or item["utility_score"] <= -2:
                    item["current_status"] = "stale"
                    item["usable"] = False
                else:
                    item["current_status"] = "cooling_down"
                    item["usable"] = True

        self._save(data)

    def update_avoid_after_retest(
        self,
        memory_id: str,
        score_delta: float,
        iteration: int,
    ) -> None:
        states = self.load_states()
        items = states.setdefault("memory_items", {})
        item = items.get(memory_id)
        if not item:
            return

        self._normalize_avoid_item(item)
        item["retest_count"] = item.get("retest_count", 0) + 1
        item["last_retest_iter"] = iteration

        if float(score_delta or 0) > 0:
            item["contradicted_count"] = item.get("contradicted_count", 0) + 1
            item["avoid_strength"] = round(
                max(0.0, float(item.get("avoid_strength", 1.0)) - 0.3),
                3,
            )
            item["current_status"] = (
                "reactivated_hypothesis"
                if item["avoid_strength"] <= 0.3
                else "contradicted"
            )
        else:
            item["strengthened_count"] = item.get("strengthened_count", 0) + 1
            item["avoid_strength"] = round(
                min(1.0, float(item.get("avoid_strength", 1.0)) + 0.2),
                3,
            )
            item["current_status"] = "active_avoid"

        items[memory_id] = item
        self.save_states(states)

        log_entry = {
            "operation": "UpdateAvoidAfterRetest",
            "memory_id": memory_id,
            "score_delta": score_delta,
            "new_avoid_strength": item["avoid_strength"],
            "new_status": item["current_status"],
            "iteration": iteration,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with (self.output_dir / "evolution_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")



    def get_item(self, memory_id: str) -> dict[str, Any] | None:
        item = self._load().get("memory_items", {}).get(memory_id)
        return dict(item) if item else None

    def is_usable(self, memory_id: str) -> bool:
        item = self.get_item(memory_id)
        if not item:
            return True
        return bool(item.get("usable", item.get("current_status") in USABLE_STATUSES))

    def _utility_for(self, memory_id: str, memory_type: str) -> dict[str, Any]:
        item = self.get_item(memory_id)
        if item is None:
            item = self.register_memory(memory_id, memory_type, "observed")
        self._normalize_item(item)
        return item

    def enrich_strategy(self, strategy: dict[str, Any]) -> dict[str, Any]:
        episode_id = strategy.get("episode_id") or strategy.get("candidate_id")
        memory_id = strategy_memory_id(episode_id)
        utility = self._utility_for(memory_id, "strategy")
        enriched = dict(strategy)
        enriched.update(self._public_utility_fields(utility))
        enriched["memory_id"] = memory_id
        return enriched

    def enrich_anchor(self, anchor: dict[str, Any]) -> dict[str, Any]:
        episode_id = anchor.get("episode_id") or anchor.get("candidate_id")
        memory_id = anchor_memory_id(episode_id)
        utility = self._utility_for(memory_id, "anchor")
        strategy_utility = self.get_item(strategy_memory_id(episode_id))
        if strategy_utility and not bool(strategy_utility.get("usable", True)):
            utility = dict(utility)
            utility["current_status"] = strategy_utility.get("current_status", "stale")
            utility["utility_score"] = strategy_utility.get("utility_score", utility.get("utility_score", 0))
            utility["recent_failures"] = strategy_utility.get("recent_failures", utility.get("recent_failures", 0))
            utility["usable"] = False
        enriched = dict(anchor)
        enriched.update(self._public_utility_fields(utility))
        enriched["memory_id"] = memory_id
        return enriched

    def _public_utility_fields(self, item: dict[str, Any]) -> dict[str, Any]:
        keys = [
            "current_status",
            "utility_score",
            "use_count",
            "success_after_use",
            "failure_after_use",
            "recent_failures",
            "last_used_iter",
            "last_success_iter",
            "usable",
            "reuse_count",
            "reuse_success",
            "reuse_failure",
        ]
        return {key: item.get(key) for key in keys}

    def enrich_guidance(self, guidance: dict[str, Any]) -> dict[str, Any]:
        result = dict(guidance)

        strategy_source = (
            guidance.get("historical_effective_strategies")
            or guidance.get("effective_strategies", [])
        )
        historical = [self.enrich_strategy(s) for s in strategy_source]
        usable = [s for s in historical if bool(s.get("usable", True))]
        stale = [s for s in historical if not bool(s.get("usable", True))]

        result["historical_effective_strategies"] = historical
        result["effective_strategies"] = usable

        target_source = guidance.get("strategy_reuse_targets", []) or historical
        enriched_targets = [self.enrich_strategy(s) for s in target_source]
        result["strategy_reuse_targets"] = [
            s for s in enriched_targets if bool(s.get("usable", True))
        ]

        anchor_source = (
            guidance.get("historical_positive_anchors")
            or guidance.get("positive_anchors", [])
        )
        anchors = [self.enrich_anchor(a) for a in anchor_source]
        result["historical_positive_anchors"] = anchors
        result["positive_anchors"] = [a for a in anchors if bool(a.get("usable", True))]



        if not guidance.get("hypothesis_test_targets") and stale:
            result["hypothesis_test_targets"] = [
                {
                    "episode_id": s.get("episode_id"),
                    "score": s.get("score"),
                    "score_delta": s.get("score_delta"),
                    "evidence_scope": s.get("evidence_scope"),
                    "components_changed": s.get("components_changed", []),
                    "current_status": s.get("current_status"),
                    "memory_id": s.get("memory_id"),
                    "instruction": (
                        "This historical effective strategy is no longer directly "
                        "usable; test one narrower hypothesis instead of reusing it."
                    ),
                }
                for s in stale[:3]
            ]

        maturity = dict(result.get("evidence_maturity", {}) or {})
        maturity["has_effective_strategy"] = bool(usable)
        maturity["has_usable_strategy"] = bool(usable)
        maturity["historical_effective_count"] = len(historical)
        maturity["stale_effective_count"] = len(stale)
        if not usable and historical:
            maturity["maturity"] = "stale_strategy_only"
            maturity["recommended_reading_strategy"] = "full_history"
            maturity["recommended_governance_mode"] = "audit_only_exploration"
        result["evidence_maturity"] = maturity

        result["memory_utility_summary"] = self.summary()
        return result

    def summary(self) -> dict[str, Any]:
        items = list(self._load().get("memory_items", {}).values())
        strategy_items = [i for i in items if i.get("memory_type") == "strategy"]
        usable_strategy = [i for i in strategy_items if bool(i.get("usable", True))]
        stale_strategy = [i for i in strategy_items if i.get("current_status") == "stale"]
        cooling = [i for i in items if i.get("current_status") == "cooling_down"]
        reactivated = [i for i in items if i.get("current_status") == "reactivated"]
        avg = (
            round(sum(float(i.get("utility_score", 0) or 0) for i in items) / len(items), 3)
            if items else 0.0
        )
        return {
            "total_memory_items": len(items),
            "effective_strategy_count": len(strategy_items),
            "usable_strategy_count": len(usable_strategy),
            "historical_effective_count": len(strategy_items),
            "stale_strategy_count": len(stale_strategy),
            "cooling_down_count": len(cooling),
            "reactivated_memory_count": len(reactivated),
            "utility_pruned_strategy_count": len(stale_strategy),
            "avg_utility_score": avg,
        }


if __name__ == "__main__":
    tracker = MemoryUtilityTracker()
    print(json.dumps(tracker.summary(), indent=2, ensure_ascii=False))
