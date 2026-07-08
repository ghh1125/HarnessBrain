import os
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory.encoding.ast_structure import component_ast_signature

if __name__.startswith("src.memory."):
    sys.modules[__name__.replace("src.", "", 1)] = sys.modules[__name__]
elif __name__.startswith("memory."):
    sys.modules[f"src.{__name__}"] = sys.modules[__name__]

COMPONENT_MEMORY_DIR = Path(os.environ.get("COMPONENT_MEMORY_DIR", str(Path(__file__).parent.parent.parent.parent / "workspace" / "component_memory")))
CATEGORY_ATTRIBUTION_MIN_IMPROVEMENTS = 3
NEW_PASS_ATTRIBUTION_MIN_TASKS = 2


TRAJECTORY_BEHAVIOR_MIN_VERIFY_TASKS = 2

_TC_COMPONENT_KEYWORDS = {
    "retrieval": [
        "overlap", "similarity", "retrieve", "jaccard",
        "stopword", "tfidf", "tf-idf", "bm25", "embedding",
        "top_k", "search", "fetch", "cosine",
    ],
    "prompt": [
        "prompt", "template", "instruction", "format",
        "prior", "label_list", "system_message",
        "few_shot", "context", "message",
    ],
    "parser": [
        "parse", "extract", "json", "split", "output",
        "prediction", "label_map", "normalize", "strip",
    ],
    "memory_update": [
        "observe", "learn", "update", "store", "append",
        "deque", "buffer", "fifo", "cache", "memory",
    ],
    "state_management": [
        "get_state", "set_state", "checkpoint",
        "restore", "serialize", "save",
    ],
}

_TB_COMPONENT_KEYWORDS = {
    "llm_call": [
        "llm", "completion", "message", "prompt",
        "anthropic", "openai", "claude", "model",
        "temperature", "max_tokens",
    ],
    "tool_parsing": [
        "tool", "function_call", "parse", "extract",
        "result", "output", "response", "json",
    ],
    "command_execution": [
        "subprocess", "shell", "bash", "command",
        "execute", "run", "terminal", "tmux",
        "popen", "check_output",
    ],
    "agent_loop": [
        "loop", "step", "iteration", "while",
        "retry", "attempt", "max_steps", "done",
    ],
    "context_management": [
        "context", "history", "memory", "state",
        "conversation", "messages", "buffer",
        "truncate", "summarize",
    ],
    "error_handling": [
        "error", "exception", "retry", "fallback",
        "timeout", "catch", "raise", "handle",
    ],
    "state_management": [
        "checkpoint", "save", "restore", "persist",
        "serialize", "snapshot", "load",
    ],
    "prompt_template": [
        "template", "format", "instruction",
        "system_prompt", "few_shot", "example",
    ],
}


COMPONENT_KEYWORDS = _TC_COMPONENT_KEYWORDS


def configure(task: str) -> None:
    global COMPONENT_KEYWORDS
    COMPONENT_KEYWORDS = _TB_COMPONENT_KEYWORDS if task == "terminal" else _TC_COMPONENT_KEYWORDS





def identify_components(diff: str) -> dict:
    hit_counts = {k: 0 for k in COMPONENT_KEYWORDS}
    hit_keywords: dict[str, list[str]] = {k: [] for k in COMPONENT_KEYWORDS}

    if not diff:
        return {
            "components_changed": [],
            "attribution": "unknown",
            "hit_counts": hit_counts,
            "hit_keywords": hit_keywords,
        }

    added_text = " ".join(
        line[1:].lower()
        for line in diff.split("\n")
        if line.startswith("+") and not line.startswith("+++")
    )
    removed_text = " ".join(
        line[1:].lower()
        for line in diff.split("\n")
        if line.startswith("-") and not line.startswith("---")
    )
    has_context = bool(removed_text.strip())

    for comp, keywords in COMPONENT_KEYWORDS.items():
        net_hits = [kw for kw in keywords if kw in added_text and kw not in removed_text]
        hit_counts[comp] = len(net_hits)
        hit_keywords[comp] = net_hits


    min_hits = 2 if not has_context else 1
    changed = [c for c, n in hit_counts.items() if n >= min_hits]


    if len(changed) == 0:
        attribution = "unknown"
    elif len(changed) == 1:
        attribution = "clear"
    else:

        sorted_changed = sorted(changed, key=lambda c: hit_counts[c], reverse=True)
        top_hits = hit_counts[sorted_changed[0]]
        second_hits = hit_counts[sorted_changed[1]] if len(sorted_changed) > 1 else 0
        if top_hits >= 2 and top_hits >= 2 * max(second_hits, 1):
            attribution = "clear"
            changed = [sorted_changed[0]]
        else:
            attribution = "ambiguous"

    return {
        "components_changed": changed,
        "attribution": attribution,
        "hit_counts": hit_counts,
        "hit_keywords": hit_keywords,
    }


def compute_evidence_strength(
    score_delta: float,
    attribution: str,
    val_size: int,
    config: dict,
) -> str:
    if attribution != "clear":
        return "weak"

    effective_delta = score_delta
    if config.get("small_validation_penalty", True) and val_size < 100:
        effective_delta = score_delta / 2

    min_effective = config.get("min_score_delta_for_effective", 2.0)
    regression_threshold = config.get("regression_threshold", -3.0)

    if effective_delta >= min_effective:
        return "medium"
    elif effective_delta > 0:
        return "weak"
    elif effective_delta < regression_threshold:
        return "negative"
    else:
        return "inconclusive"


def compute_verdict(
    evidence_strength: str,
    score_delta: float,
    attribution: str,
    regression_threshold: float = -3.0,
) -> str:
    if score_delta < regression_threshold:
        return "clear_regression" if attribution == "clear" else "ambiguous_regression"
    if evidence_strength in {"weak_clear", "medium", "strong"} and score_delta > 0:
        return "effective"
    if attribution == "ambiguous":
        return "ambiguous"
    if attribution == "unknown":
        return "unknown"
    return "inconclusive"





class ComponentEvidenceBuilder:
    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or COMPONENT_MEMORY_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_path = self.output_dir / "component_evidence.json"
        self.audit_csv_path = self.output_dir / "component_attribution_audit.csv"

        self._config: dict = {}
        self._val_size: int = 50

        try:
            from memory.config_loader import get_memory_config
            mcfg = get_memory_config()
            self._config = mcfg.get("evidence_settings", {})
        except Exception:
            pass

        try:
            import yaml
            cfg_path = Path(__file__).parent.parent.parent / "config.yaml"
            with open(cfg_path) as f:
                main_cfg = yaml.safe_load(f)
            self._val_size = main_cfg.get("dataset", {}).get("num_val", 50)
        except Exception:
            pass



    def _load(self) -> dict:
        if self.evidence_path.exists():
            return json.loads(self.evidence_path.read_text(encoding="utf-8"))
        return self._blank()

    def _blank(self) -> dict:
        return {
            "last_updated": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "total_episodes": 0,
            "evidence_summary": {
                "actionable_count": 0,
                "diagnostic_count": 0,
                "hypothesis_count": 0,
            },
            "clear_evidence_count": 0,
            "ambiguous_evidence_count": 0,
            "clear_positive_count": 0,
            "clear_regression_count": 0,
            "ambiguous_positive_count": 0,
            "ambiguous_regression_count": 0,
            "strategy_evidence_count": 0,
            "component_strategy_count": 0,
            "interaction_strategy_count": 0,
            "candidate_strategy_count": 0,
            "strategy_evidence": [],
            "positive_anchors": [],
            "ambiguous_evidence": [],
            "diagnostic_episodes": [],
            "failure_category_attribution": [],
            "new_pass_component_attribution": [],
            "task_failure_category_stats": {},
            "task_level_evidence": [],
            "attribution_audit": [],
            "evidence_maturity": self._blank_maturity(),
            "components": {k: self._blank_component() for k in COMPONENT_KEYWORDS},
        }

    def _blank_maturity(self) -> dict:
        return {
            "maturity": "raw_only",
            "has_positive_anchor": False,
            "has_effective_strategy": False,
            "has_promising_direction": False,
            "has_confirmed_direction": False,
            "positive_anchor_count": 0,
            "effective_strategy_count": 0,
            "promising_direction_count": 0,
            "confirmed_direction_count": 0,
            "avoid_count": 0,
            "avoid_only_guidance": False,
            "recommended_reading_strategy": "full_history",
            "recommended_governance_mode": "audit_only_exploration",
        }

    def _blank_component(self) -> dict:
        return {
            "evidence": [],
            "ambiguous_evidence": [],
            "task_level_evidence": [],
            "change_families": [],
            "regression_families": [],
            "unexplored_directions": [],
            "exploration_status": "unexplored",
            "component_score": 0.5,
            "total_edits": 0,
            "observed_edits": 0,
            "effective_edits": 0,
            "effective_delta_sum": 0.0,
            "clear_positive_count": 0,
            "regression_edits": 0,
            "clear_regression_edits": 0,
            "ambiguous_regression_edits": 0,
        }

    def _save(self, data: dict) -> None:
        data["last_updated"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        data.setdefault("clear_positive_count", 0)
        data.setdefault("clear_regression_count", 0)
        data.setdefault("ambiguous_positive_count", 0)
        data.setdefault("ambiguous_regression_count", 0)
        data.setdefault("strategy_evidence_count", 0)
        data.setdefault("component_strategy_count", 0)
        data.setdefault("interaction_strategy_count", 0)
        data.setdefault("candidate_strategy_count", 0)
        data.setdefault("strategy_evidence", [])
        data.setdefault("positive_anchors", [])
        data.setdefault("ambiguous_evidence", [])
        data.setdefault("diagnostic_episodes", [])
        data.setdefault("task_level_evidence", [])
        data.setdefault("task_failure_category_stats", {})
        data.setdefault("failure_category_attribution", [])
        data.setdefault("new_pass_component_attribution", [])
        data["strategy_evidence_count"] = len(data.get("strategy_evidence", []))
        data["component_strategy_count"] = sum(
            1 for item in data.get("strategy_evidence", [])
            if item.get("evidence_scope") == "component"
        )
        data["interaction_strategy_count"] = sum(
            1 for item in data.get("strategy_evidence", [])
            if item.get("evidence_scope") == "interaction"
        )
        data["candidate_strategy_count"] = sum(
            1 for item in data.get("strategy_evidence", [])
            if item.get("evidence_scope") == "candidate"
        )

        data.setdefault("evidence_summary", {})
        data["evidence_summary"]["actionable_count"] = data.get("clear_evidence_count", 0)
        data["evidence_summary"]["diagnostic_count"] = len(data.get("diagnostic_episodes", []))
        data["evidence_summary"]["hypothesis_count"] = data["evidence_summary"].get("hypothesis_count", 0)
        data["evidence_maturity"] = self._compute_evidence_maturity(data)

        clean = json.loads(json.dumps(data))
        for comp_data in clean.get("components", {}).values():
            for fam in comp_data.get("change_families", []) + comp_data.get("regression_families", []):
                for key in list(fam.keys()):
                    if key.startswith("_"):
                        del fam[key]
        self.evidence_path.write_text(
            json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self._write_audit_csv(clean)

    def _compute_evidence_maturity(self, data: dict) -> dict:
        positive_anchor_count = len(data.get("positive_anchors", []))
        effective_strategy_count = len(data.get("strategy_evidence", []))
        confirmed_direction_count = 0
        promising_direction_count = 0
        avoid_count = 0

        for comp_data in data.get("components", {}).values():
            for fam in comp_data.get("change_families", []):
                if fam.get("attribution_type", "clear") != "clear":
                    continue
                if fam.get("status") == "confirmed_guidance":
                    confirmed_direction_count += 1
                elif (
                    fam.get("status") == "promising"
                    or fam.get("verdict") == "effective"
                    or fam.get("avg_score_delta", 0) > 0
                ):
                    promising_direction_count += 1
            avoid_count += len(comp_data.get("regression_families", []))

        has_confirmed = confirmed_direction_count > 0
        has_promising = promising_direction_count > 0
        has_anchor = positive_anchor_count > 0
        has_strategy = effective_strategy_count > 0

        if has_confirmed:
            maturity = "confirmed_direction"
            reading = "guidance_only"
            governance = "strong_component_boundary"
        elif has_promising:
            maturity = "promising_direction"
            reading = "component_playbook"
            governance = (
                "anchor_refinement_soft_boundary" if has_anchor else "audit_only_exploration"
            )
        elif has_strategy or has_anchor:
            maturity = "strategy_only"
            reading = "strategy_context"
            governance = "strategy_reuse_soft_boundary"
        elif data.get("diagnostic_episodes"):
            maturity = "diagnostic_only"
            reading = "full_history"
            governance = "audit_only_exploration"
        else:
            maturity = "raw_only"
            reading = "full_history"
            governance = "audit_only_exploration"

        return {
            "maturity": maturity,
            "has_positive_anchor": has_anchor,
            "has_effective_strategy": has_strategy,
            "has_promising_direction": has_promising,
            "has_confirmed_direction": has_confirmed,
            "positive_anchor_count": positive_anchor_count,
            "effective_strategy_count": effective_strategy_count,
            "promising_direction_count": promising_direction_count,
            "confirmed_direction_count": confirmed_direction_count,
            "avoid_count": avoid_count,
            "avoid_only_guidance": avoid_count > 0 and not (has_anchor or has_promising or has_confirmed),
            "recommended_reading_strategy": reading,
            "recommended_governance_mode": governance,
        }

    def _write_audit_csv(self, data: dict) -> None:
        rows = data.get("attribution_audit", [])
        fieldnames = [
            "episode_id",
            "score_delta",
            "components_changed",
            "attribution_type",
            "modified_component_count",
            "hit_counts",
            "verdict",
        ]
        with self.audit_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    "episode_id": row.get("episode_id", ""),
                    "score_delta": row.get("score_delta", 0),
                    "components_changed": json.dumps(
                        row.get("components_changed", []), ensure_ascii=False
                    ),
                    "attribution_type": row.get("attribution_type", "unknown"),
                    "modified_component_count": row.get("modified_component_count", 0),
                    "hit_counts": json.dumps(row.get("hit_counts", {}), ensure_ascii=False),
                    "verdict": row.get("verdict", ""),
                })



    def _add_to_change_families(
        self,
        comp_data: dict,
        comp_name: str,
        episode_id: str,
        keywords: list,
        verdict: str,
        score_delta: float,
        evidence_strength: str,
        iteration: int,
        extra_fields: dict | None = None,
    ) -> None:
        families = comp_data["change_families"]
        iter_label = f"iter_{iteration}"

        best_match = None
        best_overlap = 0
        for fam in families:
            fam_kw = set(fam.get("_keywords", []))
            overlap = len(fam_kw & set(keywords))
            if overlap >= 2 and fam.get("verdict") == verdict and overlap > best_overlap:
                best_match = fam
                best_overlap = overlap

        if best_match is not None:
            fam = best_match
            if score_delta > 0:
                if episode_id not in fam.get("supporting_episodes", []):
                    fam.setdefault("supporting_episodes", []).append(episode_id)
            else:
                if episode_id not in fam.get("contradicting_episodes", []):
                    fam.setdefault("contradicting_episodes", []).append(episode_id)
            fam.setdefault("_score_deltas", []).append(score_delta)
            fam["avg_score_delta"] = round(
                sum(fam["_score_deltas"]) / len(fam["_score_deltas"]), 1
            )
            fam["last_seen"] = iter_label
            fam["_keywords"] = list(set(fam.get("_keywords", [])) | set(keywords))
            if extra_fields:
                fam.update(extra_fields)

            acc_delta = fam["avg_score_delta"]
            min_eff = self._config.get("min_score_delta_for_effective", 2.0)
            min_eps = self._config.get("min_episodes_for_promote", 3)
            n_supporting = len(fam.get("supporting_episodes", []))
            if acc_delta >= min_eff and n_supporting >= min_eps:
                fam["evidence_strength"] = "strong"
            elif acc_delta >= min_eff:
                fam["evidence_strength"] = "medium"
            elif acc_delta > 0:
                fam["evidence_strength"] = "weak"
            else:
                fam["evidence_strength"] = "inconclusive"
        else:
            n = sum(1 for f in families if f.get("verdict") == verdict) + 1
            description = ", ".join(keywords[:3]) if keywords else "unspecified change"
            fam = {
                "family_id": f"{comp_name}_{n:03d}",
                "description": description,
                "supporting_episodes": [episode_id] if score_delta >= 0 else [],
                "contradicting_episodes": [episode_id] if score_delta < 0 else [],
                "avg_score_delta": round(score_delta, 1),
                "evidence_strength": evidence_strength,
                "verdict": verdict,
                "attribution_type": "clear",
                "used_for_component_score": True,
                "weight": 1.0,
                "status": "promising" if verdict == "effective" else "inconclusive",
                "first_seen": iter_label,
                "last_seen": iter_label,
                "_keywords": keywords,
                "_score_deltas": [score_delta],
            }
            if extra_fields:
                fam.update(extra_fields)
            families.append(fam)

    def _add_to_regression_families(
        self,
        comp_data: dict,
        comp_name: str,
        episode_id: str,
        keywords: list,
        score_delta: float,
        iteration: int,
    ) -> None:
        families = comp_data["regression_families"]
        iter_label = f"iter_{iteration}"

        best_match = None
        best_overlap = 0
        for fam in families:
            fam_kw = set(fam.get("_keywords", []))
            overlap = len(fam_kw & set(keywords))
            if overlap >= 2 and overlap > best_overlap:
                best_match = fam
                best_overlap = overlap

        if best_match is not None:
            fam = best_match
            if episode_id not in fam.get("episodes", []):
                fam.setdefault("episodes", []).append(episode_id)
            fam.setdefault("_score_deltas", []).append(score_delta)
            fam["avg_score_delta"] = round(
                sum(fam["_score_deltas"]) / len(fam["_score_deltas"]), 1
            )
            fam["last_seen"] = iter_label
            fam["_keywords"] = list(set(fam.get("_keywords", [])) | set(keywords))
        else:
            n = len(families) + 1
            description = ", ".join(keywords[:3]) if keywords else "unspecified change"
            families.append({
                "family_id": f"{comp_name}_reg_{n:03d}",
                "description": description,
                "episodes": [episode_id],
                "avg_score_delta": round(score_delta, 1),
                "status": "regression_family",
                "verdict": "clear_regression",
                "attribution_type": "clear",
                "evidence_strength": "negative",
                "used_for_component_score": True,
                "first_seen": iter_label,
                "last_seen": iter_label,
                "_keywords": keywords,
                "_score_deltas": [score_delta],
            })

    def _evidence_record(
        self,
        episode: dict,
        comp_info: dict,
        evidence_strength: str,
        verdict: str,
        used_for_component_score: bool,
    ) -> dict:
        components_changed = comp_info.get("components_changed", [])
        return {
            "episode_id": episode.get("episode_id", ""),
            "iteration": episode.get("iteration", 0),
            "score_delta": float(episode.get("score_delta", 0)),
            "components_changed": components_changed,
            "attribution_type": comp_info.get("attribution", "unknown"),
            "modified_component_count": len(components_changed),
            "hit_keywords": comp_info.get("hit_keywords", {}),
            "hit_counts": comp_info.get("hit_counts", {}),
            "evidence_strength": evidence_strength,
            "verdict": verdict,
            "used_for_component_score": used_for_component_score,
            "task_outcome_summary": episode.get("task_outcome_summary", {}),
            "task_new_passes": episode.get("task_new_passes", []),
            "task_regressions": episode.get("task_regressions", []),
            "task_persistent_failures": episode.get("task_persistent_failures", []),
            "new_pass_attribution": episode.get("new_pass_attribution", {}),
            "task_failure_category_deltas": episode.get("task_failure_category_deltas", {}),
        }



    def _evidence_gating_enabled(self) -> bool:
        from memory.config_loader import evidence_quality_enabled
        return evidence_quality_enabled()

    def classify_evidence(self, episode: dict, attribution: str) -> str:
        if attribution == "clear":
            return "actionable"
        return "diagnostic"

    def _evidence_scope(self, attribution_result: dict) -> str:
        components = attribution_result.get("components_changed", [])
        attribution = attribution_result.get("attribution", "unknown")
        if attribution == "clear" and len(components) == 1:
            return "component"
        if len(components) > 1:
            return "interaction"
        return "candidate"

    def _record_strategy_evidence(
        self, episode: dict, attribution_result: dict, data: dict
    ) -> None:
        score_delta = float(episode.get("score_delta", 0) or 0)
        if score_delta <= 0:
            return

        episode_id = episode.get("episode_id", "")
        strategies = data.setdefault("strategy_evidence", [])
        if any(item.get("episode_id") == episode_id for item in strategies):
            return

        scope = self._evidence_scope(attribution_result)
        components = attribution_result.get("components_changed", [])
        allowed_uses = {
            "as_strategy": True,
            "as_anchor": True,
            "as_lesson_source": True,
            "as_next_strategy": True,
            "as_component_proof": scope == "component",
            "as_interaction_signal": scope == "interaction",
            "as_hypothesis_source": scope in {"interaction", "candidate"},
            "as_avoid_signal": False,
        }
        entry = {
            "episode_id": episode_id,
            "candidate_id": episode_id,
            "iteration": episode.get("iteration", 0),
            "parent_candidate": episode.get("parent_candidate"),
            "outcome": "positive",
            "score": episode.get("score"),
            "score_delta": score_delta,
            "strategy_evidence": True,
            "evidence_scope": scope,
            "components_changed": components,
            "attribution_type": attribution_result.get("attribution", "unknown"),
            "confidence": f"{scope}_level",
            "allowed_uses": allowed_uses,
            "can_promote_component": scope == "component",
            "used_for_component_score": scope == "component",
            "current_status": "usable",
            "utility_score": 0.0,
            "use_count": 0,
            "success_after_use": 0,
            "failure_after_use": 0,
            "recent_failures": 0,
            "last_used_iter": None,
            "last_success_iter": None,
            "usable": True,
            "status": "usable_effective",
            "reuse_count": 0,
            "reuse_success": 0,
            "reuse_failure": 0,
            "note": (
                "Effective candidate-level strategy. Reuse according to evidence_scope; "
                "only component-scoped evidence can promote a component."
            ),
        }
        strategies.append(entry)
        data["strategy_evidence_count"] = len(strategies)
        if scope == "component":
            data["component_strategy_count"] = data.get("component_strategy_count", 0) + 1
        elif scope == "interaction":
            data["interaction_strategy_count"] = data.get("interaction_strategy_count", 0) + 1
        else:
            data["candidate_strategy_count"] = data.get("candidate_strategy_count", 0) + 1

    def _record_diagnostic_evidence(
        self, episode: dict, attribution_result: dict, data: dict
    ) -> None:
        score_delta = float(episode.get("score_delta", 0) or 0)
        regression_threshold = self._config.get("regression_threshold", -3.0)
        if score_delta > 0:
            attribution_status = "ambiguous_positive"
            verdict = "ambiguous_positive"
            evidence_level = "anchor"
            evidence_role = "candidate_anchor_not_component_proof"
            can_seed_anchor_refinement = True
            data["ambiguous_positive_count"] = data.get("ambiguous_positive_count", 0) + 1
        elif score_delta < regression_threshold:
            attribution_status = "ambiguous_regression"
            verdict = "ambiguous_regression"
            evidence_level = "diagnostic"
            evidence_role = "inter_component_risk_not_component_penalty"
            can_seed_anchor_refinement = False
            data["ambiguous_regression_count"] = data.get("ambiguous_regression_count", 0) + 1
        else:
            attribution_status = "ambiguous_inconclusive"
            verdict = "ambiguous"
            evidence_level = "diagnostic"
            evidence_role = "diagnostic_not_component_proof"
            can_seed_anchor_refinement = False

        entry = {
            "episode_id": episode.get("episode_id", ""),
            "iteration": episode.get("iteration", 0),
            "components_changed": attribution_result.get("components_changed", []),
            "score": episode.get("score"),
            "score_delta": score_delta,
            "attribution": "ambiguous",
            "attribution_status": attribution_status,
            "verdict": verdict,
            "evidence_level": evidence_level,
            "evidence_role": evidence_role,
            "strategy_evidence": score_delta > 0,
            "evidence_scope": "interaction" if len(attribution_result.get("components_changed", [])) > 1 else "candidate",
            "evidence_class": "diagnostic",
            "can_promote_component": False,
            "can_seed_anchor_refinement": can_seed_anchor_refinement,
            "allowed_uses": {
                "as_strategy": score_delta > 0,
                "as_anchor": score_delta > 0,
                "as_lesson_source": score_delta > 0,
                "as_next_strategy": score_delta > 0,
                "as_component_proof": False,
                "as_interaction_signal": len(attribution_result.get("components_changed", [])) > 1,
                "as_hypothesis_source": score_delta > 0,
                "as_avoid_signal": False,
            },
            "used_for_component_score": False,
            "note": (
                "Multi-component edit; attribution to individual components is unsafe. "
                "Positive cases may seed anchor refinement but never component promotion."
            ),
        }
        data.setdefault("diagnostic_episodes", []).append(entry)
        data.setdefault("ambiguous_evidence", []).append(entry)
        data["ambiguous_evidence_count"] = data.get("ambiguous_evidence_count", 0) + 1

        if can_seed_anchor_refinement:
            anchor = dict(entry)
            anchor["parent_candidate"] = episode.get("parent_candidate")
            anchor["context_chars"] = episode.get("context_chars", {})
            anchor["reading_mode"] = episode.get("reading_mode")
            data.setdefault("positive_anchors", []).append(anchor)

    def _record_task_level_evidence(
        self, episode: dict, attribution_result: dict, data: dict
    ) -> None:
        episode_id = episode.get("episode_id", "")
        summary = episode.get("task_outcome_summary") or {}
        deltas = episode.get("task_failure_category_deltas") or {}
        new_pass_attribution = episode.get("new_pass_attribution") or {}
        new_passes = episode.get("task_new_passes") or []
        regressions = episode.get("task_regressions") or []
        persistent = episode.get("task_persistent_failures") or []
        if not any([summary, deltas, new_passes, regressions, persistent]):
            return

        existing = data.setdefault("task_level_evidence", [])
        if any(item.get("episode_id") == episode_id for item in existing):
            return

        components = attribution_result.get("components_changed", [])
        entry = {
            "episode_id": episode_id,
            "iteration": episode.get("iteration", 0),
            "score": episode.get("score"),
            "score_delta": float(episode.get("score_delta", 0) or 0),
            "components_changed": components,
            "attribution_type": attribution_result.get("attribution", "unknown"),
            "task_outcome_summary": summary,
            "new_passes": new_passes[:20],
            "regressions": regressions[:20],
            "persistent_failures": persistent[:20],
            "failure_category_deltas": deltas,
            "new_pass_attribution": new_pass_attribution,
            "used_for_component_score": False,
            "evidence_class": "task_diagnostic",
            "note": (
                "Task-level outcome delta; use to target failure modes, not as "
                "standalone proof of component causality."
            ),
        }
        existing.append(entry)

        stats = data.setdefault("task_failure_category_stats", {})
        for category, count in deltas.get("improved_categories", {}).items():
            bucket = stats.setdefault(
                category, {"improved": 0, "regressed": 0, "persistent": 0}
            )
            bucket["improved"] += int(count)
        for category, count in deltas.get("regressed_categories", {}).items():
            bucket = stats.setdefault(
                category, {"improved": 0, "regressed": 0, "persistent": 0}
            )
            bucket["regressed"] += int(count)
        for category, count in deltas.get("persistent_categories", {}).items():
            bucket = stats.setdefault(
                category, {"improved": 0, "regressed": 0, "persistent": 0}
            )
            bucket["persistent"] += int(count)

        for comp_name in components:
            if comp_name not in data["components"]:
                data["components"][comp_name] = self._blank_component()
            comp_entry = dict(entry)
            comp_entry["component"] = comp_name
            data["components"][comp_name].setdefault(
                "task_level_evidence", []
            ).append(comp_entry)

    def _category_evidence_strength(self, improvement_count: int) -> str:
        if improvement_count >= 10:
            return "strong"
        if improvement_count >= 6:
            return "medium"
        return "weak_clear"

    def _new_pass_evidence_strength(self, task_count: int) -> str:
        if task_count >= 4:
            return "medium"
        return "weak_clear"

    def _failure_category_attribution_signals(
        self, episode: dict, attribution_result: dict
    ) -> list[dict]:
        try:
            from memory.encoding.task_outcome_memory import FAILURE_COMPONENTS
        except Exception:
            return []

        deltas = episode.get("task_failure_category_deltas") or {}
        improved = deltas.get("improved_categories") or {}
        signals = []
        for category, raw_count in sorted(improved.items()):
            try:
                count = int(raw_count)
            except (TypeError, ValueError):
                continue
            component = FAILURE_COMPONENTS.get(category)
            if not component or count < CATEGORY_ATTRIBUTION_MIN_IMPROVEMENTS:
                continue
            signals.append({
                "failure_category": category,
                "component": component,
                "improvement_count": count,
                "confidence": "clear",
                "evidence_strength": self._category_evidence_strength(count),
            })
        return signals

    def _new_pass_attribution_signals(
        self, episode: dict, attribution_result: dict
    ) -> list[dict]:
        raw = episode.get("new_pass_attribution") or {}
        signals = []
        for component, payload in sorted(raw.items()):
            tasks = list(payload.get("tasks") or [])
            count = int(payload.get("count") or len(tasks))
            if count < NEW_PASS_ATTRIBUTION_MIN_TASKS:
                continue
            signals.append({
                "component": component,
                "tasks": tasks,
                "new_pass_count": count,
                "source_failure_category": payload.get("source_failure_category"),
                "confidence": "clear",
                "evidence_strength": self._new_pass_evidence_strength(count),
            })
        return signals

    def _record_failure_category_attribution(
        self,
        episode: dict,
        attribution_result: dict,
        data: dict,
    ) -> bool:
        signals = self._failure_category_attribution_signals(episode, attribution_result)
        data.setdefault("failure_category_attribution", [])
        recorded = False
        diff_clear_components = set(attribution_result.get("components_changed", []))
        diff_is_clear = attribution_result.get("attribution") == "clear"
        for signal in signals:
            already_clear = diff_is_clear and signal["component"] in diff_clear_components
            audit = {
                "episode_id": episode.get("episode_id", ""),
                "iteration": episode.get("iteration", 0),
                "score_delta": float(episode.get("score_delta", 0) or 0),
                "component": signal["component"],
                "failure_category": signal["failure_category"],
                "improvement_count": signal["improvement_count"],
                "confidence": "already_clear" if already_clear else signal["confidence"],
                "source_attribution": attribution_result.get("attribution", "unknown"),
            }
            data["failure_category_attribution"].append(audit)

            if signal["confidence"] != "clear" or already_clear:
                continue

            comp_name = signal["component"]
            if comp_name not in data["components"]:
                data["components"][comp_name] = self._blank_component()

            score_delta = float(episode.get("score_delta", 0) or 0)
            regression_threshold = self._config.get("regression_threshold", -3.0)
            evidence_strength = signal["evidence_strength"]
            verdict = compute_verdict(
                evidence_strength, score_delta, "clear", regression_threshold
            )
            if verdict != "effective":
                continue

            comp_info = dict(attribution_result)
            comp_info["components_changed"] = [comp_name]
            comp_info["attribution"] = "clear"
            evidence = self._evidence_record(
                episode,
                comp_info,
                evidence_strength,
                verdict,
                True,
            )
            evidence["component"] = comp_name
            evidence["attribution_source"] = "failure_category_delta"
            evidence["failure_category"] = signal["failure_category"]
            evidence["failure_category_improvements"] = signal["improvement_count"]

            comp_data = data["components"][comp_name]
            comp_data.setdefault("evidence", []).append(evidence)
            comp_data["observed_edits"] = comp_data.get("observed_edits", 0) + 1
            comp_data["total_edits"] = comp_data.get("total_edits", 0) + 1
            comp_data["effective_edits"] = comp_data.get("effective_edits", 0) + 1

            data["clear_evidence_count"] = data.get("clear_evidence_count", 0) + 1
            data["clear_positive_count"] = data.get("clear_positive_count", 0) + 1

            self._add_to_change_families(
                comp_data,
                comp_name,
                episode.get("episode_id", ""),
                [signal["failure_category"], "failure_category_delta"],
                verdict,
                score_delta,
                evidence_strength,
                episode.get("iteration", 0),
                extra_fields={
                    "attribution_source": "failure_category_delta",
                    "failure_category": signal["failure_category"],
                    "failure_category_improvements": signal["improvement_count"],
                },
            )
            self._refresh_component_stats(comp_data)
            recorded = True

        return recorded

    def _record_new_pass_attribution(
        self,
        episode: dict,
        attribution_result: dict,
        data: dict,
    ) -> bool:
        signals = self._new_pass_attribution_signals(episode, attribution_result)
        data.setdefault("new_pass_component_attribution", [])
        recorded = False
        diff_clear_components = set(attribution_result.get("components_changed", []))
        diff_is_clear = attribution_result.get("attribution") == "clear"
        for signal in signals:
            already_clear = diff_is_clear and signal["component"] in diff_clear_components
            audit = {
                "episode_id": episode.get("episode_id", ""),
                "iteration": episode.get("iteration", 0),
                "score_delta": float(episode.get("score_delta", 0) or 0),
                "component": signal["component"],
                "source_failure_category": signal.get("source_failure_category"),
                "new_pass_count": signal["new_pass_count"],
                "tasks": signal.get("tasks", [])[:20],
                "confidence": "already_clear" if already_clear else signal["confidence"],
                "source_attribution": attribution_result.get("attribution", "unknown"),
            }
            data["new_pass_component_attribution"].append(audit)
            if signal["confidence"] != "clear" or already_clear:
                continue

            comp_name = signal["component"]
            if comp_name not in data["components"]:
                data["components"][comp_name] = self._blank_component()

            score_delta = float(episode.get("score_delta", 0) or 0)
            regression_threshold = self._config.get("regression_threshold", -3.0)
            evidence_strength = signal["evidence_strength"]
            verdict = compute_verdict(
                evidence_strength, score_delta, "clear", regression_threshold
            )
            if verdict != "effective":
                continue

            comp_info = dict(attribution_result)
            comp_info["components_changed"] = [comp_name]
            comp_info["attribution"] = "clear"
            evidence = self._evidence_record(
                episode,
                comp_info,
                evidence_strength,
                verdict,
                True,
            )
            evidence["component"] = comp_name
            evidence["attribution_source"] = "new_pass_attribution"
            evidence["source_failure_category"] = signal.get("source_failure_category")
            evidence["new_pass_count"] = signal["new_pass_count"]
            evidence["new_pass_tasks"] = signal.get("tasks", [])[:20]

            comp_data = data["components"][comp_name]
            comp_data.setdefault("evidence", []).append(evidence)
            comp_data["observed_edits"] = comp_data.get("observed_edits", 0) + 1
            comp_data["total_edits"] = comp_data.get("total_edits", 0) + 1
            comp_data["effective_edits"] = comp_data.get("effective_edits", 0) + 1

            data["clear_evidence_count"] = data.get("clear_evidence_count", 0) + 1
            data["clear_positive_count"] = data.get("clear_positive_count", 0) + 1

            self._add_to_change_families(
                comp_data,
                comp_name,
                episode.get("episode_id", ""),
                [
                    signal.get("source_failure_category") or "new_pass",
                    "new_pass_attribution",
                ],
                verdict,
                score_delta,
                evidence_strength,
                episode.get("iteration", 0),
                extra_fields={
                    "attribution_source": "new_pass_attribution",
                    "source_failure_category": signal.get("source_failure_category"),
                    "new_pass_count": signal["new_pass_count"],
                    "new_pass_tasks": signal.get("tasks", [])[:20],
                },
            )
            self._refresh_component_stats(comp_data)
            recorded = True
        return recorded

    def _record_trajectory_behavior_attribution(
        self,
        episode: dict,
        data: dict,
    ) -> bool:
        try:
            from memory.encoding.trajectory_behavior import (
                aggregate_behavior_signal,
                extract_features_from_file,
                find_task_trajectory,
            )
        except Exception:
            return False

        job_dir_str = episode.get("job_dir")
        if not job_dir_str:
            return False

        score_delta = float(episode.get("score_delta", 0) or 0)
        if score_delta <= 0:
            return False


        raw = episode.get("new_pass_attribution") or {}
        new_pass_tasks: list[str] = []
        for payload in raw.values():
            if isinstance(payload, dict):
                new_pass_tasks.extend(payload.get("tasks") or [])

        new_pass_tasks.extend(episode.get("task_new_passes") or [])
        new_pass_tasks = list(dict.fromkeys(new_pass_tasks))

        if not new_pass_tasks:
            return False

        from pathlib import Path as _Path
        job_dir = _Path(job_dir_str)

        task_features: dict[str, dict] = {}
        for task_name in new_pass_tasks:
            traj_path = find_task_trajectory(job_dir, task_name)
            if traj_path is None:
                continue
            feat = extract_features_from_file(traj_path)
            if feat is not None:
                task_features[task_name] = feat

        if not task_features:
            return False

        signal = aggregate_behavior_signal(task_features)
        data.setdefault("trajectory_behavior_attribution", []).append({
            "episode_id": episode.get("episode_id", ""),
            "iteration": episode.get("iteration", 0),
            "score_delta": score_delta,
            "tasks_analyzed": signal["total_tasks"],
            "tasks_with_verify": signal["tasks_with_verify"],
            "tasks_with_cleanup": signal["tasks_with_cleanup"],
            "tasks_double_submit": signal["tasks_double_submit"],
            "mean_verify_ratio": signal["mean_verify_ratio"],
            "verify_task_names": signal["verify_task_names"],
        })

        if signal["tasks_with_verify"] < TRAJECTORY_BEHAVIOR_MIN_VERIFY_TASKS:
            return False

        comp_name = "prompt_template"
        regression_threshold = self._config.get("regression_threshold", -3.0)
        evidence_strength = "weak_clear" if signal["tasks_with_verify"] < 4 else "medium"
        verdict = compute_verdict(evidence_strength, score_delta, "clear", regression_threshold)
        if verdict != "effective":
            return False

        if comp_name not in data["components"]:
            data["components"][comp_name] = self._blank_component()
        comp_data = data["components"][comp_name]

        evidence = {
            "episode_id": episode.get("episode_id", ""),
            "iteration": episode.get("iteration", 0),
            "score_delta": score_delta,
            "evidence_strength": evidence_strength,
            "verdict": verdict,
            "attribution_source": "trajectory_behavior",
            "component": comp_name,
            "tasks_with_verify": signal["tasks_with_verify"],
            "mean_verify_ratio": signal["mean_verify_ratio"],
            "verify_task_names": signal["verify_task_names"],
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        }
        comp_data.setdefault("evidence", []).append(evidence)
        comp_data["observed_edits"] = comp_data.get("observed_edits", 0) + 1
        comp_data["total_edits"] = comp_data.get("total_edits", 0) + 1
        comp_data["effective_edits"] = comp_data.get("effective_edits", 0) + 1
        comp_data["effective_delta_sum"] = comp_data.get("effective_delta_sum", 0.0) + max(0.0, score_delta)
        comp_data["clear_positive_count"] = comp_data.get("clear_positive_count", 0) + 1

        data["clear_evidence_count"] = data.get("clear_evidence_count", 0) + 1
        data["clear_positive_count"] = data.get("clear_positive_count", 0) + 1

        self._add_to_change_families(
            comp_data,
            comp_name,
            episode.get("episode_id", ""),
            ["verify_before_submit", "trajectory_behavior"],
            verdict,
            score_delta,
            evidence_strength,
            episode.get("iteration", 0),
            extra_fields={
                "attribution_source": "trajectory_behavior",
                "tasks_with_verify": signal["tasks_with_verify"],
                "mean_verify_ratio": signal["mean_verify_ratio"],
            },
        )
        self._refresh_component_stats(comp_data)
        return True

    def _refresh_component_stats(self, comp_data: dict) -> None:
        total = comp_data.get("total_edits", 0)
        if total == 0:
            comp_data["exploration_status"] = "unexplored"
            comp_data["component_score"] = 0.5
            return

        if total <= 3:
            comp_data["exploration_status"] = "partially_explored"
        else:
            comp_data["exploration_status"] = "well_explored"

        eff = comp_data.get("effective_edits", 0)
        reg = comp_data.get("regression_edits", 0)
        clear_pos = comp_data.get("clear_positive_count", 0)



        weak_pos = max(clear_pos - eff, 0)


        score = 0.5 + 0.5 * (eff + 0.5 * weak_pos - reg) / max(total, 1)
        comp_data["component_score"] = round(max(0.0, min(1.0, score)), 2)



    def _process_episode(self, episode: dict, data: dict) -> bool:
        diff = episode.get("diff_from_parent", "")
        if not diff:
            return False

        score_delta = float(episode.get("score_delta", 0))
        episode_id = episode.get("episode_id", "")
        iteration = episode.get("iteration", 0)

        comp_info = identify_components(diff)
        components_changed = comp_info["components_changed"]
        try:
            from memory.encoding.episode_recorder import get_agent_source
            harness_source = get_agent_source(episode_id)
        except Exception:
            harness_source = ""
        component_ast_signatures = {
            comp_name: signature
            for comp_name in components_changed
            if (
                signature := component_ast_signature(
                    harness_source, COMPONENT_KEYWORDS.get(comp_name, [])
                )
            )
        }
        if component_ast_signatures:
            episode["component_ast_signatures"] = component_ast_signatures

        attribution = comp_info["attribution"]
        evidence_strength = compute_evidence_strength(
            score_delta, attribution, self._val_size, self._config
        )
        regression_threshold = self._config.get("regression_threshold", -3.0)
        verdict = compute_verdict(evidence_strength, score_delta, attribution, regression_threshold)
        used_for_component_score = attribution == "clear"
        evidence = self._evidence_record(
            episode, comp_info, evidence_strength, verdict, used_for_component_score
        )
        self._record_strategy_evidence(episode, comp_info, data)
        self._record_task_level_evidence(episode, comp_info, data)
        category_attribution_recorded = self._record_failure_category_attribution(
            episode, comp_info, data
        )
        new_pass_attribution_recorded = self._record_new_pass_attribution(
            episode, comp_info, data
        )
        trajectory_behavior_recorded = self._record_trajectory_behavior_attribution(
            episode, data
        )


        evidence_class = self.classify_evidence(episode, attribution)
        if (
            category_attribution_recorded or new_pass_attribution_recorded
        ) and evidence_class == "diagnostic":
            evidence_class = "diagnostic_with_category_clear"
        episode["evidence_class"] = evidence_class
        episode["attribution"] = attribution
        episode["components_changed"] = components_changed

        data.setdefault("attribution_audit", []).append({
            "episode_id": evidence["episode_id"],
            "score_delta": evidence["score_delta"],
            "components_changed": components_changed,
            "attribution_type": evidence["attribution_type"],
            "modified_component_count": evidence["modified_component_count"],
            "hit_counts": evidence["hit_counts"],
            "verdict": evidence["verdict"],
            "evidence_class": evidence_class,
            "category_attribution_recorded": category_attribution_recorded,
            "new_pass_attribution_recorded": new_pass_attribution_recorded,
            "trajectory_behavior_recorded": trajectory_behavior_recorded,
        })

        if not components_changed:
            return True

        if not used_for_component_score:
            if self._evidence_gating_enabled():

                self._record_diagnostic_evidence(episode, comp_info, data)
            else:

                data.setdefault("ambiguous_evidence", []).append(evidence)
                data["ambiguous_evidence_count"] = data.get("ambiguous_evidence_count", 0) + 1
                for comp_name in components_changed:
                    if comp_name not in data["components"]:
                        data["components"][comp_name] = self._blank_component()
                    comp_data = data["components"][comp_name]
                    comp_data.setdefault("ambiguous_evidence", []).append(evidence)
                    comp_data["observed_edits"] = comp_data.get("observed_edits", 0) + 1
                    if verdict == "ambiguous_regression":
                        comp_data["ambiguous_regression_edits"] = (
                            comp_data.get("ambiguous_regression_edits", 0) + 1
                        )
            return True

        data["clear_evidence_count"] = data.get("clear_evidence_count", 0) + 1
        if score_delta > 0:
            data["clear_positive_count"] = data.get("clear_positive_count", 0) + 1
        if verdict == "clear_regression":
            data["clear_regression_count"] = data.get("clear_regression_count", 0) + 1

        for comp_name in components_changed:
            if comp_name not in data["components"]:
                data["components"][comp_name] = self._blank_component()
            comp_data = data["components"][comp_name]
            hit_kw = comp_info["hit_keywords"].get(comp_name, [])
            comp_evidence = dict(evidence)
            comp_evidence["component"] = comp_name
            comp_evidence["hit_keywords"] = hit_kw
            if comp_name in component_ast_signatures:
                comp_evidence["component_ast_signature"] = component_ast_signatures[
                    comp_name
                ]
            comp_data.setdefault("evidence", []).append(comp_evidence)
            comp_data["observed_edits"] = comp_data.get("observed_edits", 0) + 1

            if verdict == "clear_regression":
                self._add_to_regression_families(
                    comp_data, comp_name, episode_id, hit_kw, score_delta, iteration
                )
            else:
                self._add_to_change_families(
                    comp_data, comp_name, episode_id, hit_kw, verdict, score_delta,
                    evidence_strength, iteration
                )

            comp_data["total_edits"] += 1
            if verdict == "effective":
                comp_data["effective_edits"] += 1
                comp_data["effective_delta_sum"] = comp_data.get("effective_delta_sum", 0.0) + max(0.0, score_delta)
            if score_delta > 0 and verdict != "clear_regression":
                comp_data["clear_positive_count"] = comp_data.get("clear_positive_count", 0) + 1
            if verdict == "clear_regression":
                comp_data["regression_edits"] += 1
                comp_data["clear_regression_edits"] += 1

            self._refresh_component_stats(comp_data)

        return True



    def _write_back_to_episodes_jsonl(self, episode: dict) -> None:
        from memory.encoding.episode_recorder import EpisodeRecorder
        recorder = EpisodeRecorder(self.output_dir)
        if not recorder.episodes_path.exists():
            return
        episode_id = episode.get("episode_id")
        if not episode_id:
            return
        lines = recorder.episodes_path.read_text(encoding="utf-8").splitlines()
        new_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                ep = json.loads(line)
                if ep.get("episode_id") == episode_id:
                    ep["attribution"] = episode.get("attribution")
                    ep["components_changed"] = episode.get("components_changed", [])
                    ep["evidence_class"] = episode.get("evidence_class")
                    if episode.get("component_ast_signatures"):
                        ep["component_ast_signatures"] = episode[
                            "component_ast_signatures"
                        ]
                    line = json.dumps(ep)
            except json.JSONDecodeError:
                pass
            new_lines.append(line)
        recorder.episodes_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    def update(self, episode: dict) -> None:
        data = self._load()
        if self._process_episode(episode, data):
            data["total_episodes"] = data.get("total_episodes", 0) + 1
            self._write_back_to_episodes_jsonl(episode)
        self._save(data)

    def build_from_all_episodes(self) -> None:
        from memory.encoding.episode_recorder import EpisodeRecorder
        episodes = EpisodeRecorder(self.output_dir).get_all_episodes()
        data = self._blank()
        for ep in episodes:
            self._process_episode(ep, data)
        data["total_episodes"] = len(episodes)
        self._save(data)

    def get_summary(self) -> str:
        data = self._load()
        lines = ["Component Evidence Summary:"]
        for comp_name, comp in data.get("components", {}).items():
            total = comp.get("total_edits", 0)
            observed = comp.get("observed_edits", total)
            if observed == 0:
                continue
            eff = comp.get("effective_edits", 0)
            reg_count = comp.get("regression_edits", 0)
            lines.append(
                f"  {comp_name}: {total} clear edits / {observed} observed, "
                f"{eff} effective, {reg_count} clear regressions"
                f"  [score={comp.get('component_score', 0):.2f}, {comp.get('exploration_status')}]"
            )
            for fam in comp.get("change_families", []):
                if fam.get("verdict") == "effective":
                    eps = fam.get("supporting_episodes", [])
                    lines.append(
                        f"    → promising: {fam['description']} ({fam['avg_score_delta']:+.1f}%)"
                        f"  [{len(eps)} episode(s)]"
                    )
            for fam in comp.get("regression_families", []):
                eps = fam.get("episodes", [])
                lines.append(
                    f"    → avoid: {fam['description']} ({fam['avg_score_delta']:+.1f}%)"
                    f"  [{len(eps)} episode(s)]"
                )
        return "\n".join(lines)





def _backfill_from_group_a() -> int:
    import hashlib
    from memory.encoding.episode_recorder import EpisodeRecorder

    EVOLVE_DIR = Path(__file__).parent.parent.parent
    summary_path = EVOLVE_DIR / "logs" / "group_A_5iter" / "evolution_summary.jsonl"
    harness_path = EVOLVE_DIR / "evolve.py"

    if not summary_path.exists():
        print("  No group_A_5iter/evolution_summary.jsonl found, skipping backfill.")
        return 0


    frontier_parents = {1: "fewshot_all", 2: "fewshot_all", 3: "mem_agent_i2_1",
                        4: "mem_agent_i2_1", 5: "mem_agent_i2_1"}

    recorder = EpisodeRecorder()
    count = 0
    for line in summary_path.read_text(encoding="utf-8").strip().split("\n"):
        if not line.strip():
            continue
        d = json.loads(line)
        iteration = d.get("iteration", 0)
        name = d.get("system", "")
        score = float(d.get("avg_val", 0))
        delta = float(d.get("delta") or 0)
        parent = frontier_parents.get(iteration, "fewshot_all")
        recorder.record({
            "episode_id": name,
            "iteration": iteration,
            "parent_candidate": parent,
            "score": score,
            "score_delta": delta,
            "token_cost": 0,
            "harness_code_path": str(harness_path),
            "proposer_output": "",
        })
        count += 1
    return count


if __name__ == "__main__":
    from memory.encoding.episode_recorder import EpisodeRecorder

    recorder = EpisodeRecorder()
    if not recorder.get_all_episodes():
        print("No episodes found — backfilling from group_A_5iter run...")
        n = _backfill_from_group_a()
        print(f"  Backfilled {n} episodes.\n")

    builder = ComponentEvidenceBuilder()
    builder.build_from_all_episodes()

    episodes = recorder.get_all_episodes()
    print(f"Total episodes processed: {len(episodes)}")
    print()
    print(builder.get_summary())
    print()


    data = builder._load()
    print(f"component_evidence.json — {len(data.get('components', {}))} components tracked")
    for comp_name, comp in data.get("components", {}).items():
        if comp.get("total_edits", 0) > 0:
            print(f"  {comp_name}:")
            print(f"    change_families: {len(comp.get('change_families', []))}")
            print(f"    regression_families: {len(comp.get('regression_families', []))}")
