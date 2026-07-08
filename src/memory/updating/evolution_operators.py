import os
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory.config_loader import evidence_quality_enabled, memory_efficiency_enabled, search_governance_enabled
from memory.encoding.ast_structure import ast_signature_similarity

COMPONENT_MEMORY_DIR = Path(os.environ.get("COMPONENT_MEMORY_DIR", str(Path(__file__).parent.parent.parent.parent / "workspace" / "component_memory")))


class EvolutionOperators:
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or COMPONENT_MEMORY_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_path = self.output_dir / "component_evidence.json"
        self.log_path = self.output_dir / "evolution_log.jsonl"
        self.guidance_path = self.output_dir / "search_guidance.json"

        self._config: dict = {}
        try:
            from memory.config_loader import get_memory_config
            self._config = get_memory_config().get("evidence_settings", {})
        except Exception:
            pass



    def _load(self) -> dict:
        if self.evidence_path.exists():
            return json.loads(self.evidence_path.read_text(encoding="utf-8"))
        return {"last_updated": "", "total_episodes": 0, "components": {}}

    def _save(self, data: dict) -> None:
        data["last_updated"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self.evidence_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _log(self, entry: dict) -> None:
        entry["timestamp"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _fam_keywords(self, fam: dict) -> set:
        raw = fam.get("description", "")
        return {kw.strip() for part in raw.split("/") for kw in part.split(",") if kw.strip()}

    def _kw_overlap_pct(self, a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _all_fam_eps(self, fam: dict) -> list:
        return (fam.get("supporting_episodes", []) +
                fam.get("contradicting_episodes", []) +
                fam.get("episodes", []))

    def _iter_from_label(self, label: str | None) -> int:
        if isinstance(label, str) and label.startswith("iter_"):
            try:
                return int(label.replace("iter_", ""))
            except ValueError:
                return 0
        return 0

    def _adaptive_effect_scale(self, data: dict) -> float:
        eps = float(self._config.get("effect_epsilon", 1e-6))
        vals: list[float] = []
        for comp in data.get("components", {}).values():
            for fam in comp.get("change_families", []) + comp.get("regression_families", []):
                delta = abs(float(fam.get("avg_score_delta", 0) or 0))
                if delta > eps:
                    vals.append(delta)
        fallback = float(self._config.get("min_score_delta_for_effective", 2.0))
        return max(float(median(vals)) if vals else fallback, eps)

    def _history_reliability(
        self,
        fam: dict,
        current_iteration: int,
        effect_scale: float,
        *,
        polarity: str,
    ) -> float:
        eps = float(self._config.get("effect_epsilon", 1e-6))
        if polarity == "negative":
            n_support = len(fam.get("episodes", []))
            n_reverse = len(fam.get("contradicting_episodes", []))
        else:
            n_support = len(fam.get("supporting_episodes", []))
            n_reverse = len(fam.get("contradicting_episodes", []))

        consistency = (n_support + 1.0) / (n_support + n_reverse + 2.0)
        effect = min(1.0, abs(float(fam.get("avg_score_delta", 0) or 0)) / max(effect_scale, eps))

        last_iter = self._iter_from_label(fam.get("last_seen", "iter_0"))
        age = max(0, int(current_iteration or 0) - last_iter)
        freshness_lambda = float(self._config.get("freshness_lambda", 0.9))
        freshness = max(0.0, min(1.0, freshness_lambda ** age))

        reliability = (consistency * effect * freshness) ** (1.0 / 3.0)
        fam["polarity_consistency"] = round(consistency, 4)
        fam["effect_strength"] = round(effect, 4)
        fam["freshness"] = round(freshness, 4)
        fam["historical_reliability"] = round(reliability, 4)
        return reliability

    def _state_compatibility(self, fam: dict) -> float:
        q = float(fam.get("context_discount", 1.0) or 1.0)
        q = max(0.0, min(1.0, q))
        fam["state_compatibility"] = round(q, 4)
        return q

    def _family_maturity(
        self,
        fam: dict,
        current_iteration: int,
        effect_scale: float,
        *,
        polarity: str,
    ) -> float:
        r = self._history_reliability(
            fam, current_iteration, effect_scale, polarity=polarity
        )
        q = self._state_compatibility(fam)
        maturity = (r * q) ** 0.5
        fam["maturity"] = round(maturity, 4)
        fam["effect_scale"] = round(effect_scale, 4)
        return maturity



    def log_save_event(self, episode: dict) -> None:
        episode_id = episode.get("episode_id", "")
        data = self._load()

        for comp_name, comp in data.get("components", {}).items():
            all_fams = comp.get("change_families", []) + comp.get("regression_families", [])
            for fam in all_fams:
                if episode_id in self._all_fam_eps(fam):
                    fam.setdefault("weight", 1.0)
                    self._log({
                        "operation": "LogSaveEvent",
                        "episode_id": episode_id,
                        "component": comp_name,
                        "family_id": fam["family_id"],
                    })

        self._save(data)

    def save(self, episode: dict) -> None:
        self.log_save_event(episode)



    def update_weights(self) -> None:
        data = self._load()
        min_eps = self._config.get("min_episodes_for_promote", 3)

        for comp_name, comp in data.get("components", {}).items():
            total = comp.get("total_edits", 0)
            effective_count = 0
            reg_count = 0

            for fam in comp.get("change_families", []):
                if fam.get("attribution_type", "clear") != "clear":
                    continue
                n_sup = len(fam.get("supporting_episodes", []))
                n_con = len(fam.get("contradicting_episodes", []))

                if n_con > n_sup:
                    fam["evidence_strength"] = "negative"
                elif n_sup >= min_eps and n_con == 0:
                    fam["evidence_strength"] = "strong"
                elif n_sup >= 2:
                    fam["evidence_strength"] = "medium"
                elif n_sup == 1:
                    fam["evidence_strength"] = "weak"
                else:
                    fam["evidence_strength"] = "inconclusive"

                if fam.get("verdict") == "effective":
                    cd = fam.get("context_discount", 1.0)
                    effective_count += n_sup * cd
                    if n_con == 0:
                        fam["weight"] = 1.0

            for fam in comp.get("regression_families", []):
                if fam.get("attribution_type", "clear") != "clear":
                    continue
                cd = fam.get("context_discount", 1.0)
                reg_count += len(fam.get("episodes", [])) * cd

            comp["effective_edits"] = effective_count
            if total > 0:
                clear_pos = comp.get("clear_positive_count", 0)
                weak_pos = max(clear_pos - effective_count, 0)
                score = 0.5 + 0.5 * (effective_count + 0.5 * weak_pos - reg_count) / total
                comp["component_score"] = round(max(0.0, min(1.0, score)), 2)

        self._save(data)



    def merge_families(self) -> None:
        data = self._load()
        merged_any = False

        for comp_name, comp in data.get("components", {}).items():
            reg_fams = comp.get("regression_families", [])
            if len(reg_fams) < 2:
                continue

            merged_indices: set = set()
            new_fams: list = []

            for i in range(len(reg_fams)):
                if i in merged_indices:
                    continue
                fam_i = reg_fams[i]
                kw_i = self._fam_keywords(fam_i)
                to_merge = []

                for j in range(i + 1, len(reg_fams)):
                    if j in merged_indices:
                        continue
                    kw_j = self._fam_keywords(reg_fams[j])
                    if self._kw_overlap_pct(kw_i, kw_j) >= 0.5:
                        to_merge.append(j)

                if not to_merge:
                    new_fams.append(fam_i)
                    continue


                all_ids = [fam_i["family_id"]] + [reg_fams[j]["family_id"] for j in to_merge]
                all_eps: list = list(fam_i.get("episodes", []))
                all_deltas: list = [fam_i.get("avg_score_delta", 0)]
                all_descs: list = [fam_i.get("description", "")]

                for j in to_merge:
                    fj = reg_fams[j]
                    all_eps = list(set(all_eps) | set(fj.get("episodes", [])))
                    all_deltas.append(fj.get("avg_score_delta", 0))
                    all_descs.append(fj.get("description", ""))
                    merged_indices.add(j)

                n_merged = sum(1 for f in new_fams if "merged" in f.get("family_id", ""))
                new_id = f"{comp_name}_reg_merged_{n_merged + 1:03d}"
                merged_fam = {
                    "family_id": new_id,
                    "description": " / ".join(d for d in all_descs if d),
                    "episodes": all_eps,
                    "avg_score_delta": round(sum(all_deltas) / len(all_deltas), 1),
                    "status": "regression_family",
                    "first_seen": fam_i.get("first_seen", ""),
                    "last_seen": fam_i.get("last_seen", ""),
                    "weight": 1.0,
                }
                new_fams.append(merged_fam)
                self._log({
                    "operation": "Merge",
                    "component": comp_name,
                    "merged_families": all_ids,
                    "new_family_id": new_id,
                })
                merged_any = True

            comp["regression_families"] = new_fams

        if merged_any:
            self._save(data)



    def decay(self) -> None:
        data = self._load()
        affected: list = []

        for comp_name, comp in data.get("components", {}).items():
            for fam in comp.get("regression_families", []):
                old = fam.get("weight", 1.0)
                fam["weight"] = round(old * 0.5, 4)
                affected.append(f"{fam['family_id']} ({old:.4f}→{fam['weight']:.4f})")

            for fam in comp.get("change_families", []):
                if fam.get("verdict") == "inconclusive":
                    old = fam.get("weight", 1.0)
                    fam["weight"] = round(old * 0.8, 4)
                    affected.append(f"{fam['family_id']} ({old:.4f}→{fam['weight']:.4f})")

        if affected:
            self._log({
                "operation": "Decay",
                "affected_families": affected,
                "decay_factor": 0.5,
            })
            self._save(data)



    def promote(self) -> None:
        data = self._load()
        min_eps = self._config.get("min_episodes_for_promote", 3)
        min_delta = self._config.get("min_score_delta_for_effective", 2.0)
        promoted_any = False

        for comp_name, comp in data.get("components", {}).items():
            for fam in comp.get("change_families", []):
                if fam.get("attribution_type", "clear") != "clear":
                    continue
                n_sup = len(fam.get("supporting_episodes", []))
                n_con = len(fam.get("contradicting_episodes", []))
                avg_delta = fam.get("avg_score_delta", 0)

                if (n_sup >= min_eps and avg_delta >= min_delta
                        and n_con == 0 and fam.get("status") == "promising"
                        and fam.get("verdict") == "effective"):
                    before = fam["status"]
                    fam["status"] = "confirmed_guidance"
                    self._log({
                        "operation": "Promote",
                        "component": comp_name,
                        "family_id": fam["family_id"],
                        "reason": f"{n_sup} consistent positive episodes, no counterexamples",
                        "before_status": before,
                        "after_status": "confirmed_guidance",
                    })
                    promoted_any = True

        if promoted_any:
            self._save(data)



    def demote(self, episode: dict) -> None:
        if episode.get("status") != "regression":
            return

        from memory.encoding.component_evidence import identify_components
        diff = episode.get("diff_from_parent", "")
        comp_info = identify_components(diff) if diff else {}
        ep_kw: set = set()
        for kws in comp_info.get("hit_keywords", {}).values():
            ep_kw.update(kws)

        data = self._load()
        demoted_any = False
        episode_id = episode.get("episode_id", "")

        for comp_name, comp in data.get("components", {}).items():
            for fam in comp.get("change_families", []):
                if fam.get("status") != "confirmed_guidance":
                    continue
                fam_kw = self._fam_keywords(fam)
                if len(fam_kw & ep_kw) >= 1:
                    before = fam["status"]
                    fam["status"] = "conflicted"
                    fam["weight"] = round(fam.get("weight", 1.0) - 0.2, 4)
                    self._log({
                        "operation": "Demote",
                        "component": comp_name,
                        "family_id": fam["family_id"],
                        "reason": f"Counterexample observed: {episode_id}",
                        "before_status": before,
                        "after_status": "conflicted",
                    })
                    demoted_any = True

        if demoted_any:
            self._save(data)



    def migrate_on_transition(self, transition: str | dict) -> None:
        from memory.encoding.component_evidence import identify_components
        if isinstance(transition, dict):
            diff = transition.get("diff_from_parent", "")
            current_episode_id = transition.get("episode_id", "")
            current_signatures = transition.get("component_ast_signatures", {})
        else:
            diff = transition
            current_episode_id = ""
            current_signatures = {}
        comp_info = identify_components(diff or "")
        hit_counts = comp_info.get("hit_counts", {})

        data = self._load()
        affected: list = []

        for comp_name, comp in data.get("components", {}).items():
            hit_count = hit_counts.get(comp_name, 0)

            if hit_count >= 4:
                transition_discount = 0.3
            elif hit_count >= 2:
                transition_discount = 0.6
            elif hit_count == 1:
                transition_discount = 0.8
            else:
                transition_discount = None

            all_fams = comp.get("change_families", []) + comp.get("regression_families", [])

            for fam in all_fams:
                existing = fam.get("context_discount", 1.0)
                family_episode_ids = set(self._all_fam_eps(fam))
                current_signature = current_signatures.get(comp_name)
                if current_episode_id in family_episode_ids and current_signature:
                    ast_compatibility = 1.0
                else:
                    historical_evidence = [
                        item
                        for item in comp.get("evidence", [])
                        if item.get("episode_id") in family_episode_ids
                        and item.get("episode_id") != current_episode_id
                        and item.get("component_ast_signature")
                    ]
                    historical_evidence.sort(
                        key=lambda item: int(item.get("iteration", 0) or 0)
                    )
                    historical_signature = (
                        historical_evidence[-1].get("component_ast_signature")
                        if historical_evidence
                        else None
                    )
                    ast_compatibility = ast_signature_similarity(
                        historical_signature,
                        current_signature,
                    )
                if ast_compatibility is not None:
                    new_val = round(ast_compatibility, 4)
                    source_changed = fam.get("compatibility_source") != "ast_structure"
                    fam["context_discount"] = new_val
                    fam["compatibility_source"] = "ast_structure"
                    if new_val != existing or source_changed:
                        affected.append(
                            f"{fam['family_id']} ast ({existing:.2f}→{new_val:.2f})"
                        )
                    continue

                if transition_discount is not None:
                    new_val = min(existing, transition_discount)
                    if new_val != existing:
                        fam["context_discount"] = round(new_val, 4)
                        fam["compatibility_source"] = "keyword_fallback"
                        affected.append(
                            f"{fam['family_id']} discount ({existing:.2f}→{new_val:.2f})"
                        )
                else:
                    if existing < 1.0:
                        new_val = min(1.0, existing + 0.2)
                        fam["context_discount"] = round(new_val, 4)
                        fam["compatibility_source"] = "keyword_fallback"
                        affected.append(
                            f"{fam['family_id']} recovery ({existing:.2f}→{new_val:.2f})"
                        )

        if affected:
            self._log({
                "operation": "MigrateOnTransition",
                "hit_counts": {k: v for k, v in hit_counts.items() if v > 0},
                "affected_families": affected,
            })
            self._save(data)



    def generate_retest_pool(self, current_iteration: int) -> list:
        try:
            from memory.updating.memory_utility import MemoryUtilityTracker

            tracker = MemoryUtilityTracker(self.output_dir)
            states = tracker.load_states()
            items = states.get("memory_items", {})

            candidates = []
            for mid, item in items.items():
                if item.get("memory_type") != "avoid":
                    continue
                if item.get("current_status") != "active_avoid":
                    continue
                if float(item.get("avoid_strength", 1.0) or 1.0) <= 0.3:
                    continue
                last = item.get("last_retest_iter")
                if last is not None and current_iteration - int(last) < 5:
                    continue
                candidates.append({
                    "memory_id": mid,
                    "type": "avoid_retest",
                    "component": item.get("component", ""),
                    "direction": item.get("description") or item.get("family_id", ""),
                    "family_id": item.get("family_id", ""),
                    "avoid_strength": item.get("avoid_strength", 1.0),
                    "last_retest_iter": last,
                    "retest_hint": (
                        "This direction failed before; retest it under the current "
                        "harness context before treating it as permanently unsafe."
                    ),
                })

            candidates.sort(key=lambda x: x["last_retest_iter"] if x["last_retest_iter"] is not None else -1)
            return candidates[:1]
        except Exception as e:
            print(f"[retest_pool] failed: {e}")
            return []



    def generate_search_guidance(self, current_iteration: int = 0) -> None:
        data = self._load()
        _memory_utility = None
        try:
            if evidence_quality_enabled():
                from memory.updating.memory_utility import MemoryUtilityTracker
                _memory_utility = MemoryUtilityTracker(self.output_dir)
                _memory_utility.register_from_evidence(data)
        except Exception as e:
            print(f"[memory_utility] warning: {e}")
            _memory_utility = None


        current_diagnosis: dict = {}
        _top_failure_modes: list[str] = []
        try:
            from memory.encoding.episode_recorder import EpisodeRecorder
            _episodes_all = EpisodeRecorder(self.output_dir).get_all_episodes()
            if _episodes_all:
                _latest = _episodes_all[-1]
                _task_outcomes = _latest.get("task_outcome_summary") or {}
                _fc_deltas = _latest.get("task_failure_category_deltas") or {}
                _is_agent_task = bool(_task_outcomes or _fc_deltas)

                if _is_agent_task:

                    _failure_dist: dict = {}
                    if _task_outcomes:
                        _failure_dist = {
                            k: v for k, v in _task_outcomes.items()
                            if isinstance(v, int) and k not in ("total", "pass", "n_pass")
                        }
                    elif _fc_deltas:
                        _failure_dist = {k: abs(v) for k, v in _fc_deltas.items() if v != 0}
                    _top_failure_modes = sorted(
                        _failure_dist, key=lambda k: _failure_dist[k], reverse=True
                    )[:3]
                    current_diagnosis = {
                        "task_type": "agent",
                        "episode_id": _latest.get("episode_id", ""),
                        "score": _latest.get("score"),
                        "main_failure_mode": _top_failure_modes[0] if _top_failure_modes else None,
                        "failure_distribution": _failure_dist,
                        "top_failure_modes": _top_failure_modes,
                        "diagnosis": (
                            f"当前 agent 主要失败模式: "
                            f"{', '.join(_top_failure_modes[:2]) if _top_failure_modes else '未知'}。"
                            " guidance 已按对症优先级重排。"
                        ),
                    }
                else:

                    _score = _latest.get("score", 0)
                    _delta = _latest.get("score_delta", 0)
                    _status = _latest.get("status", "")
                    _is_regressing = _status in ("regression", "bug") or _delta < -1.0
                    current_diagnosis = {
                        "task_type": "classification",
                        "episode_id": _latest.get("episode_id", ""),
                        "score": _score,
                        "score_delta": _delta,
                        "status": _status,
                        "is_regressing": _is_regressing,
                        "diagnosis": (
                            f"当前分类准确率 {_score:.1%}，delta={_delta:+.1f}%。"
                            + (" 处于回归状态，优先修复而非探索新方向。" if _is_regressing
                               else " 处于改善趋势，可继续探索高分组件方向。")
                        ),
                    }
        except Exception as e:
            print(f"[current_diagnosis] warning: {e}")

        high_priority: list = []
        avoid: list = []
        explore: list = []
        unexplored: list = []
        effect_scale = self._adaptive_effect_scale(data)
        maturity_threshold = float(self._config.get("maturity_threshold", 0.5))

        for comp_name, comp in data.get("components", {}).items():
            for fam in comp.get("change_families", []):
                status = fam.get("status", "")
                is_task_attribution_signal = (
                    fam.get("attribution_source") in {
                        "failure_category_delta",
                        "new_pass_attribution",
                        "trajectory_behavior",
                    }
                    and fam.get("verdict") == "effective"
                )
                last_seen = fam.get("last_seen", "iter_0")
                last_iter = int(last_seen.replace("iter_", "")) if last_seen.startswith("iter_") else 0
                avg_delta = fam.get("avg_score_delta", 0)
                n_sup = len(fam.get("supporting_episodes", []))
                eps_evidence = [f"{ep} delta={avg_delta:+.1f}%" for ep in fam.get("supporting_episodes", [])[:3]]
                contradicting = [str(ep) for ep in fam.get("contradicting_episodes", [])[:3]]

                if status == "confirmed_guidance" or is_task_attribution_signal:
                    maturity = self._family_maturity(
                        fam, current_iteration, effect_scale, polarity="positive"
                    )
                    reason = "Clear evidence indicates this direction is effective."
                    if is_task_attribution_signal:
                        if fam.get("attribution_source") == "new_pass_attribution":
                            reason = (
                                f"New-pass evidence shows {fam.get('new_pass_count', '?')} "
                                f"task(s) moved from {fam.get('source_failure_category')} "
                                "failure to pass."
                            )
                        elif fam.get("attribution_source") == "trajectory_behavior":
                            _mvr = fam.get("mean_verify_ratio", "?")
                            _mvr_str = f"{_mvr:.0%}" if isinstance(_mvr, float) else str(_mvr)
                            reason = (
                                f"Trajectory evidence: {fam.get('tasks_with_verify', '?')} newly-passing "
                                f"task(s) showed verification behavior before submission "
                                f"(mean verify ratio {_mvr_str})."
                            )
                        else:
                            reason = (
                                f"Failure-category evidence shows {fam.get('failure_category')} "
                                f"improved in {fam.get('failure_category_improvements', '?')} task(s)."
                            )
                    if maturity < maturity_threshold:
                        explore.append({
                            "guidance_type": "explore_retest",
                            "component": comp_name,
                            "direction": fam["description"],
                            "reason": (
                                f"Previously effective, but component was structurally changed "
                                f"(state_compatibility={fam.get('state_compatibility', 1.0):.2f}, "
                                f"maturity={maturity:.2f}); "
                                "worth retesting under current harness state."
                            ),
                            "evidence": eps_evidence,
                            "maturity": round(maturity, 3),
                            "effective_strength": round(maturity, 3),
                        })
                    else:
                        high_priority.append({
                            "guidance_type": "exploit",
                            "component": comp_name,
                            "direction": f"Continue exploiting the {fam['description']} direction",
                            "reason": reason,
                            "evidence": eps_evidence,
                            "supporting_evidence": eps_evidence,
                            "contradicting_evidence": contradicting,
                            "confidence": round(maturity, 3),
                            "maturity": round(maturity, 3),
                            "historical_reliability": fam.get("historical_reliability"),
                            "state_compatibility": fam.get("state_compatibility"),
                            "attribution_source": fam.get("attribution_source", "diff"),
                            "failure_category": fam.get("failure_category"),
                            "source_failure_category": fam.get("source_failure_category"),
                            "new_pass_count": fam.get("new_pass_count"),
                            "evidence_keywords": sorted(self._fam_keywords(fam)) or None,
                        })
                elif status == "promising" and (current_iteration - last_iter) > 2:
                    maturity = self._family_maturity(
                        fam, current_iteration, effect_scale, polarity="positive"
                    )
                    item = {
                        "component": comp_name,
                        "direction": f"Revisit the {fam['description']} direction",
                        "reason": "A positive signal exists but has not been explored further.",
                        "evidence": eps_evidence,
                        "supporting_evidence": eps_evidence,
                        "contradicting_evidence": contradicting,
                        "confidence": round(maturity, 3),
                        "maturity": round(maturity, 3),
                        "historical_reliability": fam.get("historical_reliability"),
                        "state_compatibility": fam.get("state_compatibility"),
                        "evidence_keywords": sorted(self._fam_keywords(fam)) or None,
                    }
                    if maturity >= maturity_threshold:
                        high_priority.append({"guidance_type": "exploit", **item})
                    else:
                        explore.append({
                            "guidance_type": "explore_retest",
                            **item,
                            "reason": (
                                "Positive evidence exists, but its current maturity "
                                f"is below threshold ({maturity:.2f}<"
                                f"{maturity_threshold:.2f}); treat as retest evidence."
                            ),
                            "effective_strength": round(maturity, 3),
                        })

            for fam in comp.get("regression_families", []):
                eps_list = fam.get("episodes", [])
                avg_delta = fam.get("avg_score_delta", 0)
                maturity = self._family_maturity(
                    fam, current_iteration, effect_scale, polarity="negative"
                )
                context_discount = fam.get("state_compatibility", fam.get("context_discount", 1.0))
                evidence = [f"{ep} {avg_delta:+.1f}%" for ep in eps_list[:3]]
                if maturity >= maturity_threshold:
                    avoid.append({
                        "guidance_type": "avoid",
                        "component": comp_name,
                        "direction": fam.get("description", ""),
                        "reason": f"Clear regression evidence, avg_delta={avg_delta:.1f}%",
                        "evidence": evidence,
                        "supporting_evidence": evidence,
                        "contradicting_evidence": [],
                        "weight": round(maturity, 3),
                        "maturity": round(maturity, 3),
                        "historical_reliability": fam.get("historical_reliability"),
                        "state_compatibility": fam.get("state_compatibility"),
                    })
                else:
                    explore.append({
                        "guidance_type": "explore_retest",
                        "component": comp_name,
                        "direction": fam.get("description", ""),
                        "reason": (
                            f"Prior regression (avg_delta={avg_delta:.1f}%), but component was "
                            f"structurally changed (state_compatibility={context_discount:.2f}, "
                            f"maturity={maturity:.2f}); "
                            "may be viable under current harness state."
                        ),
                        "evidence": evidence,
                        "maturity": round(maturity, 3),
                        "effective_strength": round(maturity, 3),
                    })

            for fam in comp.get("change_families", []):
                if fam.get("status") in {"confirmed_guidance", "promising"}:
                    continue
                if not (fam.get("supporting_episodes") or fam.get("contradicting_episodes")):
                    continue
                maturity = self._family_maturity(
                    fam, current_iteration, effect_scale, polarity="positive"
                )
                if maturity < maturity_threshold:
                    explore.append({
                        "guidance_type": "explore_retest",
                        "component": comp_name,
                        "direction": fam.get("description", "") + " (low maturity)",
                        "reason": (
                            f"maturity={maturity:.2f}; polarity is not reliable enough "
                            "for exploit/avoid, use only as retest evidence"
                        ),
                        "evidence": [str(ep) for ep in fam.get("supporting_episodes", [])[:2]],
                        "supporting_evidence": [str(ep) for ep in fam.get("supporting_episodes", [])[:2]],
                        "contradicting_evidence": [str(ep) for ep in fam.get("contradicting_episodes", [])[:2]],
                        "maturity": round(maturity, 3),
                        "effective_strength": round(maturity, 3),
                    })



        if _top_failure_modes and high_priority:
            def _failure_relevance(item: dict) -> int:
                fc = item.get("failure_category") or item.get("source_failure_category") or ""
                for rank, mode in enumerate(_top_failure_modes):
                    if mode and fc and mode.lower() in fc.lower():
                        return -rank
                return 1
            high_priority.sort(key=_failure_relevance)


        if current_diagnosis.get("task_type") == "agent":
            unexplored = [
                "agent_loop: adaptive step limit based on task complexity (not tried)",
                "agent_loop: early termination heuristics for solved tasks",
                "llm_call: structured output format for more reliable tool use",
                "command_execution: output truncation and streaming for long-running commands",
                "error_handling: hierarchical retry with exponential backoff",
                "context_management: rolling context window with summarization",
                "prompt_template: task-specific system prompt variants",
            ]
        else:
            unexplored = [
                "retrieval: semantic similarity embeddings (not tried)",
                "retrieval: domain dictionary filtering for symptom terms",
                "prompt: two-stage verification (candidate labels, then verify)",
                "memory_update: dynamic TF-IDF weighting store",
                "retrieval: targeted retrieval for cross-iteration error cases",
            ]

        evidence_maturity = data.get("evidence_maturity", {})
        strategy_evidence_raw = sorted(
            data.get("strategy_evidence", []),
            key=lambda a: (float(a.get("score") or 0), float(a.get("score_delta") or 0)),
            reverse=True,
        )
        if _memory_utility is not None:
            strategy_evidence = [
                _memory_utility.enrich_strategy(item)
                for item in strategy_evidence_raw
            ]
        else:
            strategy_evidence = strategy_evidence_raw
        usable_strategy_evidence = [
            item for item in strategy_evidence
            if item.get("usable", True)
        ]
        historical_effective_strategies = strategy_evidence[:5]
        effective_strategies = usable_strategy_evidence[:5]
        interaction_strategies = [
            item for item in effective_strategies
            if item.get("evidence_scope") == "interaction"
        ][:5]
        historical_interaction_strategies = [
            item for item in strategy_evidence
            if item.get("evidence_scope") == "interaction"
        ][:5]
        component_strategies = [
            item for item in effective_strategies
            if item.get("evidence_scope") == "component"
        ][:5]
        strategy_reuse_targets = []
        for item in effective_strategies[:3]:
            strategy_reuse_targets.append({
                "episode_id": item.get("episode_id"),
                "score": item.get("score"),
                "score_delta": item.get("score_delta"),
                "evidence_scope": item.get("evidence_scope"),
                "components_changed": item.get("components_changed", []),
                "allowed_uses": item.get("allowed_uses", {}),
                "instruction": (
                    f"Reuse the effective strategy from {item.get('episode_id')} "
                    "at its recorded evidence scope; do not claim unsupported component causality."
                ),
            })
        hypothesis_test_targets = []
        for item in historical_interaction_strategies[:3]:
            hypothesis_test_targets.append({
                "episode_id": item.get("episode_id"),
                "score": item.get("score"),
                "score_delta": item.get("score_delta"),
                "components_changed": item.get("components_changed", []),
                "evidence_scope": item.get("evidence_scope"),
                "current_status": item.get("current_status"),
                "memory_id": item.get("memory_id"),
                "instruction": (
                    "Test one mechanism from this interaction strategy in a controlled candidate."
                ),
            })
        positive_anchors_raw = data.get("positive_anchors", [])
        if _memory_utility is not None:
            historical_positive_anchors = [
                _memory_utility.enrich_anchor(anchor)
                for anchor in positive_anchors_raw
            ]
            positive_anchors = [
                anchor for anchor in historical_positive_anchors
                if anchor.get("usable", True)
            ]
        else:
            historical_positive_anchors = positive_anchors_raw
            positive_anchors = positive_anchors_raw
        anchor_refinement_targets = []
        for anchor in sorted(
            positive_anchors,
            key=lambda a: (float(a.get("score") or 0), float(a.get("score_delta") or 0)),
            reverse=True,
        )[:3]:
            anchor_refinement_targets.append({
                "episode_id": anchor.get("episode_id"),
                "score": anchor.get("score"),
                "score_delta": anchor.get("score_delta"),
                "components_changed": anchor.get("components_changed", []),
                "evidence_level": anchor.get("evidence_level", "anchor"),
                "evidence_role": anchor.get(
                    "evidence_role", "candidate_anchor_not_component_proof"
                ),
                "can_promote_component": False,
                "instruction": (
                    f"Use {anchor.get('episode_id')} as a parent candidate for "
                    "small refinement; do not treat it as proof of one component."
                ),
            })


        ambiguous_evidence = data.get("ambiguous_evidence", [])
        n_ambig = len(ambiguous_evidence)
        n_ambig_reg = sum(
            1 for ev in ambiguous_evidence if ev.get("verdict") == "ambiguous_regression"
        )
        n_total_evidence = data.get("clear_evidence_count", 0) + n_ambig
        ambig_ratio = n_ambig / max(n_total_evidence, 1)
        attribution_warning = None
        if ambig_ratio > 0.5:
            attribution_warning = (
                f"{ambig_ratio:.0%} of episodes have ambiguous attribution; "
                "the proposer should modify exactly one component per candidate."
            )
        if n_ambig_reg:
            extra = (
                f"{n_ambig_reg} ambiguous regression episodes are attribution warnings only; "
                "they do not penalize individual components."
            )
            attribution_warning = f"{attribution_warning}; {extra}" if attribution_warning else extra

        n_ambig_pos = data.get("ambiguous_positive_count", 0)
        if n_ambig_pos:
            extra = (
                f"{n_ambig_pos} ambiguous positive episode(s) are positive interaction "
                "strategies, not single-component proof."
            )
            attribution_warning = f"{attribution_warning}; {extra}" if attribution_warning else extra




        task_outcome_guidance: dict = {}
        task_failure_focus: dict = {}
        try:
            task_outcome_path = self.output_dir / "task_outcomes.json"
            if (
                os.environ.get("RADO_ENABLE_TASK_OUTCOME_MEMORY") == "1"
                and task_outcome_path.exists()
            ):
                from memory.encoding.task_outcome_memory import TaskOutcomeMemory
                task_outcome_guidance = TaskOutcomeMemory(
                    self.output_dir
                ).guidance_summary()
                for item in task_outcome_guidance.get("category_guidance", []):
                    if (
                        item.get("treat_as_harness_evidence")
                        and item.get("suggested_component")
                    ):
                        task_failure_focus = item
                        break
        except Exception as e:
            print(f"[task_outcome_memory] warning: {e}")
            task_outcome_guidance = {}
            task_failure_focus = {}


        exploit_candidates = {
            name: c.get("component_score", 0.5)
            for name, c in data.get("components", {}).items()
            if c.get("total_edits", 0) > 0 and c.get("effective_edits", 0) > 0
        }
        explore_candidates = {
            name: c.get("component_score", 0.5)
            for name, c in data.get("components", {}).items()
            if c.get("observed_edits", c.get("total_edits", 0)) > 0
        }
        if exploit_candidates:
            best_comp = max(exploit_candidates, key=exploit_candidates.get)
            recommendation_type = "exploit"
            recommendation_reason = "Clear positive evidence exists; exploit this direction."
        elif strategy_reuse_targets:
            best_comp = "strategy"
            recommendation_type = "strategy_reuse"
            recommendation_reason = (
                "Effective strategy evidence exists; reuse it at its recorded scope "
                "without claiming unsupported component causality."
            )
        elif task_failure_focus:
            best_comp = task_failure_focus.get("suggested_component")
            recommendation_type = "task_failure_guided_explore"
            recommendation_reason = (
                "Task-level memory shows a recurring actionable failure category; "
                "target the component most directly associated with that category."
            )
        elif explore_candidates:
            best_comp = max(explore_candidates, key=explore_candidates.get)
            recommendation_type = "explore"
            recommendation_reason = (
                "No clear effective evidence yet; select the least-penalized component "
                "for single-component exploration."
            )
        else:
            best_comp = "retrieval"
            recommendation_type = "explore"
            recommendation_reason = "No usable evidence yet; explore retrieval first."

        _fallback_dirs = {
            "retrieval": "try TF-IDF weighting or domain dictionary filtering",
            "prompt": "explore a structured prompt with label priors and contrastive examples",
            "memory_update": "try an error-focused buffer",
            "parser": "explore confidence-aware parse validation",
            "state_management": "explore a compressed state representation",
            "agent_loop": "explore adaptive step limits or early termination heuristics",
            "llm_call": "try structured output format for more reliable tool use",
            "command_execution": "explore output truncation and streaming for long-running commands",
            "prompt_template": "explore task-specific system prompt variants",
            "error_handling": "try hierarchical retry with exponential backoff",
            "context_management": "explore rolling context window with summarization",
            "tool_parsing": "strengthen tool argument parsing and recovery",
        }
        best_dir = _fallback_dirs.get(best_comp, "explore a new direction")
        if recommendation_type == "strategy_reuse":
            best_strategy = strategy_reuse_targets[0]
            best_dir = f"reuse effective strategy from {best_strategy.get('episode_id')}"
        elif recommendation_type == "task_failure_guided_explore":
            best_dir = task_failure_focus.get("guidance", best_dir)
        elif high_priority:
            hp_for_best = [h for h in high_priority if h["component"] == best_comp]
            if hp_for_best:
                best_dir = hp_for_best[0]["direction"]
            else:
                best_dir = high_priority[0]["direction"]
                best_comp = high_priority[0]["component"]

        if recommendation_type == "strategy_reuse":
            recommendation = (
                f"Reuse effective strategy this round: {best_dir}. "
                "Preserve the useful strategy scope, but do not promote any component "
                "unless the evidence is component-scoped."
            )
        elif recommendation_type == "task_failure_guided_explore":
            recommendation = (
                f"Prioritize the {best_comp} component this round because task-level "
                f"memory points to {task_failure_focus.get('failure_category')}: {best_dir}"
            )
        else:
            recommendation = (
                f"Prioritize the {best_comp} component this round; "
                f"make a single-component edit: {best_dir}"
            )
        recommendation_detail = {
            "guidance_type": recommendation_type,
            "component": best_comp,
            "direction": best_dir,
            "reason": recommendation_reason,
            "supporting_evidence": [],
            "contradicting_evidence": [
                f"{av['component']}: {av['direction']} ({av['reason']})"
                for av in avoid if av.get("component") == best_comp
            ][:3],
        }
        if task_failure_focus:
            recommendation_detail["task_failure_focus"] = task_failure_focus


        inter_component: dict = {}
        try:
            from memory.encoding.inter_component import InterComponentEvidence
            inter = InterComponentEvidence(self.output_dir)
            inter_component = inter.get_guidance()
        except Exception:
            pass


        evidence_quality: dict = {}
        diagnostic_insights: list = []

        if evidence_quality_enabled():
            n_actionable = data.get("evidence_summary", {}).get(
                "actionable_count", data.get("clear_evidence_count", 0)
            )
            diag_eps = data.get("diagnostic_episodes", [])
            n_diagnostic = len(diag_eps)
            n_total = n_actionable + n_diagnostic
            actionable_ratio = round(n_actionable / max(n_total, 1), 2)
            diagnostic_ratio = round(n_diagnostic / max(n_total, 1), 2)

            reliability_warning = None
            if diagnostic_ratio > 0.5:
                reliability_warning = (
                    f"当前 {diagnostic_ratio:.0%} 的修改为多组件修改，"
                    "单组件证据不足，建议 proposer 严格遵守单组件修改约束"
                )

            evidence_quality = {
                "actionable_ratio": actionable_ratio,
                "diagnostic_ratio": diagnostic_ratio,
                "actionable_count": n_actionable,
                "diagnostic_count": n_diagnostic,
                "reliability_warning": reliability_warning,
            }


            pattern_map: dict[str, list] = {}
            for dep in diag_eps:
                comps = tuple(sorted(dep.get("components_changed", [])))
                key = "+".join(comps) if comps else "unknown"
                pattern_map.setdefault(key, []).append(dep.get("score_delta", 0))

            for pattern_key, deltas in sorted(
                pattern_map.items(), key=lambda x: -len(x[1])
            )[:5]:
                avg_d = round(sum(deltas) / len(deltas), 1)
                label = (
                    f"同时修改 [{pattern_key}]"
                    if "+" in pattern_key
                    else f"全组件同时修改"
                    if pattern_key == "unknown"
                    else f"修改 [{pattern_key}]"
                )
                diagnostic_insights.append({
                    "pattern": label,
                    "occurrences": len(deltas),
                    "avg_delta": avg_d,
                    "insight": (
                        f"多组件修改（{pattern_key}）平均得分变化 {avg_d:+.1f}%，"
                        "无法归因到单一组件，仅作诊断参考"
                    ),
                    "evidence_class": "diagnostic",
                })

        for item in task_outcome_guidance.get("category_guidance", [])[:6]:
            diagnostic_insights.append({
                "pattern": f"task_failure:{item.get('failure_category')}",
                "occurrences": item.get("count", 0),
                "avg_delta": None,
                "insight": item.get("guidance"),
                "suggested_component": item.get("suggested_component"),
                "evidence_class": "task_diagnostic",
                "treat_as_harness_evidence": item.get(
                    "treat_as_harness_evidence", True
                ),
            })

        anchor_code = None
        try:
            import pathlib as _pathlib

            if effective_strategies:
                _best = max(
                    effective_strategies,
                    key=lambda x: (
                        float(x.get("score") or 0),
                        float(x.get("score_delta") or 0),
                    ),
                )
                _eid = _best.get("episode_id")
                _project_root = _pathlib.Path(__file__).parent.parent.parent
                _code_path = _project_root / "agents" / f"{_eid}.py"

                if _eid and _code_path.exists():
                    anchor_code = {
                        "episode_id": _eid,
                        "score": _best.get("score"),
                        "score_delta": _best.get("score_delta"),
                        "evidence_scope": _best.get("evidence_scope"),
                        "components_changed": _best.get("components_changed", []),
                        "code_snippet": _code_path.read_text(encoding="utf-8")[:1200],
                        "usage": "reference_only_for_strategy_reuse",
                    }
                else:
                    print(f"[anchor] agents/{_eid}.py not found")
        except Exception as e:
            print(f"[anchor] failed: {e}")
            anchor_code = None

        retest_pool = self.generate_retest_pool(current_iteration)

        guidance = {
            "iteration": current_iteration,
            "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "evidence_maturity": evidence_maturity,
            "historical_effective_strategies": historical_effective_strategies,
            "effective_strategies": effective_strategies,
            "interaction_strategies": interaction_strategies,
            "component_strategies": component_strategies,
            "strategy_reuse_targets": strategy_reuse_targets,
            "hypothesis_test_targets": hypothesis_test_targets,
            "historical_positive_anchors": historical_positive_anchors[:5],
            "positive_anchors": positive_anchors[:5],
            "anchor_refinement_targets": anchor_refinement_targets,
            "current_diagnosis": current_diagnosis,
            "high_priority": high_priority,
            "explore": explore,
            "avoid": avoid[:12],
            "unexplored": unexplored,
            "attribution_warning": attribution_warning,
            "inter_component": inter_component,
            "recommendation": recommendation,
            "recommendation_detail": recommendation_detail,
            "evidence_quality": evidence_quality,
            "diagnostic_insights": diagnostic_insights,
            "task_outcome_guidance": task_outcome_guidance,
            "retest_pool": retest_pool,
        }

        try:
            if _memory_utility is not None:
                guidance = _memory_utility.enrich_guidance(guidance)
        except Exception as e:
            print(f"[memory_utility] guidance enrichment warning: {e}")


        try:
            from memory.updating.guidance_tracker import GuidanceTracker
            if evidence_quality_enabled():
                _tracker = GuidanceTracker(log_path=self.log_path)
                _tracker.register_guidance(
                    guidance.get("high_priority", []) + guidance.get("avoid", [])
                )
                guidance = _tracker.merge_into_guidance(guidance)
        except Exception:
            pass


        try:
            if memory_efficiency_enabled():
                from memory.steering.component_playbook import ComponentPlaybookBuilder
                _pb = ComponentPlaybookBuilder()
                guidance["component_playbook_summary"] = _pb.get_compact_summary()
        except Exception:
            pass


        try:
            if search_governance_enabled():
                from memory.steering.constraint_layer import ConstraintLayer
                _inter_path = self.output_dir / "inter_component_evidence.json"
                _inter_ev: dict = {"confirmed_conflicts": [], "co_change_patterns": [], "confirmed_synergies": []}
                if _inter_path.exists():
                    try:
                        import json as _json
                        _inter_ev = _json.loads(_inter_path.read_text(encoding="utf-8"))
                    except Exception:
                        pass
                _constraint = ConstraintLayer()
                guidance = _constraint.rank_guidance(guidance, _inter_ev)
                guidance["governance"] = _constraint.choose_governance_mode(guidance)
        except Exception:
            pass

        if "governance" not in guidance:
            try:
                from memory.steering.constraint_layer import ConstraintLayer
                guidance["governance"] = ConstraintLayer().choose_governance_mode(guidance)
            except Exception:
                pass

        if anchor_code:
            guidance["anchor_code"] = anchor_code

        self.guidance_path.write_text(
            json.dumps(guidance, indent=2, ensure_ascii=False), encoding="utf-8"
        )



    def run(self, episode: dict, total_episodes: int) -> None:
        self.log_save_event(episode)
        self.migrate_on_transition(episode)
        self.update_weights()
        self.merge_families()
        if total_episodes % 5 == 0:
            self.decay()
        self.promote()
        self.demote(episode)
        self.generate_search_guidance(episode.get("iteration", 0))


        try:
            if evidence_quality_enabled():
                from memory.encoding.direction_cluster import DirectionClusterManager
                DirectionClusterManager().update_all_clusters()
            if memory_efficiency_enabled():
                from memory.steering.component_playbook import ComponentPlaybookBuilder
                ComponentPlaybookBuilder().build_all_playbooks()
        except Exception:
            pass


if __name__ == "__main__":
    from memory.encoding.episode_recorder import EpisodeRecorder

    episodes = EpisodeRecorder().get_all_episodes()
    if not episodes:
        print("No episodes found. Run component_evidence.py first.")
        raise SystemExit(1)

    ops = EvolutionOperators()


    print(f"Running operators on {len(episodes)} episodes...")
    for ep in episodes:
        ops.log_save_event(ep)


    ops.update_weights()
    ops.merge_families()


    if len(episodes) % 5 == 0:
        ops.decay()

    ops.promote()
    for ep in episodes:
        if ep.get("status") == "regression":
            ops.demote(ep)

    current_iter = max(ep.get("iteration", 0) for ep in episodes)
    ops.generate_search_guidance(current_iter)


    if ops.log_path.exists():
        log_entries = [
            json.loads(l) for l in ops.log_path.read_text().strip().split("\n") if l.strip()
        ]
        ops_count: dict = {}
        for e in log_entries:
            ops_count[e["operation"]] = ops_count.get(e["operation"], 0) + 1
        print(f"\nevolution_log.jsonl: {len(log_entries)} entries total")
        for op, cnt in sorted(ops_count.items()):
            print(f"  {op}: {cnt}")


    guidance = json.loads(ops.guidance_path.read_text())
    print("\n=== search_guidance.json ===")
    print(f"iteration: {guidance['iteration']}")
    print(f"\nhigh_priority ({len(guidance['high_priority'])}):")
    for hp in guidance["high_priority"]:
        print(f"  [{hp['component']}] {hp['direction']}")
        print(f"    reason: {hp['reason']}, confidence={hp['confidence']:.2f}")
    print(f"\navoid ({len(guidance['avoid'])}):")
    for av in guidance["avoid"][:5]:
        print(f"  [{av['component']}] {av['direction']}")
        print(f"    reason: {av['reason']}, weight={av['weight']}")
    print(f"\nunexplored: {len(guidance['unexplored'])} directions")
    if guidance.get("attribution_warning"):
        print(f"\nattribution_warning: {guidance['attribution_warning']}")
    print(f"\nrecommendation: {guidance['recommendation']}")
