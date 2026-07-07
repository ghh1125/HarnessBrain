import os
"""ReadingStrategyManager: HyperMem coarse-to-fine retrieval trigger.

Three reading modes (coarse → fine):
  full_history              - early iterations or no guidance signal
  guidance_plus_component   - one positive guidance + component evidence
  guidance_only             - >=2 confirmed guidance, minimal context
"""

import json
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory.config_loader import memory_efficiency_enabled, search_governance_enabled
from memory.encoding import component_evidence as _ce

MODE_FULL_HISTORY = "full_history"
MODE_STRATEGY_CONTEXT = "strategy_context"
MODE_ANCHOR_CONTEXT = "anchor_context"
MODE_COMPONENT_PLAYBOOK = "component_playbook"
MODE_GUIDANCE_PLUS_COMPONENT = "guidance_plus_component"
MODE_GUIDANCE_ONLY = "guidance_only"

BUDGET_GUIDANCE_ONLY = 800
BUDGET_GUIDANCE_PLUS_COMPONENT = 3000

COMPONENT_MEMORY_DIR = Path(os.environ.get("COMPONENT_MEMORY_DIR", str(Path(__file__).parent.parent.parent.parent / "workspace" / "component_memory")))

def _valid_components() -> frozenset:
    """Returns current valid component names (reflects configure() task selection)."""
    return frozenset(_ce.COMPONENT_KEYWORDS.keys())


class ReadingStrategyManager:
    def __init__(
        self,
        evidence_path: Optional[Path] = None,
        episodes_path: Optional[Path] = None,
    ):
        self.evidence_path = evidence_path or COMPONENT_MEMORY_DIR / "component_evidence.json"
        self.episodes_path = episodes_path or COMPONENT_MEMORY_DIR / "episodes.jsonl"

    # ── Hint ─────────────────────────────────────────────────

    def get_target_component_hint(self, guidance: dict) -> Optional[str]:
        """Infer likely target component from guidance (no plan available yet)."""
        # Rule 1: first valid component keyword found in recommendation text
        recommendation = guidance.get("recommendation", "") or ""
        valid = _valid_components()
        for comp in sorted(valid):
            if comp in recommendation:
                return comp

        # Rule 2: first high_priority entry's component
        hp = guidance.get("high_priority", [])
        if hp:
            comp = hp[0].get("component", "")
            if comp in valid:
                return comp

        return None

    # ── Mode decision ─────────────────────────────────────────

    def decide_reading_mode(
        self, iteration: int, guidance: dict, warm_up_iters: int
    ) -> str:
        # Rule 1: early warm_up phase → always full history
        if iteration <= warm_up_iters:
            return MODE_FULL_HISTORY

        high_priority = guidance.get("high_priority", [])
        confirmed = [g for g in high_priority if g.get("status") == "confirmed"]
        active_positive = [
            g for g in high_priority
            if g.get("advantage_score", 0) > 0 and g.get("status") == "active"
        ]
        maturity = guidance.get("evidence_maturity", {}) or {}
        utility_summary = guidance.get("memory_utility_summary", {}) or {}
        has_utility_signal = "usable_strategy_count" in utility_summary
        if has_utility_signal:
            has_anchor = bool(
                guidance.get("positive_anchors")
                or guidance.get("anchor_refinement_targets")
            )
        else:
            has_anchor = bool(
                maturity.get("has_positive_anchor")
                or guidance.get("positive_anchors")
                or guidance.get("anchor_refinement_targets")
            )
        if has_utility_signal:
            has_strategy = bool(
                utility_summary.get("usable_strategy_count", 0) > 0
                or guidance.get("effective_strategies")
                or guidance.get("strategy_reuse_targets")
            )
        else:
            has_strategy = bool(
                maturity.get("has_effective_strategy")
                or guidance.get("effective_strategies")
                or guidance.get("strategy_reuse_targets")
            )
        has_confirmed = bool(
            maturity.get("has_confirmed_direction") or len(confirmed) >= 1
        )
        has_promising = bool(
            maturity.get("has_promising_direction") or len(active_positive) >= 1
        )

        # Rule 2: avoid-only guidance is not enough to compress history.
        if not has_strategy and not has_anchor and not has_confirmed and not has_promising:
            return MODE_FULL_HISTORY

        # Rule 3: confirmed directions are the only case for guidance-only.
        if has_confirmed and high_priority:
            return MODE_GUIDANCE_ONLY

        # Rule 4: promising component evidence gets a component playbook.
        hint = self.get_target_component_hint(guidance)
        if hint and has_promising:
            return MODE_COMPONENT_PLAYBOOK

        # Rule 5: effective strategy evidence drives strategy-context reading.
        if has_strategy:
            return MODE_STRATEGY_CONTEXT

        # Rule 5: positive ambiguous anchors drive anchor-context reading.
        if has_anchor:
            return MODE_ANCHOR_CONTEXT

        return MODE_FULL_HISTORY

    # ── Compact builders ──────────────────────────────────────

    def build_compact_guidance(self, guidance: dict) -> dict:
        """Compact guidance view (~800 char budget)."""
        def _trunc(s: str, limit: int) -> str:
            return (s[:limit] + "…") if s and len(s) > limit else (s or "")

        _drop = {"evidence", "supporting_evidence", "contradicting_evidence"}

        hp = sorted(
            guidance.get("high_priority", []),
            key=lambda x: x.get("confidence", 0),
            reverse=True,
        )[:3]
        compact_hp = [{k: v for k, v in item.items() if k not in _drop} for item in hp]

        av = sorted(
            guidance.get("avoid", []),
            key=lambda x: x.get("advantage_score", 0),
        )[:3]
        compact_av = [{k: v for k, v in item.items() if k not in _drop} for item in av]

        ev_quality = guidance.get("evidence_quality") or {}
        compact_ev = (
            {"actionable_ratio": ev_quality.get("actionable_ratio")} if ev_quality else {}
        )

        warn = guidance.get("attribution_warning") or ""
        return {
            "high_priority": compact_hp,
            "avoid": compact_av,
            "recommendation": _trunc(guidance.get("recommendation", ""), 200),
            "attribution_warning": _trunc(warn, 100) or None,
            "evidence_quality": compact_ev,
        }

    def build_compact_component_evidence(self, component: str) -> dict:
        """Compact evidence for one component from component_evidence.json."""
        empty = {
            "component": component,
            "top_effective": [],
            "top_regressions": [],
            "top_ambiguous": [],
        }
        if not self.evidence_path.exists():
            return empty
        try:
            data = json.loads(self.evidence_path.read_text(encoding="utf-8"))
        except Exception:
            return empty

        comp_data = data.get("components", {}).get(component, {})

        def _slim(fam: dict) -> dict:
            return {
                "family_id": fam.get("family_id", ""),
                "description": fam.get("description", ""),
                "avg_score_delta": fam.get("avg_score_delta", 0),
                "evidence_strength": fam.get("evidence_strength", ""),
                "supporting_count": len(fam.get("supporting_episodes", [])),
            }

        change_fams = comp_data.get("change_families", [])
        effective = sorted(
            [
                _slim(f) for f in change_fams
                if f.get("status") in ("confirmed_guidance", "effective")
                or f.get("verdict") == "effective"
            ],
            key=lambda x: x["avg_score_delta"],
            reverse=True,
        )[:3]

        reg_fams = comp_data.get("regression_families", [])
        regressions = sorted(
            [_slim(f) for f in reg_fams],
            key=lambda x: x["avg_score_delta"],
        )[:3]

        ambiguous = [
            _slim(f) for f in change_fams
            if f.get("attribution_type", "clear") != "clear"
        ][:3]

        return {
            "component": component,
            "top_effective": effective,
            "top_regressions": regressions,
            "top_ambiguous": ambiguous,
        }

    # ── Context builder ───────────────────────────────────────

    def build_context(
        self,
        mode: str,
        guidance: dict,
        iteration: int,
        evolution_summary_path: Optional[Path] = None,
    ) -> dict:
        if mode == MODE_GUIDANCE_ONLY:
            _result: dict = {
                "mode": "guidance_only",
                "search_guidance": self.build_compact_guidance(guidance),
                "component_evidence": None,
                "recent_episodes": None,
                "evolution_summary": None,
            }

        elif mode in (MODE_GUIDANCE_PLUS_COMPONENT, MODE_COMPONENT_PLAYBOOK):
            hint = self.get_target_component_hint(guidance)
            _result = {
                "mode": mode,
                "target_component_hint": hint,
                "search_guidance": self.build_compact_guidance(guidance),
                "component_evidence": (
                    self.build_compact_component_evidence(hint) if hint else None
                ),
                "recent_episodes": None,
                "evolution_summary": None,
                "component_playbook": None,
            }
            try:
                if memory_efficiency_enabled() and hint:
                    _pb_path = self.evidence_path.parent / "component_playbooks.json"
                    if _pb_path.exists():
                        _pb_data = json.loads(_pb_path.read_text(encoding="utf-8"))
                        _result["component_playbook"] = (
                            _pb_data.get("playbooks", {}).get(hint)
                        )
            except Exception:
                pass

        elif mode in (MODE_STRATEGY_CONTEXT, MODE_ANCHOR_CONTEXT):
            recent_episodes: list = []
            if self.episodes_path.exists():
                try:
                    lines = self.episodes_path.read_text(encoding="utf-8").strip().split("\n")
                    for line in lines[-5:]:
                        if not line.strip():
                            continue
                        ep = json.loads(line)
                        recent_episodes.append({
                            "episode_id": ep.get("episode_id"),
                            "score": ep.get("score"),
                            "score_delta": ep.get("score_delta"),
                            "status": ep.get("status"),
                        })
                except Exception:
                    pass

            evolution_summary: Optional[str] = None
            if evolution_summary_path and evolution_summary_path.exists():
                evolution_summary = evolution_summary_path.read_text(encoding="utf-8")[-1500:]

            _result = {
                "mode": mode,
                "search_guidance": self.build_compact_guidance(guidance),
                "effective_strategies": guidance.get("effective_strategies", [])[:3],
                "interaction_strategies": guidance.get("interaction_strategies", [])[:3],
                "strategy_reuse_targets": guidance.get("strategy_reuse_targets", [])[:3],
                "hypothesis_test_targets": guidance.get("hypothesis_test_targets", [])[:3],
                "positive_anchors": guidance.get("positive_anchors", [])[:3],
                "anchor_refinement_targets": guidance.get("anchor_refinement_targets", [])[:3],
                "component_evidence": None,
                "recent_episodes": None,
                "evolution_summary": None,
            }

        else:
            # MODE_FULL_HISTORY
            recent_episodes: list = []
            if self.episodes_path.exists():
                try:
                    lines = self.episodes_path.read_text(encoding="utf-8").strip().split("\n")
                    for line in lines[-5:]:
                        if not line.strip():
                            continue
                        ep = json.loads(line)
                        recent_episodes.append({
                            "episode_id": ep.get("episode_id"),
                            "score": ep.get("score"),
                            "status": ep.get("status"),
                        })
                except Exception:
                    pass

            evolution_summary: Optional[str] = None
            if evolution_summary_path and evolution_summary_path.exists():
                evolution_summary = evolution_summary_path.read_text(encoding="utf-8")[-2000:]

            _result = {
                "mode": "full_history",
                "search_guidance": guidance,
                "component_evidence": None,
                "recent_episodes": recent_episodes or None,
                "evolution_summary": evolution_summary,
            }

        # M4 constraint layer — filter context by mode (gated)
        try:
            if search_governance_enabled():
                from memory.steering.constraint_layer import ConstraintLayer
                _result = ConstraintLayer().filter_context_by_mode(_result, mode)
        except Exception:
            pass

        return _result

    # ── Char estimation ───────────────────────────────────────

    def estimate_context_chars(self, context: dict) -> dict:
        def safe_len(obj) -> int:
            if obj is None:
                return 0
            return len(json.dumps(obj, ensure_ascii=False))

        g = safe_len(context.get("search_guidance"))
        ce = safe_len(context.get("component_evidence"))
        cp = safe_len(context.get("component_playbook"))
        anchors = safe_len(context.get("positive_anchors")) + safe_len(
            context.get("anchor_refinement_targets")
        )
        strategies = (
            safe_len(context.get("effective_strategies"))
            + safe_len(context.get("interaction_strategies"))
            + safe_len(context.get("strategy_reuse_targets"))
            + safe_len(context.get("hypothesis_test_targets"))
        )
        re = safe_len(context.get("recent_episodes"))
        es = safe_len(context.get("evolution_summary"))
        return {
            "guidance_chars": g,
            "component_evidence_chars": ce,
            "component_playbook_chars": cp,
            "positive_anchor_chars": anchors,
            "strategy_chars": strategies,
            "recent_episodes_chars": re,
            "evolution_summary_chars": es,
            "total_chars": g + ce + cp + anchors + strategies + re + es,
        }

    # ── Role-specific context builder ─────────────────────────

    def build_role_specific_context(
            self,
            base_context: dict,
            raw_guidance: dict,
            candidate_role: str) -> dict:
        """按 candidate_role 裁剪 base_context，不改变 base_context 本身。"""
        import copy
        ctx = copy.deepcopy(base_context)

        if candidate_role == "free_explore":
            ctx["effective_strategies"] = []
            ctx["interaction_strategies"] = []
            ctx["strategy_reuse_targets"] = []
            ctx["hypothesis_test_targets"] = []
            ctx["positive_anchors"] = []
            ctx["anchor_refinement_targets"] = []
            ctx["anchor_code"] = None
            ctx["component_evidence"] = None
            ctx["component_playbook"] = None
            ctx["search_guidance"] = {
                "avoid": raw_guidance.get("avoid", [])[:3],
                "unexplored": raw_guidance.get("unexplored", [])[:3],
                "recommendation": raw_guidance.get("recommendation"),
                "attribution_warning": raw_guidance.get("attribution_warning"),
            }

        elif candidate_role == "hypothesis_test":
            targets = raw_guidance.get("hypothesis_test_targets", [])
            effs = raw_guidance.get("effective_strategies", [])
            best = (targets[0] if targets else (effs[0] if effs else None))

            hypo_card = None
            if best:
                hypo_card = {
                    "episode_id": best.get("episode_id"),
                    "score": best.get("score"),
                    "evidence_scope": best.get("evidence_scope"),
                    "components_changed": best.get("components_changed", []),
                    "allowed_uses": best.get("allowed_uses", {}),
                }

            ctx["effective_strategies"] = []
            ctx["interaction_strategies"] = []
            ctx["strategy_reuse_targets"] = []
            ctx["hypothesis_test_targets"] = []
            ctx["positive_anchors"] = []
            ctx["anchor_refinement_targets"] = []
            ctx["anchor_code"] = None
            ctx["component_evidence"] = None
            ctx["component_playbook"] = None
            ctx["hypothesis_target"] = hypo_card
            # 裁剪 search_guidance，避免 full_history 模式下膨胀
            raw_sg = ctx.get("search_guidance", {})
            if isinstance(raw_sg, dict):
                hp = raw_sg.get("high_priority", [])
                ctx["search_guidance"] = {
                    "high_priority": hp[:1],
                    "avoid": raw_sg.get("avoid", [])[:3],
                    "recommendation": raw_sg.get("recommendation"),
                    "attribution_warning": raw_sg.get("attribution_warning"),
                }

        elif candidate_role == "avoid_retest":
            retest = raw_guidance.get("retest_pool", [])
            target = retest[0] if retest else None

            ctx["effective_strategies"] = []
            ctx["interaction_strategies"] = []
            ctx["strategy_reuse_targets"] = []
            ctx["hypothesis_test_targets"] = []
            ctx["positive_anchors"] = []
            ctx["anchor_refinement_targets"] = []
            ctx["anchor_code"] = None
            ctx["component_evidence"] = None
            ctx["component_playbook"] = None
            ctx["search_guidance"] = {
                "retest_target": target,
                "avoid": raw_guidance.get("avoid", [])[:2],
                "recommendation": (
                    "Retest this previously failing direction under the current "
                    f"context: {target.get('direction', '')}"
                    if target else None
                ),
                "attribution_warning": raw_guidance.get("attribution_warning"),
            }

        # strategy_reuse / component_exploit: use base_context as-is

        return ctx


# ── Unit tests ────────────────────────────────────────────────

if __name__ == "__main__":
    mgr = ReadingStrategyManager()
    warm_up = 3

    mock_guidance_empty = {
        "high_priority": [],
        "avoid": [],
        "recommendation": "无明确方向",
    }
    mock_guidance_one_positive = {
        "high_priority": [
            {
                "component": "retrieval",
                "direction": "stopword 过滤",
                "advantage_score": 1.0,
                "status": "active",
                "confidence": 0.8,
            }
        ],
        "avoid": [],
        "recommendation": "本轮优先修改 retrieval 组件",
    }
    mock_guidance_two_confirmed = {
        "high_priority": [
            {
                "component": "retrieval",
                "direction": "stopword 过滤",
                "advantage_score": 3.5,
                "status": "confirmed",
                "confidence": 0.95,
            },
            {
                "component": "prompt",
                "direction": "label primer",
                "advantage_score": 2.0,
                "status": "confirmed",
                "confidence": 0.90,
            },
        ],
        "avoid": [],
        "recommendation": "本轮优先修改 retrieval 组件",
    }

    scenarios = [
        ("A", 2, warm_up, mock_guidance_empty, MODE_FULL_HISTORY),
        ("B", 5, warm_up, mock_guidance_empty, MODE_FULL_HISTORY),
        ("C", 5, warm_up, mock_guidance_one_positive, MODE_COMPONENT_PLAYBOOK),
        ("D", 10, warm_up, mock_guidance_two_confirmed, MODE_GUIDANCE_ONLY),
    ]

    print("=== Unit Tests: decide_reading_mode ===")
    all_passed = True
    for label, iteration, wu, guidance, expected in scenarios:
        mode = mgr.decide_reading_mode(iteration, guidance, wu)
        passed = mode == expected
        hint = mgr.get_target_component_hint(guidance) if label == "C" else None
        extra = f" | hint={hint}" if hint is not None else ""
        status = "PASS" if passed else "FAIL"
        print(
            f"Scenario {label}: iter={iteration}, wu={wu} → "
            f"mode={mode} (expected={expected}) {status}{extra}"
        )
        if not passed:
            all_passed = False

    print(f"\nAll scenarios passed: {all_passed}")
