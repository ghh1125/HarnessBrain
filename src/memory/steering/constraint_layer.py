
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class ConstraintLayer:


    def choose_governance_mode(self, guidance: dict) -> dict:
        maturity = guidance.get("evidence_maturity", {}) or {}
        has_confirmed = bool(maturity.get("has_confirmed_direction"))
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
            has_historical_strategy = bool(
                utility_summary.get("historical_effective_count", 0) > 0
                or guidance.get("historical_effective_strategies")
            )
        else:
            has_strategy = bool(
                maturity.get("has_effective_strategy")
                or guidance.get("effective_strategies")
                or guidance.get("strategy_reuse_targets")
            )
            has_historical_strategy = has_strategy

        if has_confirmed:
            mode = "strong_component_boundary"
            roles = ["component_exploit", "component_exploit", "free_explore"]
            severity_policy = "strict"
            rationale = "Confirmed direction exists; constrain edits to component boundaries."
        elif has_strategy or has_anchor:
            mode = "strategy_reuse_soft_boundary"
            roles = ["strategy_reuse", "hypothesis_test", "free_explore"]
            severity_policy = "soft"
            rationale = (
                "Effective strategy evidence exists, but component causality is not "
                "confirmed; reuse strategy scope and test hypotheses separately."
            )
        elif has_historical_strategy:
            mode = "audit_only_exploration"
            roles = ["hypothesis_test", "free_explore", "free_explore"]
            severity_policy = "audit_only"
            rationale = (
                "Only historical/stale strategy evidence exists; avoid direct reuse "
                "and test narrower hypotheses while preserving exploration."
            )
        else:
            mode = "audit_only_exploration"
            roles = ["free_explore", "free_explore", "free_explore"]
            severity_policy = "audit_only"
            rationale = "No positive anchor or confirmed direction yet; preserve exploration."

        retest_pool = guidance.get("retest_pool", []) or []
        if retest_pool and "free_explore" in roles:
            idx = len(roles) - 1 - roles[::-1].index("free_explore")
            roles = list(roles)
            roles[idx] = "avoid_retest"

        return {
            "mode": mode,
            "candidate_roles": roles,
            "severity_policy": severity_policy,
            "rationale": rationale,
            "retest_pool": retest_pool,
        }



    def compute_confidence_score(
        self,
        guidance_item: dict,
        inter_component_evidence: dict,
    ) -> tuple:
        base_score = guidance_item.get("confidence", 0.5)
        component = guidance_item.get("component")
        confirmed_conflicts = inter_component_evidence.get("confirmed_conflicts", [])


        is_single = (
            guidance_item.get("attribution_mode") == "single_component"
            or "components" not in guidance_item
        )


        single_component_bonus = 0.2 if is_single else 0.0


        if is_single:
            component_in_conflict = any(
                component in c.get("components", [])
                for c in confirmed_conflicts
            )
            co_change_penalty = -0.1 if component_in_conflict else 0.0
        else:
            guidance_components = guidance_item.get("components", [component])
            max_overlap = max(
                (
                    len(set(guidance_components) & set(c.get("components", [])))
                    for c in confirmed_conflicts
                ),
                default=0,
            )
            co_change_penalty = -0.4 if max_overlap >= 2 else -0.1


        advantage_score = guidance_item.get("advantage_score", 0)
        if advantage_score > 2:
            advantage_bonus = 0.15
        elif advantage_score > 0:
            advantage_bonus = 0.05
        elif advantage_score < -2:
            advantage_bonus = -0.15
        else:
            advantage_bonus = 0.0

        final_score = max(
            0.0,
            min(
                1.0,
                base_score + single_component_bonus + co_change_penalty + advantage_bonus,
            ),
        )

        confidence_factors = {
            "base_score": base_score,
            "single_component_bonus": single_component_bonus,
            "co_change_penalty": co_change_penalty,
            "advantage_bonus": advantage_bonus,
        }

        return final_score, confidence_factors



    def rank_guidance(
        self,
        guidance: dict,
        inter_component_evidence: dict,
    ) -> dict:
        result = dict(guidance)

        hp = list(guidance.get("high_priority", []))
        for item in hp:
            score, factors = self.compute_confidence_score(item, inter_component_evidence)
            item["confidence_score"] = round(score, 4)
            item["confidence_factors"] = factors
        hp.sort(key=lambda x: x.get("confidence_score", 0), reverse=True)
        result["high_priority"] = hp

        avoid = list(guidance.get("avoid", []))
        for item in avoid:
            score, factors = self.compute_confidence_score(item, inter_component_evidence)
            item["confidence_score"] = round(score, 4)
            item["confidence_factors"] = factors
        avoid.sort(key=lambda x: x.get("confidence_score", 0))
        result["avoid"] = avoid

        return result



    def validate_proposal_plan(
        self,
        plan: dict,
        inter_component_evidence: dict,
        governance_mode: str | None = None,
    ) -> dict:
        confirmed_conflicts = inter_component_evidence.get("confirmed_conflicts", [])
        target_component = plan.get("target_component")
        attribution_mode = plan.get("attribution_mode", "single_component")
        is_single = attribution_mode == "single_component"

        plan_components = plan.get("components", [target_component])
        if target_component and target_component not in plan_components:
            plan_components = [target_component]

        constraint_warnings: list = []
        _warning_seen: set = set()
        severity_level = 0

        def _add_warning(msg: str) -> None:
            if msg not in _warning_seen:
                _warning_seen.add(msg)
                constraint_warnings.append(msg)

        audit_only = governance_mode == "audit_only_exploration"
        anchor_soft = governance_mode in (
            "anchor_refinement_soft_boundary",
            "strategy_reuse_soft_boundary",
        )
        strict_boundary = governance_mode == "strong_component_boundary"


        if not is_single and not audit_only:
            _add_warning(
                "Plan declares multi-component edit; attribution will be ambiguous. "
                "Consider restricting to single component."
            )
            severity_level = max(severity_level, 1)


        for conflict in confirmed_conflicts:
            conflict_components = set(conflict.get("components", []))
            plan_set = set(plan_components)
            overlap = plan_set & conflict_components

            if is_single:
                if target_component in conflict_components:
                    _add_warning(
                        f"{target_component} appears in a confirmed conflict pattern; "
                        "safe to modify alone, but avoid co-changing with conflicting components."
                    )
                    severity_level = max(severity_level, 1)
            else:
                if len(overlap) >= 2 and not audit_only:
                    _add_warning(
                        f"Plan components {sorted(overlap)} overlap >= 2 with a confirmed "
                        "conflict pattern; high-risk multi-component edit."
                    )
                    severity_level = max(severity_level, 3 if strict_boundary else 2)


        co_patterns = inter_component_evidence.get("co_change_patterns", [])
        for pattern in co_patterns:
            if (
                target_component in pattern.get("components", [])
                and pattern.get("avg_delta", 0) < -3.0
                and not is_single
            ):
                _add_warning(
                    f"Target component appears in a multi-component co-change pattern "
                    f"with avg_delta={pattern['avg_delta']:.1f}%."
                )
                severity_level = max(severity_level, 2)

        severity_map = {0: "none", 1: "low", 2: "medium", 3: "high"}

        return {
            "plan_valid": True,
            "governance_mode": governance_mode or "unspecified",
            "constraint_warnings": constraint_warnings,
            "constraint_severity": severity_map[severity_level],
            "recommendation": (
                "Restrict to single-component edit for clean attribution."
                if not is_single
                else None
            ),
        }



    def filter_context_by_mode(self, context: dict, reading_mode: str) -> dict:
        result = dict(context)

        if reading_mode == "guidance_only":
            result.pop("inter_component", None)

        elif reading_mode in ("guidance_plus_component", "component_playbook", "anchor_context"):
            inter = context.get("inter_component")
            if inter and isinstance(inter, dict):
                result["inter_component"] = {
                    "confirmed_synergies": inter.get("confirmed_synergies", []),
                    "confirmed_conflicts": inter.get("confirmed_conflicts", []),
                }



        return result




if __name__ == "__main__":
    mock_inter_evidence = {
        "confirmed_conflicts": [
            {
                "components": [
                    "retrieval", "prompt", "parser",
                    "memory_update", "state_management",
                ],
                "avg_delta": -6.4,
            }
        ],
        "co_change_patterns": [],
        "confirmed_synergies": [],
    }

    constraint = ConstraintLayer()
    all_passed = True


    item1 = {
        "component": "retrieval",
        "confidence": 0.8,
        "advantage_score": 1.5,
        "attribution_mode": "single_component",
    }
    score1, factors1 = constraint.compute_confidence_score(item1, mock_inter_evidence)

    expected1 = round(0.8 + 0.2 - 0.1 + 0.05, 4)
    ok1 = abs(score1 - expected1) < 1e-6
    print(f"Scenario 1 (single-component guidance): score={score1:.4f}  "
          f"expected={expected1:.4f}  factors={factors1}  {'PASS' if ok1 else 'FAIL'}")
    all_passed = all_passed and ok1


    item2 = {
        "component": "retrieval",
        "components": ["retrieval", "prompt"],
        "confidence": 0.8,
        "advantage_score": 0,
        "attribution_mode": "multi_component",
    }
    score2, factors2 = constraint.compute_confidence_score(item2, mock_inter_evidence)

    expected2 = round(0.8 + 0 - 0.4 + 0.0, 4)
    ok2 = abs(score2 - expected2) < 1e-6
    print(f"Scenario 2 (multi-component guidance):  score={score2:.4f}  "
          f"expected={expected2:.4f}  factors={factors2}  {'PASS' if ok2 else 'FAIL'}")
    all_passed = all_passed and ok2


    plan3 = {
        "target_component": "retrieval",
        "attribution_mode": "single_component",
    }
    result3 = constraint.validate_proposal_plan(plan3, mock_inter_evidence)
    ok3 = result3["constraint_severity"] == "low"
    print(f"Scenario 3 (single-component plan):     severity={result3['constraint_severity']}  "
          f"warnings={result3['constraint_warnings']}  {'PASS' if ok3 else 'FAIL'}")
    all_passed = all_passed and ok3


    plan4 = {
        "target_component": "retrieval",
        "components": ["retrieval", "prompt"],
        "attribution_mode": "multi_component",
    }
    result4 = constraint.validate_proposal_plan(plan4, mock_inter_evidence)
    ok4 = result4["constraint_severity"] == "medium"
    print(f"Scenario 4 (multi-component plan):      severity={result4['constraint_severity']}  "
          f"warnings={result4['constraint_warnings']}  {'PASS' if ok4 else 'FAIL'}")
    all_passed = all_passed and ok4


    ctx5 = {
        "search_guidance": {"high_priority": []},
        "inter_component": {"confirmed_conflicts": [{"components": ["a", "b"]}]},
    }
    filtered5 = constraint.filter_context_by_mode(ctx5, "guidance_only")
    ok5 = "inter_component" not in filtered5
    print(f"Scenario 5 (filter guidance_only):      inter_component_removed={ok5}  "
          f"{'PASS' if ok5 else 'FAIL'}")
    all_passed = all_passed and ok5

    print(f"\nAll scenarios passed: {all_passed}")
