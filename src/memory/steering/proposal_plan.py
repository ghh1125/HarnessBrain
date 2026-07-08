import os
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

COMPONENT_MEMORY_DIR = Path(os.environ.get("COMPONENT_MEMORY_DIR", str(Path(__file__).parent.parent.parent.parent / "workspace" / "component_memory")))
EVO_WORKSPACE = Path(__file__).parent.parent.parent / "workspace" / "evo"

VALID_COMPONENTS = {"retrieval", "prompt", "parser", "memory_update", "state_management"}


class ProposalPlanManager:
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        evo_dir: Optional[Path] = None,
    ):
        self.output_dir = output_dir or COMPONENT_MEMORY_DIR
        self.evo_dir = evo_dir or EVO_WORKSPACE
        self.guidance_path = self.output_dir / "search_guidance.json"

    def _load_guidance(self) -> dict:
        if self.guidance_path.exists():
            return json.loads(self.guidance_path.read_text(encoding="utf-8"))
        return {}



    def generate_plan_prompt(self, iteration: int) -> str:
        guidance = self._load_guidance()

        high_priority = guidance.get("high_priority", [])
        avoid = guidance.get("avoid", [])
        unexplored = guidance.get("unexplored", [])
        attribution_warning = guidance.get("attribution_warning", "")
        recommendation = guidance.get("recommendation", "")
        governance = guidance.get("governance", {})
        effective_strategies = guidance.get("strategy_reuse_targets") or guidance.get("effective_strategies", [])
        hypothesis_targets = guidance.get("hypothesis_test_targets", [])
        positive_anchors = guidance.get("anchor_refinement_targets") or guidance.get("positive_anchors", [])

        if high_priority:
            lines = []
            for hp in high_priority[:5]:
                lines.append(f"  - [{hp['component']}] {hp['direction']}")
                if hp.get("reason"):
                    lines.append(f"    Reason: {hp['reason']}")
                if hp.get("evidence"):
                    lines.append(f"    Evidence: {', '.join(hp['evidence'][:2])}")
            high_priority_text = "\n".join(lines)
        else:
            high_priority_text = f"  No high-priority direction yet. {recommendation}"

        if avoid:
            avoid_text = "\n".join(
                f"  - [{av['component']}] {av['direction']} ({av['reason']})"
                for av in avoid[:6]
            )
        else:
            avoid_text = "  No known directions to avoid yet."

        unexplored_text = (
            "\n".join(f"  - {u}" for u in unexplored[:5])
            if unexplored
            else "  No recorded unexplored directions yet."
        )

        warn_text = attribution_warning or "No special warnings."
        governance_text = (
            f"Mode: {governance.get('mode', 'unspecified')}\n"
            f"Candidate roles: {governance.get('candidate_roles', [])}\n"
            f"Rationale: {governance.get('rationale', '')}"
        )
        if positive_anchors:
            anchor_lines = []
            for anchor in positive_anchors[:3]:
                anchor_lines.append(
                    f"  - {anchor.get('episode_id')} "
                    f"(score={anchor.get('score')}, delta={anchor.get('score_delta')})"
                )
            anchors_text = "\n".join(anchor_lines)
        else:
            anchors_text = "  No positive anchor candidates yet."
        if effective_strategies:
            strategy_lines = []
            for item in effective_strategies[:3]:
                strategy_lines.append(
                    f"  - {item.get('episode_id')} "
                    f"(score={item.get('score')}, scope={item.get('evidence_scope')})"
                )
            strategies_text = "\n".join(strategy_lines)
        else:
            strategies_text = "  No effective strategy evidence yet."
        if hypothesis_targets:
            hyp_lines = []
            for item in hypothesis_targets[:3]:
                hyp_lines.append(
                    f"  - {item.get('episode_id')}: test one mechanism from "
                    f"{item.get('components_changed', [])}"
                )
            hypotheses_text = "\n".join(hyp_lines)
        else:
            hypotheses_text = "  No interaction hypotheses yet."
        valid_comps = "/".join(sorted(VALID_COMPONENTS))

        return f"""## Step 1: Generate a Proposal Plan First

Before writing any harness code, first output a JSON proposal_plan:

```json
{{
  "target_component": "Must be one of: {valid_comps}",
  "change_family": "Specific direction name for this edit",
  "parent_candidate": "Candidate to modify from",
  "hypothesis": "Why this edit should work and what evidence supports it",
  "evidence_used": ["Concrete evidence references"],
  "avoid_families": ["Directions explicitly avoided"],
  "attribution_mode": "Must be single_component"
}}
```

Current Search Guidance (iteration {iteration}):

High Priority:
{high_priority_text}

Avoid:
{avoid_text}

Unexplored:
{unexplored_text}

Warnings:
{warn_text}

Positive Anchors:
{anchors_text}

Effective Strategies:
{strategies_text}

Hypothesis Targets:
{hypotheses_text}

Governance:
{governance_text}

Rules:
1. Choose exactly one target_component, and it must be one of {valid_comps}
2. attribution_mode must be single_component
3. Strategy evidence can guide reuse, but only component-scoped evidence can prove a component
4. parent_candidate should match the selected evidence source when reusing a strategy
5. After the plan, modify only the code for target_component

"""

    def generate_soft_plan_prompt(self, iteration: int) -> str:
        guidance = self._load_guidance()

        high_priority = guidance.get("high_priority", [])
        avoid = guidance.get("avoid", [])
        unexplored = guidance.get("unexplored", [])
        attribution_warning = guidance.get("attribution_warning", "")
        recommendation = guidance.get("recommendation", "")
        governance = guidance.get("governance", {})
        effective_strategies = guidance.get("strategy_reuse_targets") or guidance.get("effective_strategies", [])
        hypothesis_targets = guidance.get("hypothesis_test_targets", [])
        positive_anchors = guidance.get("anchor_refinement_targets") or guidance.get("positive_anchors", [])

        if high_priority:
            hp_text = "\n".join(
                f"  - [{hp['component']}] {hp['direction']}"
                for hp in high_priority[:5]
            )
        else:
            hp_text = f"  No high-priority direction yet. {recommendation}"

        avoid_text = (
            "\n".join(
                f"  - [{av['component']}] {av['direction']} ({av['reason']})"
                for av in avoid[:6]
            )
            if avoid
            else "  None recorded yet."
        )

        unexplored_text = (
            "\n".join(f"  - {u}" for u in unexplored[:5])
            if unexplored
            else "  None recorded yet."
        )

        warn_text = attribution_warning or "No special warnings."
        governance_text = (
            f"Mode: {governance.get('mode', 'unspecified')}\n"
            f"Candidate roles: {governance.get('candidate_roles', [])}\n"
            f"Rationale: {governance.get('rationale', '')}"
        )
        if positive_anchors:
            anchor_lines = []
            for anchor in positive_anchors[:3]:
                anchor_lines.append(
                    f"  - {anchor.get('episode_id')} "
                    f"(score={anchor.get('score')}, delta={anchor.get('score_delta')})"
                )
            anchors_text = "\n".join(anchor_lines)
        else:
            anchors_text = "  No positive anchor candidates yet."
        if effective_strategies:
            strategy_lines = []
            for item in effective_strategies[:3]:
                strategy_lines.append(
                    f"  - {item.get('episode_id')} "
                    f"(score={item.get('score')}, scope={item.get('evidence_scope')})"
                )
            strategies_text = "\n".join(strategy_lines)
        else:
            strategies_text = "  No effective strategy evidence yet."
        if hypothesis_targets:
            hyp_lines = []
            for item in hypothesis_targets[:3]:
                hyp_lines.append(
                    f"  - {item.get('episode_id')}: test one mechanism from "
                    f"{item.get('components_changed', [])}"
                )
            hypotheses_text = "\n".join(hyp_lines)
        else:
            hypotheses_text = "  No interaction hypotheses yet."
        valid_comps = "/".join(sorted(VALID_COMPONENTS))

        return f"""## Step 1: Generate a Proposal Plan First (Soft Guidance Mode)

Before writing any harness code, output a JSON proposal_plan:

```json
{{
  "target_component": "Primary component being changed (one of: {valid_comps})",
  "change_family": "Specific direction name for this edit",
  "parent_candidate": "Candidate to modify from",
  "hypothesis": "Why this edit should work and what evidence supports it",
  "evidence_used": ["Concrete evidence references"],
  "avoid_families": ["Directions explicitly avoided"],
  "attribution_mode": "single_component OR multi_component_justified",
  "multi_component_reason": "If multi_component_justified: explain why multiple components must change together"
}}
```

Current Search Guidance (iteration {iteration}):

High Priority:
{hp_text}

Avoid:
{avoid_text}

Unexplored:
{unexplored_text}

Warnings:
{warn_text}

Positive Anchors:
{anchors_text}

Effective Strategies:
{strategies_text}

Hypothesis Targets:
{hypotheses_text}

Governance:
{governance_text}

Rules:
1. Preserve all effective strategy evidence, but respect its evidence scope
2. strategy_reuse may reuse an interaction strategy without claiming component causality
3. hypothesis_test should isolate one mechanism from an effective strategy
4. If you must change multiple components, set attribution_mode to multi_component_justified and explain why
5. Avoid directions listed above unless you have strong specific justification

"""

    def extract_plan_from_output(self, proposer_output: str) -> Optional[dict]:

        match = re.search(r"```json\s*(.*?)\s*```", proposer_output, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(1))
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass


        match = re.search(r"```\s*(\{.*?\})\s*```", proposer_output, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(1))
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass


        match = re.search(
            r'(\{[^{}]*"target_component"[^{}]*\})', proposer_output, re.DOTALL
        )
        if match:
            try:
                obj = json.loads(match.group(1))
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

        return None

    def validate_plan(self, plan: dict) -> dict:
        checks = {
            "has_target_component": plan.get("target_component") is not None,
            "valid_component": plan.get("target_component") in VALID_COMPONENTS,
            "single_component": plan.get("attribution_mode") == "single_component",
            "has_hypothesis": len(plan.get("hypothesis", "")) > 10,
            "has_parent": plan.get("parent_candidate") is not None,
        }
        compliance_score = round(sum(checks.values()) / len(checks), 2)
        return {
            "checks": checks,
            "compliance_score": compliance_score,
            "is_valid": compliance_score >= 0.8,
        }

    def _avoid_tokens(self, avoid_families: list) -> set[str]:
        tokens: set[str] = set()
        for family in avoid_families or []:
            text = str(family).lower()
            for token in re.findall(r"[a-z0-9]+", text):
                if len(token) >= 3:
                    tokens.add(token)
        return tokens

    def audit_plan_against_diff(self, plan: dict, diff_from_parent: str) -> dict:
        from memory.encoding.component_evidence import identify_components

        info = identify_components(diff_from_parent or "")
        actual_components = info.get("components_changed", [])
        target_component = plan.get("target_component")

        actual_single_component_edit = len(actual_components) == 1
        target_component_matches_actual = (
            target_component in actual_components if target_component else False
        )
        unexpected_components = [
            component for component in actual_components if component != target_component
        ]

        avoid_tokens = self._avoid_tokens(plan.get("avoid_families", []))
        diff_lower = (diff_from_parent or "").lower()
        hit_avoid_tokens = sorted(token for token in avoid_tokens if token in diff_lower)
        avoided_regression_families_actual = not hit_avoid_tokens

        checks = {
            "actual_single_component_edit": actual_single_component_edit,
            "target_component_matches_actual": target_component_matches_actual,
            "avoided_regression_families_actual": avoided_regression_families_actual,
        }
        audited_compliance_score = round(sum(checks.values()) / len(checks), 2)

        return {
            "actual_components_changed": actual_components,
            "actual_attribution": info.get("attribution", "unknown"),
            "actual_single_component_edit": actual_single_component_edit,
            "target_component_matches_actual": target_component_matches_actual,
            "avoided_regression_families_actual": avoided_regression_families_actual,
            "unexpected_components": unexpected_components,
            "hit_avoid_tokens": hit_avoid_tokens,
            "audited_compliance_score": audited_compliance_score,
            "is_audited_valid": audited_compliance_score >= 0.8,
        }

    def save_plan(
        self,
        iteration: int,
        candidate_id: str,
        plan: dict,
        validation: dict,
    ) -> None:
        plan_dir = self.evo_dir / f"iter_{iteration:02d}" / candidate_id
        plan_dir.mkdir(parents=True, exist_ok=True)

        record = {
            "iteration": iteration,
            "candidate_id": candidate_id,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "plan": plan,
            "compliance": {
                "followed_guidance": validation.get("is_valid", False),
                "target_component_matches": validation["checks"].get("valid_component", False),
                "single_component_edit": validation["checks"].get("single_component", False),
                "avoided_regression_families": True,
            },
            "validation": validation,
        }

        plan_path = plan_dir / "proposal_plan.json"
        plan_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def append_context_metadata(
        self,
        iteration: int,
        candidate_id: str,
        metadata: dict,
    ) -> None:
        plan_path = self.evo_dir / f"iter_{iteration:02d}" / candidate_id / "proposal_plan.json"
        if not plan_path.exists():
            return
        try:
            record = json.loads(plan_path.read_text(encoding="utf-8"))
            record["history_context_mode"] = metadata.get("history_context_mode")
            record["prompt_chars"] = {
                "evolution_summary": metadata.get("evolution_summary_chars", 0),
                "guidance": metadata.get("guidance_chars", 0),
                "frontier": metadata.get("frontier_chars", 0),
                "total": sum(
                    metadata.get(k, 0)
                    for k in ("evolution_summary_chars", "guidance_chars", "frontier_chars")
                ),
            }
            plan_path.write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

    def audit_saved_plan(
        self,
        iteration: int,
        candidate_id: str,
        diff_from_parent: str,
    ) -> Optional[dict]:
        plan_path = self.evo_dir / f"iter_{iteration:02d}" / candidate_id / "proposal_plan.json"
        if not plan_path.exists():
            return None

        record = json.loads(plan_path.read_text(encoding="utf-8"))
        plan = record.get("plan", {})
        audit = self.audit_plan_against_diff(plan, diff_from_parent)
        record.setdefault("compliance", {})["audit"] = audit
        plan_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return audit

    def compute_session_compliance(self) -> dict:
        plan_files = sorted(self.evo_dir.glob("iter_*/*/proposal_plan.json"))

        records = []
        for pf in plan_files:
            try:
                records.append(json.loads(pf.read_text(encoding="utf-8")))
            except Exception:
                continue

        total = len(records)
        if total == 0:
            return {
                "total_plans": 0,
                "declared_compliance": {
                    "valid_plans": 0,
                    "compliance_rate": 0.0,
                    "single_component_rate": 0.0,
                    "followed_guidance_rate": 0.0,
                },
                "audited_compliance": {
                    "audited_plans": 0,
                    "valid_audited_plans": 0,
                    "audited_compliance_rate": 0.0,
                    "actual_single_component_rate": 0.0,
                    "target_component_match_rate": 0.0,
                    "avoided_regression_family_rate": 0.0,
                },
            }

        valid = sum(1 for r in records if r.get("validation", {}).get("is_valid", False))
        single_comp = sum(
            1 for r in records
            if r.get("compliance", {}).get("single_component_edit", False)
        )
        followed = sum(
            1 for r in records
            if r.get("compliance", {}).get("followed_guidance", False)
        )

        audited_records = [
            r for r in records if r.get("compliance", {}).get("audit") is not None
        ]
        audited_total = len(audited_records)
        valid_audited = sum(
            1 for r in audited_records
            if r.get("compliance", {}).get("audit", {}).get("is_audited_valid", False)
        )
        actual_single = sum(
            1 for r in audited_records
            if r.get("compliance", {}).get("audit", {}).get("actual_single_component_edit", False)
        )
        target_match = sum(
            1 for r in audited_records
            if r.get("compliance", {}).get("audit", {}).get("target_component_matches_actual", False)
        )
        avoided_actual = sum(
            1 for r in audited_records
            if r.get("compliance", {}).get("audit", {}).get("avoided_regression_families_actual", False)
        )

        return {
            "total_plans": total,
            "valid_plans": valid,
            "compliance_rate": round(valid / total, 2),
            "single_component_rate": round(single_comp / total, 2),
            "followed_guidance_rate": round(followed / total, 2),
            "declared_compliance": {
                "valid_plans": valid,
                "compliance_rate": round(valid / total, 2),
                "single_component_rate": round(single_comp / total, 2),
                "followed_guidance_rate": round(followed / total, 2),
            },
            "audited_compliance": {
                "audited_plans": audited_total,
                "valid_audited_plans": valid_audited,
                "audited_compliance_rate": round(valid_audited / audited_total, 2)
                if audited_total else 0.0,
                "actual_single_component_rate": round(actual_single / audited_total, 2)
                if audited_total else 0.0,
                "target_component_match_rate": round(target_match / audited_total, 2)
                if audited_total else 0.0,
                "avoided_regression_family_rate": round(avoided_actual / audited_total, 2)
                if audited_total else 0.0,
            },
        }


if __name__ == "__main__":
    mgr = ProposalPlanManager()

    print("=== generate_plan_prompt (iteration 6) ===")
    prompt_text = mgr.generate_plan_prompt(6)
    print(prompt_text[:1200])

    print("\n=== extract_plan_from_output (synthetic test) ===")
    fake_output = '''
Here is my proposal plan:

```json
{
  "target_component": "retrieval",
  "change_family": "tfidf_weighting",
  "parent_candidate": "mem_agent_i2_1",
  "hypothesis": "Add TF-IDF weighting on top of stopword filtering to reduce high-frequency noise.",
  "evidence_used": ["stopword_filtering effective (+2.0%)", "search_guidance suggests further retrieval exploration"],
  "avoid_families": ["jaccard_balancing"],
  "attribution_mode": "single_component"
}
```

Now here is the Python code:
import json
'''
    plan = mgr.extract_plan_from_output(fake_output)
    print(f"extracted plan: {plan}")
    if plan:
        validation = mgr.validate_plan(plan)
        print(f"validation: {validation}")
        mgr.save_plan(6, "mem_agent_i6_test", plan, validation)
        print("saved to workspace/evo/iter_06/mem_agent_i6_test/proposal_plan.json")

    print("\n=== compute_session_compliance ===")
    stats = mgr.compute_session_compliance()
    print(json.dumps(stats, indent=2, ensure_ascii=False))
