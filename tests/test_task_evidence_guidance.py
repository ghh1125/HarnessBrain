import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory.encoding.component_evidence import ComponentEvidenceBuilder, configure
from memory.encoding.ast_structure import (
    ast_node_count,
    ast_signature_similarity,
    ast_tree_edit_distance,
    component_ast_signature,
)
from memory.updating.evolution_operators import EvolutionOperators
from memory.encoding.task_outcome_memory import TaskOutcomeMemory


class TaskEvidenceGuidanceTests(unittest.TestCase):
    def setUp(self):
        self._old_task_mem_env = os.environ.get(
            "HARNESSBRAIN_ENABLE_TASK_OUTCOME_MEMORY"
        )
        os.environ["HARNESSBRAIN_ENABLE_TASK_OUTCOME_MEMORY"] = "1"
        configure("terminal")

    def tearDown(self):
        configure("text")
        if self._old_task_mem_env is None:
            os.environ.pop("HARNESSBRAIN_ENABLE_TASK_OUTCOME_MEMORY", None)
        else:
            os.environ["HARNESSBRAIN_ENABLE_TASK_OUTCOME_MEMORY"] = self._old_task_mem_env

    def test_failure_category_delta_can_promote_component_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            episode = {
                "episode_id": "native_tool_calling",
                "iteration": 4,
                "score": 38.0,
                "score_delta": 5.0,
                "diff_from_parent": (
                    "@@\n"
                    "+def parse_tool_json(tool_call):\n"
                    "+    parsed = json.loads(tool_call.arguments)\n"
                    "+system_prompt = instruction_template\n"
                    "+while step < max_steps:\n"
                    "+    state['retry'] = retry_count\n"
                    "+    messages.append(system_prompt)\n"
                ),
                "task_failure_category_deltas": {
                    "improved_categories": {"tool_argument_parse": 8},
                    "regressed_categories": {},
                    "persistent_categories": {},
                },
            }

            ComponentEvidenceBuilder(out).update(episode)
            EvolutionOperators(out).generate_search_guidance(current_iteration=5)
            evidence = json.loads((out / "component_evidence.json").read_text())
            guidance = json.loads((out / "search_guidance.json").read_text())

        tool = evidence["components"]["tool_parsing"]
        self.assertEqual(tool["effective_edits"], 1)
        self.assertEqual(tool["change_families"][0]["verdict"], "effective")
        self.assertEqual(
            tool["change_families"][0]["attribution_source"],
            "failure_category_delta",
        )
        self.assertEqual(guidance["high_priority"][0]["component"], "tool_parsing")
        self.assertEqual(
            guidance["high_priority"][0]["attribution_source"],
            "failure_category_delta",
        )

    def test_new_pass_attribution_can_promote_component_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            episode = {
                "episode_id": "loop_budget_fix",
                "iteration": 5,
                "score": 42.0,
                "score_delta": 4.0,
                "diff_from_parent": (
                    "@@\n"
                    "+while step < max_steps:\n"
                    "+    if solved_marker_seen:\n"
                    "+        done = True\n"
                    "+        break\n"
                    "+system_prompt = instruction_template\n"
                    "+messages.append(system_prompt)\n"
                ),
                "new_pass_attribution": {
                    "agent_loop": {
                        "tasks": ["task_a", "task_b"],
                        "source_failure_category": "agent_timeout",
                        "count": 2,
                    }
                },
            }

            ComponentEvidenceBuilder(out).update(episode)
            evidence = json.loads((out / "component_evidence.json").read_text())

        loop = evidence["components"]["agent_loop"]
        self.assertEqual(loop["effective_edits"], 1)
        self.assertEqual(
            loop["change_families"][0]["attribution_source"],
            "new_pass_attribution",
        )
        self.assertEqual(loop["change_families"][0]["new_pass_count"], 2)

    def test_task_outcome_guidance_can_drive_next_search_recommendation(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            TaskOutcomeMemory(out).update_episode({
                "episode_id": "agent_a",
                "iteration": 1,
                "score": 0.0,
                "task_outcomes": {
                    "extract-elf": {
                        "pass_rate": 0.0,
                        "failure_category": "agent_timeout",
                        "failure_component": "agent_loop",
                    },
                },
            })
            EvolutionOperators(out).generate_search_guidance(current_iteration=2)
            guidance = json.loads((out / "search_guidance.json").read_text())

        self.assertIn("task_outcome_guidance", guidance)
        self.assertEqual(
            guidance["recommendation_detail"]["guidance_type"],
            "task_failure_guided_explore",
        )
        self.assertEqual(guidance["recommendation_detail"]["component"], "agent_loop")

    def test_evidence_maturity_geometrically_aggregates_reliability_and_compatibility(self):
        with tempfile.TemporaryDirectory() as td:
            operators = EvolutionOperators(Path(td))
            family = {
                "supporting_episodes": ["episode_a", "episode_b"],
                "contradicting_episodes": [],
                "avg_score_delta": 1.0,
                "last_seen": "iter_5",
                "context_discount": 0.25,
            }

            reliability = operators._history_reliability(
                family,
                current_iteration=5,
                effect_scale=1.0,
                polarity="positive",
            )
            compatibility = operators._state_compatibility(family)
            maturity = operators._family_maturity(
                family,
                current_iteration=5,
                effect_scale=1.0,
                polarity="positive",
            )

        self.assertAlmostEqual(
            maturity,
            (reliability * compatibility) ** 0.5,
        )

    def test_history_reliability_geometrically_aggregates_its_three_factors(self):
        with tempfile.TemporaryDirectory() as td:
            operators = EvolutionOperators(Path(td))
            operators._config["freshness_lambda"] = 0.9
            family = {
                "supporting_episodes": ["episode_a", "episode_b"],
                "contradicting_episodes": [],
                "avg_score_delta": 0.5,
                "last_seen": "iter_5",
            }

            reliability = operators._history_reliability(
                family,
                current_iteration=6,
                effect_scale=1.0,
                polarity="positive",
            )

        consistency = 3.0 / 4.0
        effect = 0.5
        freshness = 0.9
        self.assertAlmostEqual(
            reliability,
            (consistency * effect * freshness) ** (1.0 / 3.0),
        )

    def test_ast_signature_ignores_identifier_names_but_detects_structure(self):
        original = """
def parse_result(raw):
    payload = json.loads(raw)
    return payload["tool"]
"""
        renamed = """
def parse_result(response):
    parsed = json.loads(response)
    return parsed["result"]
"""
        changed = """
def parse_result(response):
    parsed = json.loads(response)
    if "result" not in parsed:
        return None
    return parsed["result"]
"""

        original_signature = component_ast_signature(
            original, ["parse", "json", "result"]
        )
        renamed_signature = component_ast_signature(
            renamed, ["parse", "json", "result"]
        )
        changed_signature = component_ast_signature(
            changed, ["parse", "json", "result"]
        )

        self.assertEqual(
            ast_signature_similarity(original_signature, renamed_signature),
            1.0,
        )
        self.assertLess(
            ast_signature_similarity(original_signature, changed_signature),
            1.0,
        )

    def test_ast_tree_edit_distance_counts_single_node_insertion(self):
        original_signature = component_ast_signature(
            """
def parse_result(raw):
    return raw
""",
            ["parse"],
        )
        changed_signature = component_ast_signature(
            """
def parse_result(raw):
    pass
    return raw
""",
            ["parse"],
        )

        distance = ast_tree_edit_distance(
            original_signature,
            changed_signature,
        )
        expected_similarity = 1.0 - 1.0 / (
            ast_node_count(original_signature)
            + ast_node_count(changed_signature)
        )

        self.assertEqual(distance, 1)
        self.assertAlmostEqual(
            ast_signature_similarity(original_signature, changed_signature),
            expected_similarity,
        )

    def test_transition_uses_ast_similarity_when_signatures_are_available(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            historical_signature = component_ast_signature(
                """
def parse_result(raw):
    return json.loads(raw)
""",
                ["parse", "json"],
            )
            current_signature = component_ast_signature(
                """
def parse_result(raw):
    result = json.loads(raw)
    if result is None:
        return {}
    return result
""",
                ["parse", "json"],
            )
            evidence = {
                "components": {
                    "tool_parsing": {
                        "evidence": [
                            {
                                "episode_id": "historical",
                                "iteration": 1,
                                "component_ast_signature": historical_signature,
                            }
                        ],
                        "change_families": [
                            {
                                "family_id": "tool_parsing_001",
                                "supporting_episodes": ["historical"],
                                "contradicting_episodes": [],
                                "context_discount": 1.0,
                            }
                        ],
                        "regression_families": [],
                    }
                }
            }
            (out / "component_evidence.json").write_text(
                json.dumps(evidence), encoding="utf-8"
            )
            expected = ast_signature_similarity(
                historical_signature, current_signature
            )

            EvolutionOperators(out).migrate_on_transition(
                {
                    "episode_id": "current",
                    "diff_from_parent": "",
                    "component_ast_signatures": {
                        "tool_parsing": current_signature,
                    },
                }
            )
            updated = json.loads(
                (out / "component_evidence.json").read_text(encoding="utf-8")
            )

        family = updated["components"]["tool_parsing"]["change_families"][0]
        self.assertAlmostEqual(family["context_discount"], expected, places=4)
        self.assertEqual(family["compatibility_source"], "ast_structure")

    def test_component_evidence_stores_ast_signature_when_source_is_available(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "memory"
            agents = Path(td) / "agents"
            agents.mkdir()
            (agents / "parser_fix.py").write_text(
                """
def parse_json_tool_result(payload):
    parsed = json.loads(payload)
    return parsed.get("tool")
""",
                encoding="utf-8",
            )
            episode = {
                "episode_id": "parser_fix",
                "iteration": 1,
                "score": 10.0,
                "score_delta": 2.0,
                "diff_from_parent": (
                    "@@\n"
                    "+def parse_json_tool_result(payload):\n"
                    "+    parsed = json.loads(payload)\n"
                    "+    return parsed.get('tool')\n"
                ),
            }

            with (
                patch("memory.encoding.episode_recorder.EVO_DIR", agents),
                patch("memory.encoding.episode_recorder.AGENTS_DIR", agents),
            ):
                ComponentEvidenceBuilder(out).update(episode)
            evidence = json.loads(
                (out / "component_evidence.json").read_text(encoding="utf-8")
            )

        stored = evidence["components"]["tool_parsing"]["evidence"][0]
        self.assertTrue(stored["component_ast_signature"])
        self.assertIn("tool_parsing", episode["component_ast_signatures"])

    def test_transition_keeps_keyword_fallback_without_ast_signatures(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            evidence = {
                "components": {
                    "tool_parsing": {
                        "evidence": [],
                        "change_families": [
                            {
                                "family_id": "tool_parsing_001",
                                "supporting_episodes": ["historical"],
                                "contradicting_episodes": [],
                                "context_discount": 1.0,
                            }
                        ],
                        "regression_families": [],
                    }
                }
            }
            (out / "component_evidence.json").write_text(
                json.dumps(evidence), encoding="utf-8"
            )

            EvolutionOperators(out).migrate_on_transition(
                "@@\n+parse tool result response json output\n"
            )
            updated = json.loads(
                (out / "component_evidence.json").read_text(encoding="utf-8")
            )

        family = updated["components"]["tool_parsing"]["change_families"][0]
        self.assertEqual(family["context_discount"], 0.3)
        self.assertEqual(family["compatibility_source"], "keyword_fallback")

    def test_current_evidence_is_fully_compatible_with_current_harness(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            signature = component_ast_signature(
                "def parse_result(raw):\n    return json.loads(raw)\n",
                ["parse", "json"],
            )
            evidence = {
                "components": {
                    "tool_parsing": {
                        "evidence": [
                            {
                                "episode_id": "current",
                                "iteration": 2,
                                "component_ast_signature": signature,
                            }
                        ],
                        "change_families": [
                            {
                                "family_id": "tool_parsing_001",
                                "supporting_episodes": ["current"],
                                "contradicting_episodes": [],
                                "context_discount": 0.3,
                            }
                        ],
                        "regression_families": [],
                    }
                }
            }
            (out / "component_evidence.json").write_text(
                json.dumps(evidence), encoding="utf-8"
            )

            EvolutionOperators(out).migrate_on_transition(
                {
                    "episode_id": "current",
                    "diff_from_parent": "",
                    "component_ast_signatures": {"tool_parsing": signature},
                }
            )
            updated = json.loads(
                (out / "component_evidence.json").read_text(encoding="utf-8")
            )

        family = updated["components"]["tool_parsing"]["change_families"][0]
        self.assertEqual(family["context_discount"], 1.0)
        self.assertEqual(family["compatibility_source"], "ast_structure")


if __name__ == "__main__":
    unittest.main()
