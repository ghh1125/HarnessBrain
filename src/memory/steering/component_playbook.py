import os

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if __name__.startswith("src.memory."):
    sys.modules[__name__.replace("src.", "", 1)] = sys.modules[__name__]
elif __name__.startswith("memory."):
    sys.modules[f"src.{__name__}"] = sys.modules[__name__]

COMPONENT_MEMORY_DIR = Path(os.environ.get("COMPONENT_MEMORY_DIR", str(Path(__file__).parent.parent.parent.parent / "workspace" / "component_memory")))

_TC_COMPONENTS = ["retrieval", "prompt", "parser", "memory_update", "state_management"]
_TB_COMPONENTS = [
    "agent_loop",
    "llm_call",
    "command_execution",
    "prompt_template",
    "error_handling",
    "context_management",
    "state_management",
    "tool_parsing",
]
_ALL_COMPONENTS = list(_TC_COMPONENTS)


def configure(task: str) -> None:
    global _ALL_COMPONENTS
    _ALL_COMPONENTS = list(_TB_COMPONENTS if task == "terminal" else _TC_COMPONENTS)


class ComponentPlaybookBuilder:
    def __init__(
        self,
        evidence_path: Optional[Path] = None,
        clusters_path: Optional[Path] = None,
        episodes_path: Optional[Path] = None,
        playbooks_path: Optional[Path] = None,
    ):
        self.evidence_path = evidence_path or COMPONENT_MEMORY_DIR / "component_evidence.json"
        self.clusters_path = clusters_path or COMPONENT_MEMORY_DIR / "direction_clusters.json"
        self.episodes_path = episodes_path or COMPONENT_MEMORY_DIR / "episodes.jsonl"
        self.playbooks_path = playbooks_path or COMPONENT_MEMORY_DIR / "component_playbooks.json"



    def _load_json(self, path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _load_episode_scores(self) -> dict:
        scores: dict[str, float] = {}
        if not self.episodes_path.exists():
            return scores
        for line in self.episodes_path.read_text(encoding="utf-8").strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                ep = json.loads(line)
                eid = ep.get("episode_id", "")
                if eid:
                    scores[eid] = float(ep.get("score", 0))
            except Exception:
                pass
        return scores

    def _find_current_best(
        self,
        comp_data: dict,
        comp_clusters: dict,
        episode_scores: dict,
    ) -> Optional[dict]:
        best_score: float = -1e9
        best_ep: Optional[str] = None
        best_fam: Optional[dict] = None

        for fam in comp_data.get("change_families", []):
            if fam.get("attribution_type", "clear") != "clear":
                continue
            for ep_id in fam.get("supporting_episodes", []):
                ep_id_str = ep_id if isinstance(ep_id, str) else str(ep_id)
                score = episode_scores.get(ep_id_str)
                if score is not None and score > best_score:
                    best_score = score
                    best_ep = ep_id_str
                    best_fam = fam

        if best_ep is None or best_fam is None:
            return None


        from memory.encoding.direction_cluster import DirectionClusterManager
        mgr = DirectionClusterManager(
            evidence_path=self.evidence_path,
            clusters_path=self.clusters_path,
        )
        comp_name = comp_data.get("_component_name", "")
        cluster_id = mgr.assign_cluster(comp_name, best_fam)

        return {
            "technique": best_fam.get("description", ""),
            "cluster": cluster_id,
            "score": best_score,
            "episode_id": best_ep,
        }

    def build_playbook(self, component: str) -> dict:
        evidence = self._load_json(self.evidence_path)
        clusters_data = self._load_json(self.clusters_path)
        episode_scores = self._load_episode_scores()

        comp_data = evidence.get("components", {}).get(component, {})
        comp_data["_component_name"] = component
        comp_clusters = clusters_data.get("components", {}).get(component, {})

        current_best = self._find_current_best(comp_data, comp_clusters, episode_scores)

        promising = []
        avoid = []
        unexplored = []

        for cid, cdata in comp_clusters.items():
            verdict = cdata.get("cluster_verdict", "exploring")
            avg_delta = cdata.get("cluster_avg_delta", 0.0)

            if verdict in ("confirmed", "promising"):
                ev_level = "strong_positive" if verdict == "confirmed" else "weak_positive"
                promising.append({
                    "cluster": cid,
                    "description": cdata.get("description", ""),
                    "cluster_verdict": verdict,
                    "cluster_avg_delta": avg_delta,
                    "evidence_level": ev_level,
                })
            elif verdict == "avoid":
                avoid.append({
                    "cluster": cid,
                    "description": cdata.get("description", ""),
                    "cluster_verdict": "avoid",
                    "cluster_avg_delta": avg_delta,
                    "evidence_level": "strong_negative",
                })
            else:

                techniques = cdata.get("techniques", [])
                if not techniques:
                    ev_level = "unverified"
                else:
                    has_positive = any(
                        t.get("verdict") == "effective" for t in techniques
                    )
                    ev_level = "hypothesis" if has_positive else "unverified"
                unexplored.append({
                    "cluster": cid,
                    "description": cdata.get("description", ""),
                    "cluster_verdict": verdict,
                    "cluster_avg_delta": avg_delta,
                    "evidence_level": ev_level,
                })


        promising.sort(key=lambda x: x["cluster_avg_delta"], reverse=True)
        avoid.sort(key=lambda x: x["cluster_avg_delta"])
        unexplored.sort(key=lambda x: x["cluster_avg_delta"], reverse=True)


        n_promising = len(promising)
        n_avoid = len(avoid)
        n_unexplored = len(unexplored)
        if n_promising > 0:
            search_status = (
                f"{n_promising} promising direction(s) found; "
                f"{n_avoid} to avoid; {n_unexplored} unexplored."
            )
        else:
            search_status = (
                f"No clear positive evidence yet; "
                f"{n_unexplored} unexplored direction(s) remain."
            )

        return {
            "component": component,
            "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "current_best": current_best,
            "promising": promising,
            "avoid": avoid,
            "unexplored": unexplored,
            "search_status": search_status,
        }

    def build_all_playbooks(self) -> None:
        playbooks: dict = {}
        for comp in _ALL_COMPONENTS:
            playbooks[comp] = self.build_playbook(comp)

        output = {
            "last_updated": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "playbooks": playbooks,
        }
        self.playbooks_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[component_playbook] Written {self.playbooks_path}")

    def get_compact_summary(self) -> str:
        data = self._load_json(self.playbooks_path)
        playbooks = data.get("playbooks", {})

        lines: list[str] = []
        for comp in _ALL_COMPONENTS:
            pb = playbooks.get(comp, {})
            if not pb:
                continue

            promising_list = pb.get("promising", [])
            avoid_list = pb.get("avoid", [])
            unexplored_list = pb.get("unexplored", [])


            if promising_list:
                p = promising_list[0]
                delta_str = f"{p['cluster_avg_delta']:+.1f}%"
                promising_str = f"{p['cluster']} ({delta_str}) [{p['evidence_level']}]"
            else:
                promising_str = "(none)"


            if avoid_list:
                a = avoid_list[0]
                delta_str = f"{a['cluster_avg_delta']:+.1f}%"
                avoid_str = f"{a['cluster']} ({delta_str}) [{a['evidence_level']}]"
            else:
                avoid_str = "(none)"


            unexp_parts = [
                f"{u['cluster']} [{u['evidence_level']}]"
                for u in unexplored_list[:2]
            ]
            unexplored_str = ", ".join(unexp_parts) if unexp_parts else "(none)"

            lines.append(f"{comp}:")
            lines.append(f"  promising: {promising_str}")
            lines.append(f"  avoid: {avoid_str}")
            lines.append(f"  unexplored: {unexplored_str}")

        summary = "\n".join(lines)


        if len(summary) > 500:
            summary = summary[:497] + "..."

        return summary


if __name__ == "__main__":
    from memory.encoding.direction_cluster import DirectionClusterManager


    cluster_mgr = DirectionClusterManager()
    cluster_mgr.update_all_clusters()

    builder = ComponentPlaybookBuilder()
    builder.build_all_playbooks()
    print("\n=== Compact Summary ===")
    summary = builder.get_compact_summary()
    print(summary)
    print(f"\nSummary length: {len(summary)} chars")


    data = json.loads(builder.playbooks_path.read_text(encoding="utf-8"))
    ret = data["playbooks"]["retrieval"]
    print(f"\nretrieval current_best: {ret.get('current_best')}")
    print(f"retrieval promising: {len(ret.get('promising', []))} entries")
    print(f"retrieval avoid: {len(ret.get('avoid', []))} entries")
    print(f"retrieval unexplored: {len(ret.get('unexplored', []))} entries")
    print(f"search_status: {ret.get('search_status')}")
