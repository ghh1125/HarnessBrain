import os

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

COMPONENT_MEMORY_DIR = Path(os.environ.get("COMPONENT_MEMORY_DIR", str(Path(__file__).parent.parent.parent.parent / "workspace" / "component_memory")))
ADVANTAGE_FILE = COMPONENT_MEMORY_DIR / "guidance_advantage.json"
EVOLUTION_LOG = COMPONENT_MEMORY_DIR / "evolution_log.jsonl"


class GuidanceTracker:
    def __init__(
        self,
        advantage_path: Optional[Path] = None,
        log_path: Optional[Path] = None,
    ):
        self.advantage_path = advantage_path or ADVANTAGE_FILE
        self.log_path = log_path or EVOLUTION_LOG
        self.advantage_path.parent.mkdir(parents=True, exist_ok=True)



    def load_states(self) -> dict:
        if self.advantage_path.exists():
            try:
                data = json.loads(self.advantage_path.read_text(encoding="utf-8"))
                return data.get("guidance_states", {})
            except Exception:
                pass
        return {}

    def save_states(self, states: dict) -> None:
        data = {
            "last_updated": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "guidance_states": states,
        }
        self.advantage_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _log(self, entry: dict) -> None:
        entry.setdefault("timestamp", datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")



    def make_guidance_id(self, component: str, direction: str) -> str:
        short_hash = hashlib.md5(direction.encode()).hexdigest()[:6]
        return f"g_{component}_{short_hash}"



    def register_guidance(self, guidance_list: list) -> None:
        states = self.load_states()
        changed = False
        for item in guidance_list:
            comp = item.get("component", "unknown")
            direction = item.get("direction", "")
            if not direction:
                continue
            gid = self.make_guidance_id(comp, direction)
            if gid not in states:
                states[gid] = {
                    "component": comp,
                    "direction": direction,
                    "advantage_score": 0.0,
                    "followed_count": 0,
                    "followed_success": 0,
                    "followed_failure": 0,
                    "audit_failed_count": 0,
                    "last_followed_episode": None,
                    "status": "active",
                }
                changed = True
        if changed:
            self.save_states(states)



    def match_guidance(
        self, proposal_plan: dict, guidance_list: list
    ) -> Optional[str]:
        target_comp = proposal_plan.get("target_component", "")
        if not target_comp:
            return None
        for item in guidance_list:
            if item.get("component", "") == target_comp:
                direction = item.get("direction", "")
                return self.make_guidance_id(target_comp, direction)
        return None



    def update_advantage(
        self, guidance_id: str, episode: dict, audit_passed: bool
    ) -> None:
        states = self.load_states()
        if guidance_id not in states:
            return
        state = states[guidance_id]
        score_delta = episode.get("score_delta", 0)

        if not audit_passed:
            advantage_delta = -1
            state["audit_failed_count"] = state.get("audit_failed_count", 0) + 1
        elif score_delta > 0:
            advantage_delta = 1
            state["followed_success"] = state.get("followed_success", 0) + 1
            state["followed_count"] = state.get("followed_count", 0) + 1
            state["last_followed_episode"] = episode.get("episode_id", "")
        else:
            advantage_delta = -1
            state["followed_failure"] = state.get("followed_failure", 0) + 1
            state["followed_count"] = state.get("followed_count", 0) + 1
            state["last_followed_episode"] = episode.get("episode_id", "")

        new_score = round(state["advantage_score"] + advantage_delta, 2)
        state["advantage_score"] = new_score


        if new_score > 3:
            new_status = "confirmed"
        elif new_score < -4:
            new_status = "removed"
        elif new_score < -2:
            new_status = "degraded"
        else:
            new_status = "active"
        state["status"] = new_status

        states[guidance_id] = state
        self.save_states(states)

        self._log({
            "operation": "UpdateAdvantage",
            "guidance_id": guidance_id,
            "episode_id": episode.get("episode_id", ""),
            "audit_passed": audit_passed,
            "score_delta": score_delta,
            "advantage_delta": advantage_delta,
            "new_advantage_score": new_score,
            "new_status": new_status,
        })



    def prune_guidance(self) -> int:
        states = self.load_states()
        pruned_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        pruned = 0
        for gid, state in states.items():
            if state.get("status") == "removed" and "pruned_at" not in state:
                state["pruned_at"] = pruned_at
                self._log({
                    "operation": "PruneGuidance",
                    "guidance_id": gid,
                    "reason": f"advantage_score={state['advantage_score']}，持续无效",
                    "followed_count": state.get("followed_count", 0),
                    "followed_success": state.get("followed_success", 0),
                    "followed_failure": state.get("followed_failure", 0),
                    "audit_failed_count": state.get("audit_failed_count", 0),
                })
                pruned += 1
        if pruned:
            self.save_states(states)
        return pruned



    def get_advantage_summary(self) -> dict:
        states = self.load_states()
        if not states:
            return {
                "total_guidance": 0,
                "active": 0,
                "confirmed": 0,
                "degraded": 0,
                "removed": 0,
                "avg_advantage_score": 0.0,
                "most_effective": None,
                "least_effective": None,
            }

        status_counts: dict = {"active": 0, "confirmed": 0, "degraded": 0, "removed": 0}
        score_pairs = []
        for gid, state in states.items():
            status = state.get("status", "active")
            status_counts[status] = status_counts.get(status, 0) + 1
            score_pairs.append((gid, state.get("advantage_score", 0.0)))

        avg_score = round(sum(s for _, s in score_pairs) / len(score_pairs), 2)
        best = max(score_pairs, key=lambda x: x[1])
        worst = min(score_pairs, key=lambda x: x[1])

        return {
            "total_guidance": len(states),
            "active": status_counts.get("active", 0),
            "confirmed": status_counts.get("confirmed", 0),
            "degraded": status_counts.get("degraded", 0),
            "removed": status_counts.get("removed", 0),
            "avg_advantage_score": avg_score,
            "most_effective": f"{best[0]} (score={best[1]})",
            "least_effective": f"{worst[0]} (score={worst[1]})",
        }



    def merge_into_guidance(self, guidance_json: dict) -> dict:
        states = self.load_states()

        def _enrich_and_filter(items: list) -> list:
            result = []
            for item in items:
                comp = item.get("component", "unknown")
                direction = item.get("direction", "")
                gid = self.make_guidance_id(comp, direction)
                state = states.get(gid, {})
                status = state.get("status", "active")
                if status == "removed" and "pruned_at" in state:
                    continue
                item["guidance_id"] = gid
                item["advantage_score"] = state.get("advantage_score", 0.0)
                item["advantage_status"] = status
                item["followed_count"] = state.get("followed_count", 0)
                result.append(item)
            return result

        guidance_json["high_priority"] = _enrich_and_filter(
            guidance_json.get("high_priority", [])
        )
        guidance_json["avoid"] = _enrich_and_filter(guidance_json.get("avoid", []))
        return guidance_json


def _arm_d_guidance_advantage_enabled() -> bool:
    from memory.config_loader import evidence_quality_enabled
    return evidence_quality_enabled()


if __name__ == "__main__":
    tracker = GuidanceTracker()
    summary = tracker.get_advantage_summary()
    print("=== Guidance Advantage Summary ===")
    import json as _json
    print(_json.dumps(summary, indent=2, ensure_ascii=False))
    print()
    states = tracker.load_states()
    print(f"Total guidance states in guidance_advantage.json: {len(states)}")
    for gid, state in list(states.items())[:5]:
        print(f"  {gid}: score={state['advantage_score']}, status={state['status']}, "
              f"followed={state.get('followed_count', 0)}")
