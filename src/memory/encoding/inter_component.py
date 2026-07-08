import os
import json
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

COMPONENT_MEMORY_DIR = Path(os.environ.get("COMPONENT_MEMORY_DIR", str(Path(__file__).parent.parent.parent.parent / "workspace" / "component_memory")))


class InterComponentEvidence:
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or COMPONENT_MEMORY_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_path = self.output_dir / "inter_component_evidence.json"



    def _load(self) -> dict:
        if self.evidence_path.exists():
            return json.loads(self.evidence_path.read_text(encoding="utf-8"))
        return self._blank()

    def _blank(self) -> dict:
        return {
            "last_updated": "",
            "co_change_patterns": [],
            "confirmed_synergies": [],
            "confirmed_conflicts": [],
            "stats": {},
        }

    def _save(self, data: dict) -> None:
        data["last_updated"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        clean = json.loads(json.dumps(data))
        self._strip_internal_fields(clean)
        self.evidence_path.write_text(
            json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _strip_internal_fields(self, value) -> None:
        if isinstance(value, dict):
            for key in list(value.keys()):
                if key.startswith("_"):
                    del value[key]
                else:
                    self._strip_internal_fields(value[key])
        elif isinstance(value, list):
            for item in value:
                self._strip_internal_fields(item)

    def _all_patterns(self, data: dict) -> list:
        return (
            data.get("co_change_patterns", [])
            + data.get("confirmed_synergies", [])
            + data.get("confirmed_conflicts", [])
        )

    def _recompute_stats(self, data: dict) -> None:
        data["stats"] = self._compute_stats(self._all_patterns(data))



    def _pattern_key(self, components: list) -> frozenset:
        return frozenset(components)

    def _trend(self, avg_delta: float) -> str:
        if avg_delta > 1.0:
            return "positive"
        if avg_delta < -1.0:
            return "negative"
        return "neutral"

    def _note(self, components: list, count: int, avg_delta: float, trend: str) -> str:
        comps_str = "+".join(sorted(components))
        if trend == "negative":
            return (
                f"{comps_str} co-changed {count} times; "
                f"avg_delta={avg_delta:.1f}%, associated with regression"
            )
        if trend == "positive":
            return (
                f"{comps_str} co-changed {count} times; "
                f"avg_delta={avg_delta:.1f}%, associated with improvement"
            )
        return (
            f"{comps_str} co-changed {count} times; "
            f"avg_delta={avg_delta:.1f}%, neutral trend"
        )



    def update(self, episode: dict, components_changed: list) -> None:
        if len(components_changed) < 2:
            return

        episode_id = episode.get("episode_id", "")
        score_delta = float(episode.get("score_delta", 0))
        key = self._pattern_key(components_changed)
        data = self._load()


        for pat in data["co_change_patterns"]:
            if frozenset(pat["components"]) == key:
                if episode_id not in pat["episodes"]:
                    old_count = pat["count"]
                    old_avg = pat["avg_score_delta"]
                    pat["episodes"].append(episode_id)
                    pat["count"] = len(pat["episodes"])
                    pat["avg_score_delta"] = round(
                        ((old_avg * old_count) + score_delta) / pat["count"], 1
                    )
                    pat["trend"] = self._trend(pat["avg_score_delta"])
                    pat["note"] = self._note(
                        pat["components"], pat["count"], pat["avg_score_delta"], pat["trend"]
                    )
                self._recompute_stats(data)
                self._save(data)
                return


        n = len(data["co_change_patterns"]) + 1
        new_pat = {
            "pattern_id": f"co_{n:03d}",
            "components": sorted(components_changed),
            "episodes": [episode_id],
            "avg_score_delta": round(score_delta, 1),
            "count": 1,
            "attribution": "clear" if len(components_changed) == 1 else "ambiguous",
            "trend": self._trend(score_delta),
            "note": self._note(sorted(components_changed), 1, score_delta, self._trend(score_delta)),
        }
        data["co_change_patterns"].append(new_pat)
        self._recompute_stats(data)
        self._save(data)

    def check_synergies(self) -> None:
        data = self._load()
        remaining = []

        for pat in data["co_change_patterns"]:
            avg = pat["avg_score_delta"]
            count = pat["count"]
            trend = pat["trend"]
            components = pat["components"]
            is_whole_class = len(components) >= 5

            if count >= 3 and avg >= 2.0 and trend == "positive":
                already = any(p["pattern_id"] == pat["pattern_id"]
                              for p in data["confirmed_synergies"])
                if not already:
                    data["confirmed_synergies"].append({
                        "pattern_id": pat["pattern_id"],
                        "components": components,
                        "episodes": pat["episodes"],
                        "avg_score_delta": avg,
                        "count": count,
                        "conflict_type": None,
                        "evidence_strength": "strong" if count >= 5 else "medium",
                        "note": "Confirmed synergy: co-change has repeated positive association.",
                    })
                    continue

            if count >= 3 and avg <= -3.0 and trend == "negative":
                already = any(p["pattern_id"] == pat["pattern_id"]
                              for p in data["confirmed_conflicts"])
                if not already:
                    conflict_type = "whole_class_rewrite" if is_whole_class else "multi_component_conflict"
                    if is_whole_class:
                        interpretation = (
                            "whole-class rewrite is associated with regression; "
                            "attribution ambiguous, not causal evidence for any single component"
                        )
                    else:
                        interpretation = (
                            "component co-change is associated with regression; "
                            "attribution ambiguous, treat as a combination-level warning"
                        )
                    data["confirmed_conflicts"].append({
                        "pattern_id": pat["pattern_id"],
                        "components": components,
                        "episodes": pat["episodes"],
                        "avg_score_delta": avg,
                        "count": count,
                        "conflict_type": conflict_type,
                        "interpretation": interpretation,
                        "evidence_strength": "strong" if count >= 5 else "medium",
                        "note": "Co-change is associated with regression; split into single-component experiments.",
                    })
                    continue

            remaining.append(pat)

        data["co_change_patterns"] = remaining
        self._recompute_stats(data)
        self._save(data)

    def _compute_stats(self, all_patterns: list) -> dict:
        total_multi = sum(p["count"] for p in all_patterns)
        n_ambiguous = sum(
            p["count"] for p in all_patterns if p.get("attribution") == "ambiguous"
        )
        ambiguous_ratio = round(n_ambiguous / max(total_multi, 1), 2)


        pair_counts: dict = {}
        pair_deltas: dict = {}
        for pat in all_patterns:
            comps = pat["components"]
            count = pat["count"]
            avg_d = pat["avg_score_delta"]
            for a, b in combinations(sorted(comps), 2):
                k = (a, b)
                pair_counts[k] = pair_counts.get(k, 0) + count
                pair_deltas.setdefault(k, []).append(avg_d)

        most_common_pair = max(pair_counts, key=pair_counts.get) if pair_counts else []
        most_common_pair_count = pair_counts.get(most_common_pair, 0) if pair_counts else 0
        most_common_pair_avg_delta = round(
            sum(pair_deltas.get(most_common_pair, [0])) / max(len(pair_deltas.get(most_common_pair, [1])), 1), 1
        ) if pair_counts else 0

        return {
            "total_multi_component_episodes": total_multi,
            "ambiguous_ratio": ambiguous_ratio,
            "most_common_pair": list(most_common_pair) if most_common_pair else [],
            "most_common_pair_count": most_common_pair_count,
            "most_common_pair_avg_delta": most_common_pair_avg_delta,
        }

    def get_guidance(self) -> dict:
        data = self._load()


        warnings = []
        for pat in data.get("co_change_patterns", []):
            if pat["count"] >= 3 and pat["trend"] == "negative":
                comps = "+".join(pat["components"])
                warnings.append(
                    f"{comps} co-changed {pat['count']} times; "
                    f"avg_delta={pat['avg_score_delta']:.1f}%, split into separate experiments"
                )
        for pat in data.get("confirmed_conflicts", []):
            comps = "+".join(pat["components"])
            warnings.append(
                f"[confirmed conflict] {comps} co-changed {pat['count']} times; "
                f"avg_delta={pat['avg_score_delta']:.1f}%, avoid simultaneous edits"
            )

        return {
            "confirmed_synergies": data.get("confirmed_synergies", []),
            "confirmed_conflicts": data.get("confirmed_conflicts", []),
            "warning": "\n".join(warnings) if warnings else None,
        }

    def get_stats(self) -> dict:
        data = self._load()
        all_pats = (data.get("co_change_patterns", [])
                    + data.get("confirmed_synergies", [])
                    + data.get("confirmed_conflicts", []))
        return self._compute_stats(all_pats)

    def build_from_episodes(self) -> None:
        from memory.encoding.episode_recorder import EpisodeRecorder
        from memory.encoding.component_evidence import identify_components

        episodes = EpisodeRecorder(self.output_dir).get_all_episodes()


        data = self._blank()
        self._save(data)

        for ep in episodes:
            diff = ep.get("diff_from_parent", "")
            info = identify_components(diff)
            changed = info.get("components_changed", [])
            self.update(ep, changed)

        self.check_synergies()


if __name__ == "__main__":
    from memory.encoding.episode_recorder import EpisodeRecorder
    from memory.encoding.component_evidence import identify_components

    episodes = EpisodeRecorder().get_all_episodes()
    if not episodes:
        print("No episodes found. Run component_evidence.py first.")
        raise SystemExit(1)

    inter = InterComponentEvidence()
    print(f"Rebuilding from {len(episodes)} episodes...")
    inter.build_from_episodes()

    data = inter._load()
    patterns = data.get("co_change_patterns", [])
    synergies = data.get("confirmed_synergies", [])
    conflicts = data.get("confirmed_conflicts", [])
    stats = data.get("stats", {})

    print(f"\n=== Inter-Component Evidence ===")
    print(f"co_change_patterns: {len(patterns)}")
    print(f"confirmed_synergies: {len(synergies)}")
    print(f"confirmed_conflicts: {len(conflicts)}")

    print(f"\n--- Patterns ---")
    for pat in patterns:
        print(f"  [{pat['pattern_id']}] {pat['components']}")
        print(f"    count={pat['count']}  avg_delta={pat['avg_score_delta']}  trend={pat['trend']}")
        print(f"    episodes: {pat['episodes']}")

    if conflicts:
        print(f"\n--- Confirmed Conflicts ---")
        for c in conflicts:
            print(f"  {c['components']}: count={c['count']}, avg_delta={c['avg_score_delta']}")
            print(f"  note: {c['note']}")

    if synergies:
        print(f"\n--- Confirmed Synergies ---")
        for s in synergies:
            print(f"  {s['components']}: count={s['count']}, avg_delta={s['avg_score_delta']}")

    print(f"\n--- Stats ---")
    print(f"  total_multi_component_episodes: {stats.get('total_multi_component_episodes')}")
    print(f"  ambiguous_ratio: {stats.get('ambiguous_ratio')}")
    print(f"  most_common_pair: {stats.get('most_common_pair')}")
    print(f"  most_common_pair_count: {stats.get('most_common_pair_count')}")
    print(f"  most_common_pair_avg_delta: {stats.get('most_common_pair_avg_delta')}")

    guidance = inter.get_guidance()
    if guidance.get("warning"):
        print(f"\n--- Warning ---")
        print(f"  {guidance['warning']}")
