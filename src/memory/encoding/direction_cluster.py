import os

import json
from datetime import datetime
from pathlib import Path

COMPONENT_MEMORY_DIR = Path(os.environ.get("COMPONENT_MEMORY_DIR", str(Path(__file__).parent.parent.parent.parent / "workspace" / "component_memory")))

DIRECTION_CLUSTERS: dict = {
    "retrieval": {
        "lexical_noise_reduction": {
            "description": "Reduce lexical noise to improve retrieval precision",
            "keywords": [
                "stopword", "functional_word",
                "tfidf", "tf-idf", "idf", "noise",
                "filter", "weight", "lexical",
            ],
        },
        "similarity_balancing": {
            "description": "Balance retrieval results across categories",
            "keywords": [
                "jaccard", "balance", "uniform",
                "diversity", "equal", "distribution",
            ],
        },
        "semantic_retrieval": {
            "description": "Semantic-level retrieval approaches",
            "keywords": [
                "embedding", "semantic", "cosine",
                "dense", "bert", "encode", "vector",
            ],
        },
        "structural_retrieval": {
            "description": "Structured retrieval strategies",
            "keywords": [
                "bm25", "structured", "index",
                "domain", "dictionary", "corpus",
            ],
        },
    },
    "prompt": {
        "label_space_exposure": {
            "description": "Expose label space inside the prompt",
            "keywords": [
                "label_list", "label_space",
                "primer", "all_labels", "category",
            ],
        },
        "contrastive_examples": {
            "description": "Contrastive example presentation",
            "keywords": [
                "contrastive", "contrast",
                "challenger", "negative", "different",
            ],
        },
        "prior_statistics": {
            "description": "Incorporate statistical priors into the prompt",
            "keywords": [
                "prior", "frequency", "distribution",
                "statistics", "count", "ratio",
            ],
        },
        "two_stage_verification": {
            "description": "Two-stage draft-and-verify prompt",
            "keywords": [
                "two_stage", "draft", "verify",
                "verification", "confirm", "revise",
            ],
        },
    },
    "parser": {
        "strict_format": {
            "description": "Strict format parsing",
            "keywords": [
                "json", "strict", "exact",
                "format", "structured", "schema",
            ],
        },
        "flexible_extraction": {
            "description": "Flexible text extraction",
            "keywords": [
                "regex", "extract", "flexible",
                "fuzzy", "normalize", "strip",
            ],
        },
    },
    "memory_update": {
        "fifo_management": {
            "description": "First-in-first-out memory management",
            "keywords": [
                "fifo", "deque", "queue",
                "recent", "latest", "window",
            ],
        },
        "selective_retention": {
            "description": "Selective retention of important samples",
            "keywords": [
                "select", "retain", "important",
                "score", "quality", "filter",
            ],
        },
        "error_memory": {
            "description": "Special memory for error cases",
            "keywords": [
                "error", "wrong", "mistake",
                "failure", "contrastive", "hard",
            ],
        },
    },
    "state_management": {
        "checkpoint": {
            "description": "State checkpoint management",
            "keywords": [
                "checkpoint", "save", "restore",
                "persist", "serialize", "snapshot",
            ],
        },
    },
}


class DirectionClusterManager:
    def __init__(
        self,
        evidence_path: Path | None = None,
        clusters_path: Path | None = None,
    ):
        self.evidence_path = evidence_path or COMPONENT_MEMORY_DIR / "component_evidence.json"
        self.clusters_path = clusters_path or COMPONENT_MEMORY_DIR / "direction_clusters.json"



    def _family_text(self, family: dict) -> str:
        return " ".join([
            family.get("description", ""),
            family.get("change_summary", ""),
        ]).lower()

    def _derive_evidence_class(self, family: dict) -> str:
        if family.get("attribution_type", "clear") == "clear":
            return "actionable"
        return "diagnostic"



    def assign_cluster(self, component: str, family: dict) -> str:
        clusters = DIRECTION_CLUSTERS.get(component, {})
        if not clusters:
            return "other"

        text = self._family_text(family)
        best_cluster = "other"
        best_count = 0

        for cluster_id, cluster_def in clusters.items():
            count = sum(1 for kw in cluster_def["keywords"] if kw.lower() in text)
            if count > best_count:
                best_count = count
                best_cluster = cluster_id

        return best_cluster

    def build_clusters(self, component: str, all_families: list) -> dict:
        cluster_defs = DIRECTION_CLUSTERS.get(component, {})


        buckets: dict[str, dict] = {}
        for cid, cdef in cluster_defs.items():
            buckets[cid] = {
                "description": cdef["description"],
                "techniques": [],
                "ambiguous_warnings": [],
                "effective_count": 0,
                "regression_count": 0,
                "delta_sum": 0.0,
                "delta_count": 0,
                "_seen_keywords": set(),
            }

        other_bucket: dict = {
            "description": "Uncategorised families",
            "techniques": [],
            "ambiguous_warnings": [],
            "effective_count": 0,
            "regression_count": 0,
            "delta_sum": 0.0,
            "delta_count": 0,
            "_seen_keywords": set(),
        }

        for fam in all_families:
            is_regression = fam.get("status") == "regression_family"
            cid = self.assign_cluster(component, fam)
            target = buckets.get(cid, other_bucket)

            if is_regression:

                target["regression_count"] += 1
                target["delta_sum"] += float(fam.get("avg_score_delta", 0))
                target["delta_count"] += 1
                continue

            evidence_class = self._derive_evidence_class(fam)

            if evidence_class == "diagnostic":
                target["ambiguous_warnings"].append({
                    "family_id": fam.get("family_id", ""),
                    "description": fam.get("description", ""),
                    "evidence_class": "diagnostic",
                    "note": "Multi-component edit; cannot attribute to single component",
                })
            else:

                for kw in cluster_defs.get(cid, {}).get("keywords", []):
                    if kw.lower() in self._family_text(fam):
                        target["_seen_keywords"].add(kw)

                slim = {
                    "family_id": fam.get("family_id", ""),
                    "description": fam.get("description", ""),
                    "avg_score_delta": float(fam.get("avg_score_delta", 0)),
                    "verdict": fam.get("verdict", "inconclusive"),
                    "evidence_strength": fam.get("evidence_strength", ""),
                    "evidence_class": "actionable",
                }
                target["techniques"].append(slim)

                verdict = fam.get("verdict", "inconclusive")
                status = fam.get("status", "")
                if verdict == "effective" or status == "confirmed_guidance":
                    target["effective_count"] += 1
                    target["delta_sum"] += float(fam.get("avg_score_delta", 0))
                    target["delta_count"] += 1
                else:
                    target["delta_sum"] += float(fam.get("avg_score_delta", 0))
                    target["delta_count"] += 1


        result: dict = {}
        for cid, bucket in buckets.items():
            eff = bucket["effective_count"]
            reg = bucket["regression_count"]
            if eff >= 2 and reg == 0:
                cluster_verdict = "confirmed"
            elif eff >= 1 and reg == 0:
                cluster_verdict = "promising"
            elif reg >= 2 and eff == 0:
                cluster_verdict = "avoid"
            else:
                cluster_verdict = "exploring"

            n = bucket["delta_count"]
            cluster_avg_delta = round(bucket["delta_sum"] / n, 2) if n > 0 else 0.0


            all_kws = cluster_defs[cid]["keywords"]
            unexplored_kws = [kw for kw in all_kws if kw not in bucket["_seen_keywords"]]

            result[cid] = {
                "description": bucket["description"],
                "techniques": bucket["techniques"],
                "ambiguous_warnings": bucket["ambiguous_warnings"],
                "cluster_verdict": cluster_verdict,
                "cluster_avg_delta": cluster_avg_delta,
                "effective_count": eff,
                "regression_count": reg,
                "ambiguous_count": len(bucket["ambiguous_warnings"]),
                "unexplored_techniques": unexplored_kws[:4],
            }

        return result

    def update_all_clusters(self) -> None:
        if not self.evidence_path.exists():
            print("[direction_cluster] component_evidence.json not found; skipping.")
            return

        data = json.loads(self.evidence_path.read_text(encoding="utf-8"))
        components_out: dict = {}

        for comp_name, comp in data.get("components", {}).items():
            all_families = (
                comp.get("change_families", [])
                + comp.get("regression_families", [])
            )
            components_out[comp_name] = self.build_clusters(comp_name, all_families)

        output = {
            "last_updated": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "components": components_out,
        }
        self.clusters_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[direction_cluster] Written {self.clusters_path}")


if __name__ == "__main__":
    mgr = DirectionClusterManager()
    mgr.update_all_clusters()

    data = json.loads(mgr.clusters_path.read_text(encoding="utf-8"))
    retrieval_clusters = data.get("components", {}).get("retrieval", {})
    print(f"\nretrieval: {len(retrieval_clusters)} clusters")
    for cid, cdata in retrieval_clusters.items():
        n_tech = len(cdata.get("techniques", []))
        n_warn = len(cdata.get("ambiguous_warnings", []))
        print(
            f"  {cid}: verdict={cdata['cluster_verdict']}"
            f"  effective={cdata['effective_count']}"
            f"  regression={cdata['regression_count']}"
            f"  techniques={n_tech}"
            f"  ambiguous_warnings={n_warn}"
            f"  avg_delta={cdata['cluster_avg_delta']:+.2f}"
        )
