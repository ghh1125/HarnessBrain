"""Dataset loading utilities for HarnessBrain text classification tasks."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

try:
    from rdkit import Chem as _Chem
    from rdkit.Chem import AllChem as _AllChem
    from rdkit.Chem import DataStructs as _DataStructs

    def _mol_fp(smi: str):
        mol = _Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return _AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

    def _snap_to_label_space(pred: str, label_space: list[str]) -> str:
        """Map a predicted SMILES to the most structurally similar label-space entry."""
        best_sim = -1.0
        best_label = pred
        for p_part in pred.split("."):
            fp_p = _mol_fp(p_part.strip())
            if fp_p is None:
                continue
            for label in label_space:
                for g_part in label.split("."):
                    fp_g = _mol_fp(g_part.strip())
                    if fp_g is None:
                        continue
                    sim = _DataStructs.TanimotoSimilarity(fp_p, fp_g)
                    if sim > best_sim:
                        best_sim = sim
                        best_label = label
        return best_label

    _RDKIT_AVAILABLE = True

except ImportError:
    _RDKIT_AVAILABLE = False

    def _snap_to_label_space(pred: str, label_space: list[str]) -> str:
        return pred


# data/ lives at project root (one level above src/)
DATA_DIR = Path(__file__).parent.parent / "data"

TASK_DIRS = {
    "Symptom2Disease": "symptom2disease",
    "LawBench": "lawbench",
    "USPTO": "uspto",
}

ALL_TASKS = sorted(TASK_DIRS)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset file: {path}. Run scripts/download_data.py first."
        )
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_label_space(base: Path) -> list[str]:
    path = base / "label_space.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _format_input(dataset: str, text: str, label_space: list[str]) -> str:
    if dataset == "LawBench":
        labels = "; ".join(label_space)
        return (
            "Predict the criminal charge name(s) for the legal fact pattern. "
            "Choose only labels from this label space, separated by semicolons if multiple apply.\n\n"
            f"Label space: {labels}\n\n{text}"
        )
    if dataset == "USPTO":
        labels = "\n".join(f"- {label}" for label in label_space)
        return (
            "Predict the precursor reactant SMILES. Choose exactly one answer from the label space.\n\n"
            f"Label space:\n{labels}\n\n{text}"
        )
    labels = ", ".join(label_space)
    return (
        "Diagnose the disease from the symptom description. "
        "Choose exactly one disease label from the label space and output only that label.\n\n"
        f"Label space: {labels}\n\nSymptoms: {text}"
    )


def _to_examples(
    rows: list[dict[str, Any]], dataset: str, label_space: list[str]
) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        text = row.get("input", row.get("text", ""))
        label = row.get("target", row.get("label", ""))
        ex = {
            "input": _format_input(dataset, str(text), label_space),
            "target": label,
            "raw_question": str(text),
        }
        if label_space:
            ex["label_space"] = label_space
        examples.append(ex)
    return examples


def _sample(rows: list[dict[str, Any]], n: int | None, seed: int) -> list[dict[str, Any]]:
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    if n is None or n >= len(rows):
        return rows
    return rows[:n]


def _normalize_label(label: str, dataset: str) -> str:
    label = label.strip()
    if dataset == "USPTO":
        return label
    label = re.sub(r"\([^)]*\)", "", label)
    label = re.sub(r"\s+", " ", label)
    return label.strip().lower()


def _labels(value: Any, dataset: str) -> set[str]:
    if isinstance(value, list):
        return {_normalize_label(str(v), dataset) for v in value if str(v).strip()}
    text = str(value)
    if text.startswith("罪名:"):
        text = text.split(":", 1)[1]
    text = re.sub(r"^[\[\s]*(?:罪名|答案)[:：]?", "", text)
    text = text.split("<eoa>", 1)[0]
    text = text.strip("[] \n\t")
    parts = [p.strip() for p in re.split(r"[;；]", text)]
    return {_normalize_label(p, dataset) for p in parts if p}


def _make_evaluator(dataset: str, label_space: list[str] | None = None):
    def exact_or_multilabel(prediction: str, target: Any, **_: Any):
        gold = _labels(target, dataset)
        pred = _labels(prediction, dataset)
        if dataset == "LawBench":
            tp = len(gold & pred)
            fp = len(pred - gold)
            fn = len(gold - pred)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            return {
                "was_correct": gold == pred,
                "metrics": {"tp": tp, "fp": fp, "fn": fn, "f1": f1},
            }
        if dataset == "USPTO" and label_space and _RDKIT_AVAILABLE:
            if not gold or not pred:
                return False
            snapped = {_snap_to_label_space(p, label_space) for p in pred}
            return bool(snapped & gold)
        return bool(gold and pred and gold == pred)

    return exact_or_multilabel


def load_dataset_splits(
    dataset: str,
    num_train: int | None = None,
    num_test: int | None = None,
    shuffle_seed: int = 42,
):
    if dataset not in TASK_DIRS:
        raise ValueError(f"Unknown dataset: {dataset}. Options: {ALL_TASKS}")
    base = DATA_DIR / TASK_DIRS[dataset]
    label_space = _read_label_space(base)
    train = _sample(
        _to_examples(_read_jsonl(base / "train_stream.jsonl"), dataset, label_space),
        num_train,
        shuffle_seed,
    )
    test = _sample(
        _to_examples(_read_jsonl(base / "test_set.jsonl"), dataset, label_space),
        num_test,
        shuffle_seed + 2,
    )
    return train, test, _make_evaluator(dataset, label_space)


def load_dataset_splits_3way(
    dataset: str,
    num_train: int | None = None,
    num_val: int | None = None,
    num_test: int | None = None,
    shuffle_seed: int = 42,
):
    if dataset not in TASK_DIRS:
        raise ValueError(f"Unknown dataset: {dataset}. Options: {ALL_TASKS}")
    base = DATA_DIR / TASK_DIRS[dataset]
    label_space = _read_label_space(base)
    train = _sample(
        _to_examples(_read_jsonl(base / "train_stream.jsonl"), dataset, label_space),
        num_train,
        shuffle_seed,
    )
    val = _sample(
        _to_examples(_read_jsonl(base / "search_set.jsonl"), dataset, label_space),
        num_val,
        shuffle_seed + 1,
    )
    test = _sample(
        _to_examples(_read_jsonl(base / "test_set.jsonl"), dataset, label_space),
        num_test,
        shuffle_seed + 2,
    )
    return train, val, test, _make_evaluator(dataset, label_space)
