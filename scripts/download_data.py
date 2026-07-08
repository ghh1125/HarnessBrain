
from __future__ import annotations

import argparse
import json
import os
import random
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LAW_URL = (
    "https://gh-proxy.com/https://raw.githubusercontent.com/"
    "open-compass/LawBench/main/data/zero_shot/3-3.json"
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_dataset(out_dir: Path, train: list[dict], search: list[dict], test: list[dict], labels: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "train_stream.jsonl", train)
    _write_jsonl(out_dir / "search_set.jsonl", search)
    _write_jsonl(out_dir / "test_set.jsonl", test)
    (out_dir / "label_space.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _split(rows: list[dict], seed: int, train_n: int | None = None, search_n: int = 100, test_n: int = 100):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    test = rows[:test_n]
    search = rows[test_n : test_n + search_n]
    train = rows[test_n + search_n :] if train_n is None else rows[test_n + search_n : test_n + search_n + train_n]
    return train, search, test


def _law_labels(answer: str) -> list[str]:
    answer = answer.split(":", 1)[-1]
    return [p.strip() for p in re.split(r"[;；]", answer) if p.strip()]


def prepare_symptom2disease(seed: int) -> dict[str, int]:
    ds = load_dataset("gretelai/symptom_to_diagnosis")
    train = [
        {"text": r["input_text"].strip(), "label": r["output_text"].strip()}
        for r in ds["train"]
    ]
    test = [
        {"text": r["input_text"].strip(), "label": r["output_text"].strip()}
        for r in ds["test"]
    ]
    labels = sorted({r["label"] for r in train + test})
    search = random.Random(seed).sample(test, min(100, len(test)))
    _write_dataset(DATA_DIR / "symptom2disease", train, search, test, labels)
    return {"train": len(train), "search": len(search), "test": len(test), "labels": len(labels)}


def prepare_lawbench(seed: int) -> dict[str, int]:
    raw_path = DATA_DIR / "_raw" / "lawbench_3-3.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        local_cache = ROOT / "tmp_lawbench_3-3.json"
        if local_cache.exists():
            raw_path.write_bytes(local_cache.read_bytes())
        else:
            with urllib.request.urlopen(LAW_URL, timeout=60) as resp:
                raw_path.write_bytes(resp.read())
    rows = json.loads(raw_path.read_text(encoding="utf-8"))
    examples = []
    labels = set()
    for row in rows:
        labs = _law_labels(row["answer"])
        labels.update(labs)
        examples.append({"text": row["instruction"] + "\n\n" + row["question"], "label": labs})
    train, search, test = _split(examples, seed, search_n=100, test_n=100)
    label_space = sorted(labels)
    _write_dataset(DATA_DIR / "lawbench", train, search, test, label_space)
    return {"train": len(train), "search": len(search), "test": len(test), "labels": len(label_space)}


def _reactants(rxn_smiles: str) -> str:
    return rxn_smiles.split(">>", 1)[0].strip()


def prepare_uspto(seed: int) -> dict[str, int]:
    try:
        ds = load_dataset("seyonec/USPTO_50k")
        splits = [ds[k] for k in ds]
    except Exception:
        ds = load_dataset("pingzhili/uspto-50k")
        splits = [ds[k] for k in ds]

    rows = []
    for split in splits:
        for row in split:
            if not row.get("keep", True):
                continue
            reactants = _reactants(row["rxn_smiles"])
            rows.append(
                {
                    "text": f"Product SMILES: {row['prod_smiles']}\nPredict the precursor reactant SMILES.",
                    "label": reactants,
                }
            )

    top_labels = [label for label, _ in Counter(r["label"] for r in rows).most_common(180)]
    top = set(top_labels)
    filtered = [r for r in rows if r["label"] in top]
    train, search, test = _split(filtered, seed, search_n=100, test_n=100)
    _write_dataset(DATA_DIR / "uspto", train, search, test, sorted(top_labels))
    return {"train": len(train), "search": len(search), "test": len(test), "labels": len(top_labels)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["all", "Symptom2Disease", "LawBench", "USPTO"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    targets = ["Symptom2Disease", "LawBench", "USPTO"] if args.dataset == "all" else [args.dataset]
    funcs = {
        "Symptom2Disease": prepare_symptom2disease,
        "LawBench": prepare_lawbench,
        "USPTO": prepare_uspto,
    }
    for name in targets:
        info = funcs[name](args.seed)
        print(f"{name}: {info}")


if __name__ == "__main__":
    main()
