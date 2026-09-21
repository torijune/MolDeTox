import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

_QA_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _QA_DIR.parent
# QA sets are written inside the repository, one directory per build.
_QA_SETS_ROOT = _QA_DIR / "QA"
_DEFAULT_QA_SET_BUNDLE_NAME = "MolDeTox_QA"

# Ensure MolDetox_QA_template imports resolve beside this folder
if str(_QA_DIR) not in sys.path:
    sys.path.insert(0, str(_QA_DIR))


def _count_dot_separated_fragments(dot_separated: str) -> int:
    """Dot-separated SAFE token count."""
    s = (dot_separated or "").strip()
    if not s:
        return 0
    return len([p.strip() for p in s.split(".") if p.strip()])


def classify_step_task1(only_toxic_safe_fragments: str) -> str:
    """Task 1 stepping: multi when more than one toxic fragment token."""
    return "single_step" if _count_dot_separated_fragments(only_toxic_safe_fragments) == 1 else "multi_step"


def classify_step_task2_or_task3(
    only_toxic_safe_fragments: str,
    only_nontoxic_safe_fragments: str,
) -> str:
    """Task 2/3 stepping based on toxic & nontoxic fragment token counts."""
    n_t = _count_dot_separated_fragments(only_toxic_safe_fragments)
    n_nt = _count_dot_separated_fragments(only_nontoxic_safe_fragments)
    return "single_step" if n_t == 1 and n_nt == 1 else "multi_step"


from MolDetox_QA_template import (
    task1_toxic_fragment_identification,
    task2_nontoxic_fragment_generation,
    task3_nontoxic_smiles_generation,
    task3_nontoxic_safe_generation,
    task3_stepwise_cot_nontoxic_safe_generation,
)

# Where spliter.py writes train.csv and test.csv by default.
_DEFAULT_SPLIT_DIR = _QA_DIR / "splits" / "default"
_DEFAULT_TRAIN_CSV = _DEFAULT_SPLIT_DIR / "train.csv"
_DEFAULT_TEST_CSV = _DEFAULT_SPLIT_DIR / "test.csv"

# Data paths (configured at runtime in main())
DATA_TASK1 = _DEFAULT_TEST_CSV  # task1_toxic_fragment_identification paths refresh at runtime via _configure_paths
DATA_TASK2 = _DEFAULT_TEST_CSV         # task2_nontoxic_fragment_generation
DATA_TASK3 = _DEFAULT_TEST_CSV         # task3_nontoxic_smiles_generation
CURRENT_SPLIT = "test"
# Locked representation subdirectory name
CURRENT_MOLECULE_REPR = "both_repre"
# Shuffle seed when non-None before writing JSONL (defaults to chronological order).
BUILD_QA_SHUFFLE_SEED: Optional[int] = None
# Append endpoint narratives to prompts (default True).
INCLUDE_ENDPOINT_DESCRIPTION: bool = True

# QA output root: MolDeTox/QA/<bundle>/
QA_OUT_ROOT = _QA_SETS_ROOT / _DEFAULT_QA_SET_BUNDLE_NAME

# One file per task and fragment setting, named as in the released dataset:
#   <split>/<split>_<task>_<single|multi>.jsonl
SPLIT_DIR = QA_OUT_ROOT / "test"

TASK_SLUGS = {
    "task1": "task1",
    "task2": "task2",
    "task3_smiles": "task3_smiles_gen",
    "task3_safe": "task3_safe_gen",
    "task3_cot": "task3_cot_safe_gen",
}


def qa_path(task: str, step: str) -> Path:
    """Output file for one task and fragment setting. step is 'single' or 'multi'."""
    return SPLIT_DIR / f"{CURRENT_SPLIT}_{TASK_SLUGS[task]}_{step}.jsonl"

# Toxicity cliff exports may alias decoded columns via toxic_smiles / nontoxic_smiles entries
REQUIRED_COLUMNS_TASK_MIN = [
    "dataset_name",
    "endpoint",
    "toxic_safe",
    "nontoxic_safe",
    "only_toxic_safe_fragments",
    "only_nontoxic_safe_fragments",
]


def _str_or_empty(val) -> str:
    if val is None:
        return ""
    # Robust handling for pandas-style NaNs
    try:
        if isinstance(val, float) and val != val:  # NaN
            return ""
    except Exception:
        pass
    if pd is not None:
        try:
            if pd.isna(val):
                return ""
        except Exception:
            pass
    return str(val).strip()


def _has_dataset_or_endpoint(row: dict) -> bool:
    """Require at least dataset_name or endpoint text."""
    dataset_name = _str_or_empty(row.get("dataset_name", ""))
    endpoint = _str_or_empty(row.get("endpoint", ""))
    return bool(dataset_name or endpoint)


def _merge_toxicity_cliff_row_aliases(row: dict) -> None:
    """Normalize decoded SMILES aliases on cliff-style CSV rows."""
    if not _str_or_empty(row.get("toxic_safe_decoded_smiles", "")):
        row["toxic_safe_decoded_smiles"] = _str_or_empty(row.get("toxic_smiles", ""))
    if not _str_or_empty(row.get("nontoxic_safe_decoded_smiles", "")):
        row["nontoxic_safe_decoded_smiles"] = _str_or_empty(row.get("nontoxic_smiles", ""))
    row.setdefault("common_safe_fragments", "")


def _validate_pair_csv_headers(fieldnames: list[str], path: Path) -> None:
    missing = [c for c in REQUIRED_COLUMNS_TASK_MIN if c not in fieldnames]
    if missing:
        raise ValueError(f"Missing column(s) in {path}: {missing}")
    if "toxic_safe_decoded_smiles" not in fieldnames and "toxic_smiles" not in fieldnames:
        raise ValueError(
            f"{path}: need column toxic_safe_decoded_smiles or toxic_smiles (ToxicityCliff_pairing output)"
        )
    if "nontoxic_safe_decoded_smiles" not in fieldnames and "nontoxic_smiles" not in fieldnames:
        raise ValueError(
            f"{path}: need column nontoxic_safe_decoded_smiles or nontoxic_smiles"
        )


def _iter_csv_rows(path: Path, _required_cols_unused: list[str] | None = None) -> tuple[list[str], list[tuple[int, dict]]]:
    """Read cliff/merged-compatible CSV pairs with enforced headers."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        _validate_pair_csv_headers(fieldnames, path)
        rows: list[tuple[int, dict]] = []
        for idx, row in enumerate(reader):
            _merge_toxicity_cliff_row_aliases(row)
            rows.append((idx, row))
        return fieldnames, rows

def _shuffle_and_reid(records: list[dict], seed: Optional[int]) -> list[dict]:
    """Shuffle records and reassign id to 0..n-1. Preserves dataset_name, endpoint, source_index."""
    if not records:
        return records
    if seed is not None:
        shuffled = list(records)
        random.Random(seed).shuffle(shuffled)
        for i, r in enumerate(shuffled):
            r = dict(r)
            r["id"] = i
            shuffled[i] = r
        return shuffled
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_task2():
    """Task 2: nontoxic_fragment_generation"""
    if not DATA_TASK2.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_TASK2}")
    _, rows = _iter_csv_rows(DATA_TASK2)

    records_single: list[dict] = []
    records_multi: list[dict] = []
    skipped_missing_dataset_endpoint = 0
    for idx, row in rows:
        if not _has_dataset_or_endpoint(row):
            skipped_missing_dataset_endpoint += 1
            continue
        only_toxic = _str_or_empty(row["only_toxic_safe_fragments"])
        only_nontoxic = _str_or_empty(row["only_nontoxic_safe_fragments"])
        step = classify_step_task2_or_task3(only_toxic, only_nontoxic)

        question, answer = task2_nontoxic_fragment_generation(
            toxic_safe=_str_or_empty(row["toxic_safe"]),
            only_toxic_safe_fragments=only_toxic,
            only_nontoxic_safe_fragments=only_nontoxic,
            dataset_name=_str_or_empty(row["dataset_name"]) or None,
            endpoint=_str_or_empty(row["endpoint"]) or None,
            toxic_safe_decoded_smiles=_str_or_empty(row.get("toxic_safe_decoded_smiles", "")),
            nontoxic_safe_decoded_smiles=_str_or_empty(row.get("nontoxic_safe_decoded_smiles", "")),
            nontoxic_safe=_str_or_empty(row.get("nontoxic_safe", "")),
            step=step,
            molecule_repr=CURRENT_MOLECULE_REPR,
            include_endpoint_description=INCLUDE_ENDPOINT_DESCRIPTION,
        )
        rec = {
            "id": int(idx),
            "question": question,
            "answer": answer,
            "dataset_name": _str_or_empty(row.get("dataset_name", "")),
            "endpoint": _str_or_empty(row.get("endpoint", "")),
            "source_index": int(idx),
            "common_safe_fragments": _str_or_empty(row.get("common_safe_fragments", "")),
            "nontoxic_safe_decoded_smiles": _str_or_empty(row.get("nontoxic_safe_decoded_smiles", "")),
        }
        (records_multi if step == "multi_step" else records_single).append(rec)

    out_single = qa_path("task2", "single")
    out_multi = qa_path("task2", "multi")
    _write_jsonl(out_single, _shuffle_and_reid(records_single, BUILD_QA_SHUFFLE_SEED))
    _write_jsonl(out_multi, _shuffle_and_reid(records_multi, BUILD_QA_SHUFFLE_SEED))
    print(f"Task 2: single_step={len(records_single)} -> {out_single}")
    print(f"Task 2: multi_step ={len(records_multi)} -> {out_multi}")
    print(f"Task 2: skipped dataset/endpoint missing rows = {skipped_missing_dataset_endpoint}")
    return out_single, out_multi


def build_task1():
    """Task 1: toxic_fragment_identification"""
    if not DATA_TASK1.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_TASK1}")
    _, rows = _iter_csv_rows(DATA_TASK1)

    records_single: list[dict] = []
    records_multi: list[dict] = []
    skipped_missing_dataset_endpoint = 0
    for idx, row in rows:
        if not _has_dataset_or_endpoint(row):
            skipped_missing_dataset_endpoint += 1
            continue
        only_toxic = _str_or_empty(row["only_toxic_safe_fragments"])
        step = classify_step_task1(only_toxic)
        question, answer = task1_toxic_fragment_identification(
            toxic_safe=_str_or_empty(row["toxic_safe"]),
            only_toxic_safe_fragments=only_toxic,
            dataset_name=_str_or_empty(row["dataset_name"]) or None,
            endpoint=_str_or_empty(row["endpoint"]) or None,
            toxic_safe_decoded_smiles=_str_or_empty(row.get("toxic_safe_decoded_smiles", "")),
            step=step,
            molecule_repr=CURRENT_MOLECULE_REPR,
            include_endpoint_description=INCLUDE_ENDPOINT_DESCRIPTION,
        )
        rec = {
            "id": int(idx),
            "question": question,
            "answer": answer,
            "dataset_name": _str_or_empty(row.get("dataset_name", "")),
            "endpoint": _str_or_empty(row.get("endpoint", "")),
            "source_index": int(idx),
        }
        (records_multi if step == "multi_step" else records_single).append(rec)

    out_single = qa_path("task1", "single")
    out_multi = qa_path("task1", "multi")
    _write_jsonl(out_single, _shuffle_and_reid(records_single, BUILD_QA_SHUFFLE_SEED))
    _write_jsonl(out_multi, _shuffle_and_reid(records_multi, BUILD_QA_SHUFFLE_SEED))
    print(f"Task 1: single_step={len(records_single)} -> {out_single}")
    print(f"Task 1: multi_step ={len(records_multi)} -> {out_multi}")
    print(f"Task 1: skipped dataset/endpoint missing rows = {skipped_missing_dataset_endpoint}")
    return out_single, out_multi


def build_task3():
    """Task 3: nontoxic_smiles_generation"""
    if not DATA_TASK3.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_TASK3}")
    _, rows = _iter_csv_rows(DATA_TASK3)

    records_single: list[dict] = []
    records_multi: list[dict] = []
    skipped_missing_dataset_endpoint = 0
    for idx, row in rows:
        if not _has_dataset_or_endpoint(row):
            skipped_missing_dataset_endpoint += 1
            continue
        only_toxic = _str_or_empty(row["only_toxic_safe_fragments"])
        only_nontoxic = _str_or_empty(row["only_nontoxic_safe_fragments"])
        step = classify_step_task2_or_task3(only_toxic, only_nontoxic)

        question, answer = task3_nontoxic_smiles_generation(
            toxic_safe=_str_or_empty(row["toxic_safe"]),
            dataset_name=_str_or_empty(row["dataset_name"]) or None,
            endpoint=_str_or_empty(row["endpoint"]) or None,
            toxic_safe_decoded_smiles=_str_or_empty(row.get("toxic_safe_decoded_smiles", "")),
            nontoxic_safe_decoded_smiles=_str_or_empty(row.get("nontoxic_safe_decoded_smiles", "")),
            step=step,
            molecule_repr=CURRENT_MOLECULE_REPR,
            include_endpoint_description=INCLUDE_ENDPOINT_DESCRIPTION,
        )
        rec = {
            "id": int(idx),
            "question": question,
            "answer": answer,
            "dataset_name": _str_or_empty(row.get("dataset_name", "")),
            "endpoint": _str_or_empty(row.get("endpoint", "")),
            "source_index": int(idx),
        }
        (records_multi if step == "multi_step" else records_single).append(rec)

    out_single = qa_path("task3_smiles", "single")
    out_multi = qa_path("task3_smiles", "multi")
    _write_jsonl(out_single, _shuffle_and_reid(records_single, BUILD_QA_SHUFFLE_SEED))
    _write_jsonl(out_multi, _shuffle_and_reid(records_multi, BUILD_QA_SHUFFLE_SEED))
    print(f"Task 3: single_step={len(records_single)} -> {out_single}")
    print(f"Task 3: multi_step ={len(records_multi)} -> {out_multi}")
    print(f"Task 3: skipped dataset/endpoint missing rows = {skipped_missing_dataset_endpoint}")
    return out_single, out_multi


def build_task3_nontoxic_safe_generation():
    """Task 3: nontoxic_safe_generation"""
    if not DATA_TASK3.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_TASK3}")
    _, rows = _iter_csv_rows(DATA_TASK3)

    records_single: list[dict] = []
    records_multi: list[dict] = []
    skipped_missing_dataset_endpoint = 0
    for idx, row in rows:
        if not _has_dataset_or_endpoint(row):
            skipped_missing_dataset_endpoint += 1
            continue
        only_toxic = _str_or_empty(row["only_toxic_safe_fragments"])
        only_nontoxic = _str_or_empty(row["only_nontoxic_safe_fragments"])
        step = classify_step_task2_or_task3(only_toxic, only_nontoxic)

        question, answer = task3_nontoxic_safe_generation(
            toxic_safe=_str_or_empty(row["toxic_safe"]),
            nontoxic_safe=_str_or_empty(row["nontoxic_safe"]),
            dataset_name=_str_or_empty(row["dataset_name"]) or None,
            endpoint=_str_or_empty(row["endpoint"]) or None,
            toxic_safe_decoded_smiles=_str_or_empty(row.get("toxic_safe_decoded_smiles", "")),
            nontoxic_safe_decoded_smiles=_str_or_empty(row.get("nontoxic_safe_decoded_smiles", "")),
            step=step,
            molecule_repr=CURRENT_MOLECULE_REPR,
            include_endpoint_description=INCLUDE_ENDPOINT_DESCRIPTION,
        )
        rec = {
            "id": int(idx),
            "question": question,
            "answer": answer,
            "dataset_name": _str_or_empty(row.get("dataset_name", "")),
            "endpoint": _str_or_empty(row.get("endpoint", "")),
            "source_index": int(idx),
        }
        (records_multi if step == "multi_step" else records_single).append(rec)

    out_single = qa_path("task3_safe", "single")
    out_multi = qa_path("task3_safe", "multi")
    _write_jsonl(out_single, _shuffle_and_reid(records_single, BUILD_QA_SHUFFLE_SEED))
    _write_jsonl(out_multi, _shuffle_and_reid(records_multi, BUILD_QA_SHUFFLE_SEED))
    print(f"Task 3 nontoxic safe: single_step={len(records_single)} -> {out_single}")
    print(f"Task 3 nontoxic safe: multi_step ={len(records_multi)} -> {out_multi}")
    print(
        "Task 3 nontoxic safe: "
        f"skipped dataset/endpoint missing rows = {skipped_missing_dataset_endpoint}"
    )
    return out_single, out_multi


def build_task3_stepwise_cot_safe_generation():
    """
    Task 3 stepwise CoT emitting full SAFE strings:
      one model message; Step 1/2 mirror SMILES-variant reasoning with SAFE fragments; final JSON answer is SAFE.
    """
    if not DATA_TASK3.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_TASK3}")
    _, rows = _iter_csv_rows(DATA_TASK3)

    records_single: list[dict] = []
    records_multi: list[dict] = []
    skipped_missing_dataset_endpoint = 0
    for idx, row in rows:
        if not _has_dataset_or_endpoint(row):
            skipped_missing_dataset_endpoint += 1
            continue
        only_toxic = _str_or_empty(row["only_toxic_safe_fragments"])
        only_nontoxic = _str_or_empty(row["only_nontoxic_safe_fragments"])
        step = classify_step_task2_or_task3(only_toxic, only_nontoxic)

        question, answer = task3_stepwise_cot_nontoxic_safe_generation(
            toxic_safe=_str_or_empty(row["toxic_safe"]),
            dataset_name=_str_or_empty(row["dataset_name"]) or None,
            endpoint=_str_or_empty(row["endpoint"]) or None,
            toxic_safe_decoded_smiles=_str_or_empty(row.get("toxic_safe_decoded_smiles", "")),
            nontoxic_safe=_str_or_empty(row.get("nontoxic_safe", "")),
            only_toxic_safe_fragments=only_toxic,
            only_nontoxic_safe_fragments=only_nontoxic,
            step=step,
            molecule_repr=CURRENT_MOLECULE_REPR,
            include_endpoint_description=INCLUDE_ENDPOINT_DESCRIPTION,
        )
        rec = {
            "id": int(idx),
            "question": question,
            "answer": answer,
            "dataset_name": _str_or_empty(row.get("dataset_name", "")),
            "endpoint": _str_or_empty(row.get("endpoint", "")),
            "source_index": int(idx),
        }
        (records_multi if step == "multi_step" else records_single).append(rec)

    out_single = qa_path("task3_cot", "single")
    out_multi = qa_path("task3_cot", "multi")
    _write_jsonl(out_single, _shuffle_and_reid(records_single, BUILD_QA_SHUFFLE_SEED))
    _write_jsonl(out_multi, _shuffle_and_reid(records_multi, BUILD_QA_SHUFFLE_SEED))
    print(f"Task 3 stepwise CoT (SAFE): single_step={len(records_single)} -> {out_single}")
    print(f"Task 3 stepwise CoT (SAFE): multi_step ={len(records_multi)} -> {out_multi}")
    print(
        "Task 3 stepwise CoT (SAFE): "
        f"skipped dataset/endpoint missing rows = {skipped_missing_dataset_endpoint}"
    )
    return out_single, out_multi


def _configure_paths(
    split: str,
    input_csv: Path | None,
    molecule_repr: str = "both_repre",
) -> None:
    """
    Configure DATA_* and OUT_DIR_* globals for the desired split and input CSV.

    molecule_repr is accepted only for backwards compatibility - the writer always persists both_repre.
    """
    global DATA_TASK1, DATA_TASK2, DATA_TASK3
    global SPLIT_DIR
    global CURRENT_SPLIT, CURRENT_MOLECULE_REPR

    CURRENT_SPLIT = split
    repr_dir = "both_repre"
    CURRENT_MOLECULE_REPR = repr_dir

    root = QA_OUT_ROOT
    if split == "train":
        data_path = input_csv or _DEFAULT_TRAIN_CSV
        split_dir = root / "train"
    else:
        data_path = input_csv or _DEFAULT_TEST_CSV
        split_dir = root / "test"

    DATA_TASK1 = data_path   # toxic_fragment_identification
    DATA_TASK2 = data_path   # nontoxic_fragment_generation
    DATA_TASK3 = data_path   # nontoxic_smiles_generation

    SPLIT_DIR = split_dir


def main():
    ap = argparse.ArgumentParser(description="MolDeTox QA jsonl builder.")
    ap.add_argument(
        "--task",
        choices=[
            "task1",
            "task2",
            "task3",
            "task3_nontoxic_safe_generation",
            "task3_stepwise_cot_safe_generation",
            "all",
        ],
        default="all",
        help="Task to build, or 'all' for every task (default).",
    )
    ap.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="test",
        help=(
            "Split to materialize QA for ('train','test'); use 'all' to loop train then test "
            "(default 'test')."
        ),
    )
    ap.add_argument(
        "--input_csv",
        type=Path,
        default=None,
        help=(
            "Override the pair CSV. Defaults to the scaffold split under splits/default/."
        ),
    )
    ap.add_argument(
        "--train_csv",
        type=Path,
        default=None,
        help="Train CSV override when --split train or all.",
    )
    ap.add_argument(
        "--test_csv",
        type=Path,
        default=None,
        help="Test CSV override when --split test or all.",
    )
    ap.add_argument(
        "--qa_set",
        type=str,
        default=None,
        help=(
            f"Name of the QA set directory under MolDeTox/QA/ (default: {_DEFAULT_QA_SET_BUNDLE_NAME})."
        ),
    )
    ap.add_argument(
        "--no_desc",
        action="store_true",
        help="Strip bundled endpoint narratives from prompts.",
    )
    ap.add_argument(
        "--shuffle_seed",
        type=int,
        default=42,
        help="Shuffle RNG seed consulted when --shuffle is provided (default 42).",
    )
    ap.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle records before emitting JSONL (uses --shuffle_seed).",
    )
    ap.add_argument(
        "--no_shuffle",
        action="store_true",
        default=True,
        help="(Default) Preserve CSV order unless --shuffle is passed.",
    )
    args = ap.parse_args()

    global INCLUDE_ENDPOINT_DESCRIPTION
    INCLUDE_ENDPOINT_DESCRIPTION = not bool(args.no_desc)

    global BUILD_QA_SHUFFLE_SEED
    BUILD_QA_SHUFFLE_SEED = args.shuffle_seed if args.shuffle else None

    global QA_OUT_ROOT
    bundle = (args.qa_set or "").strip() or _DEFAULT_QA_SET_BUNDLE_NAME
    if args.no_desc and not bundle.endswith("_no_desc"):
        bundle = f"{bundle}_no_desc"
    safe = bundle.strip("/").replace("..", "").replace("\\", "_").replace("/", "_")
    if not safe:
        raise ValueError("--qa_set cannot be empty when provided.")
    QA_OUT_ROOT = (_QA_SETS_ROOT / safe).resolve()
    QA_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[qa_set] output_root={QA_OUT_ROOT}")

    def _input_csv_for_split(split: str) -> Path | None:
        if split == "train":
            return args.train_csv or args.input_csv
        if split == "test":
            return args.test_csv or args.input_csv
        return args.input_csv

    splits_to_run = ["train", "test"] if args.split == "all" else [args.split]

    for split_name in splits_to_run:
        if args.split == "all":
            print(f"[split={split_name}]")

        _configure_paths(
            split=split_name,
            input_csv=_input_csv_for_split(split_name),
            molecule_repr="both_repre",
        )

        if args.task in ("task1", "all"):
            build_task1()
        if args.task in ("task2", "all"):
            build_task2()
        if args.task in ("task3", "all"):
            build_task3()
        if args.task in ("task3_nontoxic_safe_generation", "all"):
            build_task3_nontoxic_safe_generation()
        if args.task in ("task3_stepwise_cot_safe_generation", "all"):
            build_task3_stepwise_cot_safe_generation()


if __name__ == "__main__":
    main()
