from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import evaluation as ev

ROOT = Path(__file__).resolve().parent

# A QA file is named <split>_<task>_<single|multi>.jsonl; this maps the task part onto the
# key evaluation.py uses. Longest match wins, so task3_safe_gen is not read as task3.
TASKS = {
    "task1": "task1",
    "task2": "task2",
    "task3_smiles_gen": "task3",
    "task3_safe_gen": "task3_nontoxic_safe_generation",
    "task3_cot_safe_gen": "task3_stepwise_cot_safe_generation",
}

_PREAMBLE = (
    "You are a molecular toxicity reasoning assistant specialized in SAFE and SMILES "
    "representations.\n"
    "Follow the task instruction exactly and return ONLY the requested JSON object.\n"
    "Do not add explanations, markdown, code fences, or prose outside the JSON.\n"
)
_ONE_KEY = 'Output schema: {"answer": "..."}\n'

# A QA row holds the question but no system message, and each task expects a different answer
# format, so the instruction is attached here.
SYSTEM = {
    "task1": _PREAMBLE
    + "Identify the fragment(s) of the toxic molecule responsible for its toxicity.\n"
    + "Return the toxic-only SAFE fragment string, dot-separated if there are several.\n"
    + _ONE_KEY,
    "task2": _PREAMBLE
    + "Generate the non-toxic replacement for the toxic fragment(s).\n"
    + "Return the non-toxic-only SAFE fragment string, dot-separated if there are several.\n"
    + _ONE_KEY,
    "task3": _PREAMBLE
    + "Generate the resulting non-toxic molecule as a single SMILES string.\n"
    + "Preserve the characteristics of the input molecule as far as possible.\n"
    + _ONE_KEY,
    "task3_nontoxic_safe_generation": _PREAMBLE
    + "Generate the resulting non-toxic molecule in SAFE representation.\n"
    + "Return the complete SAFE string, dot-separated if there are several fragments.\n"
    + _ONE_KEY,
    "task3_stepwise_cot_safe_generation": _PREAMBLE
    + "Work through the intermediate steps and return one JSON object holding "
    + '"step1_only_toxic_safe_fragments", "step2_only_nontoxic_safe_fragments" and "answer" '
    + "(the full non-toxic SAFE string), with the reasoning fields the prompt asks for.\n",
}


def find_qa(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if path.is_file():
        return [path]
    files = sorted(p for p in path.rglob("*.jsonl") if task_of(p))
    if not files:
        raise SystemExit(f"no QA files under {path}")
    return files


def task_of(qa: Path) -> str | None:
    stem = qa.stem
    for name in sorted(TASKS, key=len, reverse=True):
        if f"_{name}_" in stem or stem.startswith(f"{name}_"):
            return TASKS[name]
    return None


def step_of(qa: Path) -> str:
    return "multi_step" if qa.stem.endswith("_multi") else "single_step"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_json(text: str):
    """Pull the JSON object out of a reply, which often carries text around it."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return text


class OpenAIModel:
    def __init__(self, name: str, temperature: float, workers: int, retries: int = 3):
        from openai import OpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set")
        self.client = OpenAI()
        self.name = name
        self.temperature = temperature
        self.workers = workers
        self.retries = retries

    def _ask(self, system: str, question: str) -> str:
        for attempt in range(self.retries):
            try:
                reply = self.client.chat.completions.create(
                    model=self.name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": question},
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
                return reply.choices[0].message.content or ""
            except Exception as err:
                if attempt == self.retries - 1:
                    return f"ERROR: {err}"
                time.sleep(2 ** attempt)
        return ""

    def generate(self, system: str, questions: list[str]) -> list[str]:
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(lambda q: self._ask(system, q), questions))


class LocalModel:
    """Local checkpoint served through vLLM.

    Structured decoding is deliberately not used: constraining the output to a JSON schema lets
    a model satisfy the schema with empty fields, which then scores as a well-formed answer
    while containing nothing. Plain sampling plus parse_json is more honest here.
    """

    def __init__(self, path: str, temperature: float, tensor_parallel_size: int = 1,
                 max_model_len: int = 4096, max_tokens: int = 1024):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.llm = LLM(
            model=path,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
        )
        self.params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    def generate(self, system: str, questions: list[str]) -> list[str]:
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for q in questions
        ]
        return [out.outputs[0].text for out in self.llm.generate(prompts, self.params)]


def load_model(name: str, temperature: float, workers: int, tensor_parallel_size: int):
    if Path(name).expanduser().is_dir():
        return LocalModel(name, temperature, tensor_parallel_size=tensor_parallel_size)
    return OpenAIModel(name, temperature, workers)


def score(task: str, step: str, gold, pred, row: dict) -> dict:
    """Score one answer. The QA row goes along so PRS can read the toxic input molecule."""
    gold = gold if isinstance(gold, dict) else {"answer": gold}
    pred = pred if isinstance(pred, dict) else {"answer": pred or ""}
    if task == "task1":
        return ev.task1_toxic_fragment_identification_eval(gold, pred, step=step)
    if task == "task2":
        return ev.task2_nontoxic_fragment_generation_eval(gold, pred, step=step)
    if task == "task3":
        return ev.task3_nontoxic_smiles_generation_eval(gold, pred, context_row=row)
    if task == "task3_nontoxic_safe_generation":
        return ev.task3_nontoxic_safe_generation_eval(gold, pred, context_row=row)
    return ev.task3_stepwise_cot_nontoxic_safe_generation_eval(gold, pred, context_row=row)


def summarize(pred_path: Path, task: str, step: str, model: str,
              elapsed: float, n_new: int) -> None:
    rows = read_jsonl(pred_path)
    keys = ev.metric_keys_for(task, step)
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        ev.merge_finite_metrics_into_sums(sums, counts, {k: row.get(k) for k in keys})
    means = ev.augment_metrics_mean_with_em_accuracy(
        ev.mean_metrics_from_sums(sums, counts, keys)
    )
    out = pred_path.with_name(pred_path.stem.replace("predictions", "summary") + ".json")
    out.write_text(
        json.dumps(
            {
                "task": task,
                "step": step,
                "model": model,
                "total": len(rows),
                "inference_seconds": round(elapsed, 2),
                "inference_samples": n_new,
                "metrics_mean": means,
                "predictions_path": str(pred_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    shown = {k: round(v, 4) for k, v in means.items() if isinstance(v, float)}
    print(f"  n={len(rows)}  {shown}")
    print(f"  -> {out}")


def run(qa: Path, model, model_name: str, out_root: Path, limit: int) -> None:
    task, step = task_of(qa), step_of(qa)
    rows = read_jsonl(qa)
    if limit:
        rows = rows[:limit]

    out_dir = out_root / qa.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    # Naming the file after the QA it came from keeps the tasks, the two fragment settings and
    # any prompt variants from overwriting one another.
    tag = model_name.rstrip("/").split("/")[-1]
    pred_path = out_dir / f"predictions_{tag}_{qa.stem}.jsonl"

    answered = set()
    if pred_path.exists():
        answered = {
            row.get("id")
            for row in read_jsonl(pred_path)
            if not str(row.get("raw", "")).startswith("ERROR:")
        }
    todo = [row for row in rows if row.get("id") not in answered]
    print(f"{qa.stem}: {len(todo)} to run, {len(answered)} already answered")

    started = time.perf_counter()
    if todo:
        replies = model.generate(SYSTEM[task], [str(r.get("question", "")) for r in todo])
        with pred_path.open("a", encoding="utf-8") as f:
            for row, raw in zip(todo, replies):
                pred = parse_json(raw)
                gold = row.get("answer", "")
                record = {
                    "model": model_name,
                    "task": task,
                    "step": step,
                    "id": row.get("id"),
                    "dataset_name": row.get("dataset_name", ""),
                    "endpoint": row.get("endpoint") or row.get("dataset_name", ""),
                    "source_index": row.get("source_index"),
                    "gold": gold,
                    "pred": pred,
                    "raw": raw,
                    **score(task, step, gold, pred, row),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    summarize(pred_path, task, step, model_name, time.perf_counter() - started, len(todo))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--qa", type=Path, required=True, help="QA jsonl, or a directory of them")
    parser.add_argument("--model", required=True,
                        help="OpenAI model name, or path to a local checkpoint")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs")
    parser.add_argument("--limit", type=int, default=0, help="only the first N questions")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--workers", type=int, default=8, help="parallel OpenAI requests")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="GPUs to shard a local model over")
    args = parser.parse_args()

    files = find_qa(args.qa)
    print(f"{len(files)} QA file(s) | model: {args.model}")

    model = load_model(args.model, args.temperature, args.workers, args.tensor_parallel_size)
    for qa in files:
        run(qa, model, args.model, args.out.resolve(), args.limit)


if __name__ == "__main__":
    main()
