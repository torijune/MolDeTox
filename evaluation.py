from __future__ import annotations

from collections import Counter
import math
import re
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, QED

try:
    from rdkit.Chem.Fingerprints import FingerprintMols
except ImportError:
    FingerprintMols = None
try:
    from rdkit.Chem import MACCSkeys
except ImportError:
    MACCSkeys = None
# ---------------------------------------------------------------------------
# PRS: QED 6 descriptors (MW, ALOGP, HBA, HBD, PSA, ROTB), no extra scaling
# (inlined from eval_property.py)
# ---------------------------------------------------------------------------
PROPS_6 = ("MW", "ALOGP", "HBA", "HBD", "PSA", "ROTB")
ScoreMode = Literal["exponential", "linear", "reciprocal"]


def qed_six_from_mol(mol: Any) -> float:
    if mol is None:
        return float("nan")
    p = QED.properties(mol)
    w_mean = QED.WEIGHT_MEAN
    t = 0.0
    sw = 0.0
    for name in PROPS_6:
        pi = getattr(p, name)
        di = QED.ads(pi, QED.adsParameters[name])
        wi = getattr(w_mean, name)
        t += wi * math.log(di)
        sw += wi
    return math.exp(t / sw)


def qed_six_from_smiles(smi: str) -> float:
    return qed_six_from_mol(Chem.MolFromSmiles(smi))


def _score_from_x_abs_diff(x: float, mode: ScoreMode) -> float:
    if mode == "exponential":
        return math.exp(-x)
    if mode == "linear":
        return max(0.0, 1.0 - x)
    if mode == "reciprocal":
        return 1.0 / (1.0 + x)
    raise ValueError(f"unknown mode: {mode!r}")


def eval_prs(
    toxic_smiles: str,
    nontoxic_smiles: str,
    *,
    mode: ScoreMode = "exponential",
) -> dict[str, Any]:
    qt = qed_six_from_smiles(str(toxic_smiles))
    qn = qed_six_from_smiles(str(nontoxic_smiles))
    if not (math.isfinite(qt) and math.isfinite(qn)):
        return {
            "score": 0.0,
            "x_abs_diff": 0.0,
            "qed6_toxic": 0.0,
            "qed6_nontoxic": 0.0,
            "mode": mode,
        }
    x = abs(qt - qn)
    score = _score_from_x_abs_diff(x, mode)
    return {
        "score": float(score),
        "x_abs_diff": float(x),
        "qed6_toxic": float(qt),
        "qed6_nontoxic": float(qn),
        "mode": mode,
    }


def compute_qed(smiles: str) -> float:
    """RDKit drug-likeness QED (0-1). Invalid SMILES raises ValueError."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return float(QED.qed(mol))


# SAFE -> SMILES. Without the `safe` package, SAFE answers cannot be decoded and every
# task 3 prediction scores as invalid, so the error is kept for a clearer message later.
SAFE_DECODE_IMPORT_ERROR: Optional[BaseException] = None
try:
    from safe.converter import decode as safe_decode
except Exception as err:
    safe_decode = None
    SAFE_DECODE_IMPORT_ERROR = err

MORGAN_FP_NBITS = 1024

# The task 3 columns of the paper table, in the same order:
# Acc.(%), BLEU1, Morgan, Validity, PRS, Copy(%).
TASK3_METRIC_KEYS: list[str] = [
    "exact_match",
    "bleu",
    "morgan_fts",
    "validity",
    "prs_property_score",
    "copy_rate",
]

# Still computed and written per row, but left out of the reported means. Levenshtein tracks
# string length more than chemistry, and on activity-cliff pairs MACCS/RDK similarity reaches
# ~0.89 for a prediction that simply copies the toxic input, so neither is meaningful alone.
TASK3_EXTRA_METRIC_KEYS: list[str] = ["levenshtein", "rdk_fts", "maccs_fts"]

TASK_METRIC_KEYS: dict[str, dict[str, list[str]]] = {
    "task1": {
        "single_step": ["fragment_EM"],
        "multi_step": ["fragment_EM", "fragment_F1"],
    },
    "task2": {
        "single_step": ["fragment_EM", "fragment_Levenshtein"],
        "multi_step": ["fragment_EM", "fragment_Levenshtein", "fragment_F1"],
    },
    "task3": {
        "single_step": list(TASK3_METRIC_KEYS),
        "multi_step": list(TASK3_METRIC_KEYS),
    },
    "task3_nontoxic_safe_generation": {
        "single_step": list(TASK3_METRIC_KEYS),
        "multi_step": list(TASK3_METRIC_KEYS),
    },
    "task3_stepwise_cot_safe_generation": {
        "single_step": list(TASK3_METRIC_KEYS),
        "multi_step": list(TASK3_METRIC_KEYS),
    },
}


def augment_metrics_mean_with_em_accuracy(
    metric_means: Dict[str, Optional[float]],
) -> Dict[str, Optional[float]]:
    """
    Append {base}_Acc = mean * 100 companions for exact-match style metrics when metrics_mean stores 0-1 means.

    Targets keys named EM, ending with _EM, or exact_match for SMILES equality.
    """
    out: Dict[str, Optional[float]] = dict(metric_means)
    for k, v in metric_means.items():
        if not (k == "EM" or k.endswith("_EM") or k == "exact_match"):
            continue
        acc_key = f"{k}_Acc"
        if v is None:
            out[acc_key] = None
        else:
            out[acc_key] = float(v) * 100.0
    return out


StepNorm = Literal["single_step", "multi_step"]

_TASK3_LIKE: frozenset[str] = frozenset(
    {
        "task3",
        "task3_nontoxic_safe_generation",
        "task3_stepwise_cot_safe_generation",
    }
)


def metric_keys_for(task: str, step: StepNorm | str) -> list[str]:
    """Ordered metric keys expected for task + step aggregation."""
    s = str(step).strip()
    if s not in ("single_step", "multi_step"):
        raise ValueError(f"step must be single_step or multi_step, got {step!r}")
    tmap = TASK_METRIC_KEYS.get(task)
    if tmap is None:
        raise KeyError(f"unknown task: {task!r}")
    if task in _TASK3_LIKE:
        return list(tmap["single_step"])
    return list(tmap[s])


def merge_finite_metrics_into_sums(
    metric_sums: Dict[str, float],
    metric_counts: Dict[str, int],
    metrics: Dict[str, Any],
) -> None:
    """Accumulate finite numeric metrics; skip NaN/inf entries (e.g., unstable PRS rows)."""
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            fv = float(v)
            if math.isfinite(fv):
                metric_sums[k] = metric_sums.get(k, 0.0) + fv
                metric_counts[k] = metric_counts.get(k, 0) + 1


def mean_metrics_from_sums(
    metric_sums: Dict[str, float],
    metric_counts: Dict[str, int],
    keys: list[str],
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for k in keys:
        c = metric_counts.get(k, 0)
        if c > 0 and k in metric_sums:
            out[k] = metric_sums[k] / c
        else:
            out[k] = None
    return out


_QUESTION_SMILES_RE = re.compile(r"SMILES = \'([^\']+)\'")


def _toxic_smiles_from_context(context_row: Optional[dict]) -> str:
    """Toxic input molecule of a QA row, from toxic_smiles or from the question text."""
    if not context_row:
        return ""
    v = str(context_row.get("toxic_smiles", "") or "").strip()
    if v:
        return v
    m = _QUESTION_SMILES_RE.search(str(context_row.get("question", "") or ""))
    return m.group(1) if m else ""


def _prs_for_prediction(
    pred_smiles: str,
    context_row: Optional[dict] = None,
) -> Tuple[float, float, float, float]:
    """PRS between the toxic input molecule and the predicted non-toxic SMILES.

    Returns zeros when either molecule is missing or RDKit rejects it, so that a failed
    prediction counts as a miss rather than dropping out of the average.
    """
    zero4 = (0.0, 0.0, 0.0, 0.0)
    toxic = _toxic_smiles_from_context(context_row)
    pred_s = (pred_smiles or "").strip()
    if not toxic or not pred_s:
        return zero4
    try:
        r = eval_prs(toxic, pred_s, mode="exponential")
    except Exception:
        return zero4

    def _prs_float(v: Any) -> float:
        try:
            x = float(v)
        except (TypeError, ValueError):
            return 0.0
        return x if math.isfinite(x) else 0.0

    return (
        _prs_float(r.get("score", 0.0)),
        _prs_float(r.get("x_abs_diff", 0.0)),
        _prs_float(r.get("qed6_toxic", 0.0)),
        _prs_float(r.get("qed6_nontoxic", 0.0)),
    )


def _copy_of_input(can_pred: Optional[str], context_row: Optional[dict]) -> float:
    """1.0 when the prediction is the toxic input molecule handed back unchanged.

    The two molecules of a cliff pair are never identical, so a copy is always wrong; it
    nevertheless scores near the top on similarity and validity, and above the ground truth
    on PRS. Counting copies separates a model that did not edit from one that edited badly.
    """
    toxic = _toxic_smiles_from_context(context_row)
    if not can_pred or not toxic:
        return 0.0
    mol = _mol_from_smiles(toxic)
    if mol is None:
        return 0.0
    return 1.0 if Chem.MolToSmiles(mol, canonical=True) == can_pred else 0.0


def _extract_answer(ans: Any) -> str:
    """Pull string answers from dict-or-string gold/pred payloads."""
    if ans is None:
        return ""
    if isinstance(ans, dict):
        return str(ans.get("answer", "")).strip()
    return str(ans).strip()


def _tokenize_safe_fragments(s: str) -> list[str]:
    """
    Split SAFE strings on . after stripping whitespace; empty tokens drop out.

    Single-step labels usually carry one token; multi-step examples carry two or more.
    """
    s = (s or "").strip()
    if not s:
        return []
    # Remove inline spaces then split on separator dots
    return [tok for tok in s.replace(" ", "").split(".") if tok]


def _fragments_multiset_equal(gold: str, pred: str) -> bool:
    """
    Compare dot-separated SAFE fragments as multisets (order ignored, duplicates respected).

    Example: Frag1.Frag2.Frag3 matches Frag2.Frag3.Frag1.
    """
    g = _tokenize_safe_fragments(gold)
    p = _tokenize_safe_fragments(pred)
    return sorted(g) == sorted(p)


def _fragment_set_precision_recall_f1(gold: str, pred: str) -> Tuple[float, float, float]:
    """Precision, recall and F1 over the sets of dot-separated SAFE fragments.

    Two empty answers count as a match, so a pair with nothing to change is not penalised.
    """
    gold_toks = _tokenize_safe_fragments(gold)
    pred_toks = _tokenize_safe_fragments(pred)
    gold_set = set(gold_toks)
    pred_set = set(pred_toks)

    if not gold_set and not pred_set:
        return 1.0, 1.0, 1.0
    if not gold_set:
        return 0.0 if pred_set else 1.0, 1.0, (0.0 if pred_set else 1.0)
    if not pred_set:
        return 0.0, 0.0, 0.0

    tp = len(gold_set & pred_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _tokenize_char_ngrams(s: str, n: int = 4) -> list[str]:
    """Character n-gram iterator with sliding windows (BLEU-style overlap)."""
    s = (s or "").strip().replace(" ", "")
    if not s or n < 1:
        return []
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def _bleu1_safe_fragments(gold: str, pred: str, use_char_ngrams: bool = True, ngram_n: int = 1) -> float:
    """
    BLEU-1 style precision for SAFE strings.

    With use_char_ngrams the text is scored over character n-grams (ngram_n=1 means single
    characters); otherwise the dot-separated fragments are the tokens.
    """
    if use_char_ngrams:
        gold_tokens = _tokenize_char_ngrams(gold, ngram_n)
        pred_tokens = _tokenize_char_ngrams(pred, ngram_n)
    else:
        gold_tokens = _tokenize_safe_fragments(gold)
        pred_tokens = _tokenize_safe_fragments(pred)

    if not pred_tokens or not gold_tokens:
        return 0.0

    gold_counts = Counter(gold_tokens)
    pred_counts = Counter(pred_tokens)

    overlap = 0
    for t, c in pred_counts.items():
        overlap += min(c, gold_counts.get(t, 0))

    precision = overlap / max(len(pred_tokens), 1)
    return float(precision)


def _safe_to_smiles_validity(safe_str: str) -> float:
    """
    Decode SAFE to SMILES (when safe_decode exists) and score RDKit parseability.

    Returns 0.0 if helpers are missing or Mol construction fails; 1.0 on success.
    """
    safe_str = (safe_str or "").strip()
    if not safe_str or safe_decode is None:
        return 0.0
    try:
        # SAFE decode -> SMILES
        decoded_smiles = safe_decode(safe_str)
    except Exception:
        return 0.0

    if not decoded_smiles:
        return 0.0

    try:
        mol = Chem.MolFromSmiles(str(decoded_smiles))
    except Exception:
        return 0.0

    return 1.0 if mol is not None else 0.0


def _decode_safe_to_smiles(safe_str: str) -> Optional[str]:
    """Decode SAFE to SMILES or return None."""
    safe_str = (safe_str or "").strip()
    if not safe_str or safe_decode is None:
        return None
    try:
        decoded_smiles = safe_decode(safe_str)
    except Exception:
        return None
    decoded_smiles = (decoded_smiles or "").strip()
    return decoded_smiles or None


def _mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    smiles = (smiles or "").strip()
    if not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def _morgan_tanimoto(
    smiles1: str,
    smiles2: str,
    radius: int = 2,
    nbits: int = MORGAN_FP_NBITS,
) -> Optional[float]:
    """
    Morgan fingerprint Tanimoto between SMILES strings (None on failure).

    Callers should canonicalize SMILES first because downstream comparisons assume normalized forms.
    """
    mol1 = _mol_from_smiles(smiles1)
    mol2 = _mol_from_smiles(smiles2)
    if mol1 is None or mol2 is None:
        return None
    try:
        # Prefer MorganGenerator to avoid legacy GetMorganFingerprintAsBitVect warnings.
        try:
            from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

            gen = GetMorganGenerator(radius=radius, fpSize=nbits)
            fp1 = gen.GetFingerprint(mol1)
            fp2 = gen.GetFingerprint(mol2)
        except Exception:
            # Older RDKit fallback
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits=nbits)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits=nbits)
        return float(DataStructs.TanimotoSimilarity(fp1, fp2))
    except Exception:
        return None


def _rdkit_tanimoto(smiles1: str, smiles2: str) -> float:
    """RDKit topological fingerprint Tanimoto (0.0 if unavailable)."""
    if FingerprintMols is None:
        return 0.0
    mol1 = _mol_from_smiles(smiles1)
    mol2 = _mol_from_smiles(smiles2)
    if mol1 is None or mol2 is None:
        return 0.0
    try:
        fp1 = FingerprintMols.FingerprintMol(mol1)
        fp2 = FingerprintMols.FingerprintMol(mol2)
        return float(DataStructs.TanimotoSimilarity(fp1, fp2))
    except Exception:
        return 0.0


def _maccs_tanimoto(smiles1: str, smiles2: str) -> float:
    """MACCS key Tanimoto similarity (0.0 if unavailable)."""
    if MACCSkeys is None:
        return 0.0
    mol1 = _mol_from_smiles(smiles1)
    mol2 = _mol_from_smiles(smiles2)
    if mol1 is None or mol2 is None:
        return 0.0
    try:
        fp1 = MACCSkeys.GenMACCSKeys(mol1)
        fp2 = MACCSkeys.GenMACCSKeys(mol2)
        return float(DataStructs.TanimotoSimilarity(fp1, fp2))
    except Exception:
        return 0.0


def _levenshtein(a: str, b: str) -> int:
    """Classic Levenshtein edit distance."""
    a = a or ""
    b = b or ""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,      # deletion
                dp[j - 1] + 1,  # insertion
                prev + cost,    # substitution
            )
            prev = cur
    return dp[lb]


def _fragment_levenshtein_mean_over_pred(gold: str, pred: str) -> float:
    """
    Fragment-aware Levenshtein for Task2 (lower is better, mirroring Task3 string distance).

    For each predicted fragment token pick the minimum distance to any gold fragment and average those minima.
    """
    gold_toks = _tokenize_safe_fragments(gold)
    pred_toks = _tokenize_safe_fragments(pred)
    if not gold_toks or not pred_toks:
        return float(_levenshtein((gold or "").replace(" ", ""), (pred or "").replace(" ", "")))
    per_pred: list[float] = []
    for p in pred_toks:
        best: float | None = None
        for g in gold_toks:
            d = float(_levenshtein(p, g))
            if best is None or d < best:
                best = d
                if best <= 0.0:
                    break
        per_pred.append(float(best or 0.0))
    if not per_pred:
        return float(_levenshtein((gold or "").replace(" ", ""), (pred or "").replace(" ", "")))
    return float(sum(per_pred) / len(per_pred))


def task1_toxic_fragment_identification_eval(
    gold_answer: Any,
    llm_answer: Any,
    *,
    step: StepNorm,
) -> Dict[str, float]:
    """
    Task 1 toxic fragment tagging.

    single_step reports fragment_EM; multi_step adds fragment_F1 over the fragment multiset.
    """
    gold = _extract_answer(gold_answer)
    pred = _extract_answer(llm_answer)

    fragment_EM = 1.0 if gold and _fragments_multiset_equal(gold, pred) else 0.0
    out: Dict[str, float] = {"fragment_EM": fragment_EM}
    if step == "multi_step":
        _, _, fragment_F1 = _fragment_set_precision_recall_f1(gold, pred)
        out["fragment_F1"] = float(fragment_F1)
    return out

def task2_nontoxic_fragment_generation_eval(
    gold_answer: Any,
    llm_answer: Any,
    *,
    step: StepNorm,
    context_row: Optional[dict] = None,
) -> Dict[str, float]:
    """Task 2: exact match and Levenshtein over the replacement fragments, plus F1 for multi_step.

    context_row is accepted for a uniform call signature and is not used.
    """
    del context_row
    gold = _extract_answer(gold_answer)
    pred = _extract_answer(llm_answer)

    fragment_EM = 1.0 if gold and _fragments_multiset_equal(gold, pred) else 0.0
    fragment_Levenshtein = _fragment_levenshtein_mean_over_pred(gold, pred)
    out: Dict[str, float] = {
        "fragment_EM": fragment_EM,
        "fragment_Levenshtein": float(fragment_Levenshtein),
    }
    if step == "multi_step":
        _, _, fragment_F1 = _fragment_set_precision_recall_f1(gold, pred)
        out["fragment_F1"] = float(fragment_F1)
    return out

def task3_nontoxic_smiles_generation_eval(
    gold_answer: Any,
    llm_answer: Any,
    context_row: Optional[dict] = None,
) -> Dict[str, float]:
    """Task 3 when the answer is a SMILES string.

    Scores are computed on RDKit-canonical SMILES and fall back to 0 when either side fails to
    parse; validity still reports whether the raw prediction parsed.
    """
    gold_s = (_extract_answer(gold_answer) or "").strip()
    pred_s = (_extract_answer(llm_answer) or "").strip()

    validity = 1.0 if pred_s and _mol_from_smiles(pred_s) is not None else 0.0

    can_gold: Optional[str] = None
    can_pred: Optional[str] = None
    if gold_s:
        mol_g = _mol_from_smiles(gold_s)
        if mol_g is not None:
            can_gold = Chem.MolToSmiles(mol_g, canonical=True)
    if pred_s:
        mol_p = _mol_from_smiles(pred_s)
        if mol_p is not None:
            can_pred = Chem.MolToSmiles(mol_p, canonical=True)

    exact_match = 1.0 if (can_gold and can_pred and can_gold == can_pred) else 0.0
    if can_gold is not None and can_pred is not None:
        bleu = _bleu1_safe_fragments(can_gold, can_pred, use_char_ngrams=True, ngram_n=1)
        levenshtein = float(_levenshtein(can_gold, can_pred))
    else:
        bleu = 0.0
        levenshtein = 0.0

    rdk_fts = 0.0
    maccs_fts = 0.0
    morgan_fts = 0.0
    if can_gold and can_pred:
        rdk_fts = _rdkit_tanimoto(can_gold, can_pred)
        maccs_fts = _maccs_tanimoto(can_gold, can_pred)
        m = _morgan_tanimoto(can_gold, can_pred)
        morgan_fts = m if m is not None else 0.0

    pred_for_prs = ((can_pred or pred_s) or "").strip()
    prs_s, _, _, _ = _prs_for_prediction(pred_for_prs, context_row=context_row)

    return {
        "exact_match": float(exact_match),
        "bleu": float(bleu),
        "levenshtein": float(levenshtein),
        "rdk_fts": float(rdk_fts),
        "maccs_fts": float(maccs_fts),
        "morgan_fts": float(morgan_fts),
        "validity": float(validity),
        "prs_property_score": float(prs_s),
        "copy_rate": _copy_of_input(can_pred, context_row),
    }


def task3_nontoxic_safe_generation_eval(
    gold_answer: Any,
    llm_answer: Any,
    context_row: Optional[dict] = None,
) -> Dict[str, float]:
    """
    Task 3 SAFE generation (dot-concatenated SAFE strings).

    After decoding to canonical SMILES the eight Task3 keys match task3_nontoxic_smiles_generation_eval.
    """
    gold_safe = (_extract_answer(gold_answer) or "").strip()
    pred_safe = (_extract_answer(llm_answer) or "").strip()

    gold_decoded = _decode_safe_to_smiles(gold_safe)
    pred_decoded = _decode_safe_to_smiles(pred_safe)

    can_gold: Optional[str] = None
    can_pred: Optional[str] = None

    mol_gold = _mol_from_smiles(gold_decoded or "") if gold_decoded else None
    mol_pred = _mol_from_smiles(pred_decoded or "") if pred_decoded else None
    if mol_gold is not None:
        can_gold = Chem.MolToSmiles(mol_gold, canonical=True)
    if mol_pred is not None:
        can_pred = Chem.MolToSmiles(mol_pred, canonical=True)

    decode_ok = pred_decoded is not None
    mol_ok = mol_pred is not None
    validity = 1.0 if (decode_ok and mol_ok) else 0.0

    exact_match = (
        1.0
        if (can_gold is not None and can_pred is not None and can_gold == can_pred)
        else 0.0
    )

    if can_gold is not None and can_pred is not None:
        bleu = _bleu1_safe_fragments(can_gold, can_pred, use_char_ngrams=True, ngram_n=1)
        levenshtein = float(_levenshtein(can_gold, can_pred))
    else:
        bleu = 0.0
        levenshtein = 0.0

    rdk_fts = 0.0
    maccs_fts = 0.0
    morgan_fts = 0.0
    if can_gold is not None and can_pred is not None:
        rdk_fts = _rdkit_tanimoto(can_gold, can_pred)
        maccs_fts = _maccs_tanimoto(can_gold, can_pred)
        morgan = _morgan_tanimoto(can_gold, can_pred)
        morgan_fts = morgan if morgan is not None else 0.0

    pred_for_prs = (can_pred or pred_decoded or "").strip()
    prs_s, _, _, _ = _prs_for_prediction(pred_for_prs, context_row=context_row)

    return {
        "exact_match": float(exact_match),
        "bleu": float(bleu),
        "levenshtein": float(levenshtein),
        "rdk_fts": float(rdk_fts),
        "maccs_fts": float(maccs_fts),
        "morgan_fts": float(morgan_fts),
        "validity": float(validity),
        "prs_property_score": float(prs_s),
        "copy_rate": _copy_of_input(can_pred, context_row),
    }


def task3_stepwise_cot_nontoxic_safe_generation_eval(
    gold_answer: Any,
    llm_answer: Any,
    context_row: Optional[dict] = None,
) -> Dict[str, float]:
    """
    Stepwise SAFE CoT evaluated solely on final JSON answer using task3_nontoxic_safe_generation_eval.
    """
    return task3_nontoxic_safe_generation_eval(
        gold_answer, llm_answer, context_row=context_row
    )
