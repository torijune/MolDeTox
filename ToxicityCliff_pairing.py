from __future__ import annotations

import argparse
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, MolSurf
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from safe_functions import SAFEEncodeError, smiles_to_safe as _encode_smiles_to_safe

_SCRIPT_DIR = Path(__file__).resolve().parent
MOLDETOX_ROOT = _SCRIPT_DIR


FINAL_TOXICITYCLIFF_COLUMNS: List[str] = [
    "dataset_name",
    "endpoint",
    "toxic_smiles",
    "nontoxic_smiles",
    "toxic_safe",
    "nontoxic_safe",
    "only_toxic_safe_fragments",
    "only_nontoxic_safe_fragments",
    "common_safe_fragments",
]

# safe-pipeline always writes this single nine-column artifact
TOXICITYCLIFF_CSV = MOLDETOX_ROOT / "toxicitycliff.csv"

EXCLUDE_PAIRING_DATASETS = {"toxcast_df"}




def _canonical_smiles_for_export(val: Any) -> str:
    """Canonical RDKit SMILES (isomeric) for exported toxic/nontoxic reference columns."""
    if val is None:
        return ""
    try:
        if isinstance(val, float) and np.isnan(val):
            return ""
    except Exception:
        pass
    if pd.isna(val):
        return ""
    t = str(val).strip()
    if not t or t.lower() == "nan":
        return ""
    mol = Chem.MolFromSmiles(t)
    if mol is None:
        return t
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def finalize_toxicitycliff_pair_table(df: pd.DataFrame) -> pd.DataFrame:
    """Keep nine export columns and canonicalize SMILES strings."""
    out = df.copy()
    if "toxic_smiles" in out.columns:
        out["toxic_smiles"] = out["toxic_smiles"].map(_canonical_smiles_for_export)
    if "nontoxic_smiles" in out.columns:
        out["nontoxic_smiles"] = out["nontoxic_smiles"].map(_canonical_smiles_for_export)
    missing = [c for c in FINAL_TOXICITYCLIFF_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Final export missing columns: {missing}")
    return out[FINAL_TOXICITYCLIFF_COLUMNS]


ECFP_RADIUS = 2
ECFP_SIZE = 1024
SIM_THRESHOLD = 0.9
CHUNK_TOXIC_SMILES = 200

PROPERTY_DESCRIPTOR_NAMES = ["MW", "logP", "TPSA", "HBD", "HBA", "RotB"]
SAFE_SEP = "."
COL_ONLY_TOXIC_FRAG = "only_toxic_safe_fragments"
COL_ONLY_NONTOXIC_FRAG = "only_nontoxic_safe_fragments"

def _levenshtein_distance_py(a: str, b: str) -> int:
    """Pure-Python Wagner-Fischer distance (no optional deps)."""
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
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[lb]


def _smiles_similarity_batch_pure_python(
    toxic_chunk: list[str], nontoxic_list: list[str]
) -> np.ndarray:
    """Fallback normalized Levenshtein similarity without rapidfuzz."""
    n_t, n_n = len(toxic_chunk), len(nontoxic_list)
    out = np.zeros((n_t, n_n), dtype=np.float32)
    for i in range(n_t):
        for j in range(n_n):
            d = float(_levenshtein_distance_py(toxic_chunk[i], nontoxic_list[j]))
            mx = float(max(len(toxic_chunk[i]), len(nontoxic_list[j]), 1))
            out[i, j] = 1.0 - (d / mx)
    return out


# =============================================================================
# SMILES / similarity helpers (MolecularACE-aligned)
# =============================================================================


def canonicalize_smiles_list(
    smiles_list: list[str],
    *,
    isomeric: bool = True,
    keep_invalid: bool = True,
) -> list[str]:
    """Return RDKit-canonical SMILES variants."""
    out: list[str] = []
    for smi in smiles_list:
        if not smi or not isinstance(smi, str):
            out.append("" if not keep_invalid else (smi if isinstance(smi, str) else ""))
            continue
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                out.append(smi if keep_invalid else "")
                continue
            out.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric))
        except Exception:
            out.append(smi if keep_invalid else "")
    return out


def build_ecfp4_list(smiles_list: list[str], include_chirality: bool = False) -> list:
    """ECFP fingerprints (radius 2 / 1024 bits); None placeholders on failure."""
    fpgen = AllChem.GetMorganGenerator(
        radius=ECFP_RADIUS, fpSize=ECFP_SIZE, includeChirality=include_chirality
    )
    out = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is not None:
                out.append(fpgen.GetFingerprint(mol))
            else:
                out.append(None)
        except Exception:
            out.append(None)
    return out


def build_scaffold_fp_list(smiles_list: list[str]) -> list:
    """Scaffold fingerprints; None when Murcko decomposition fails."""
    fpgen = AllChem.GetMorganGenerator(radius=ECFP_RADIUS, fpSize=ECFP_SIZE, includeChirality=False)
    out = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                out.append(None)
                continue
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold is None or scaffold.GetNumHeavyAtoms() == 0:
                out.append(None)
                continue
            out.append(fpgen.GetFingerprint(scaffold))
        except Exception:
            out.append(None)
    return out


def smiles_similarity_batch(toxic_chunk: list[str], nontoxic_list: list[str]) -> np.ndarray:
    """Normalized Levenshtein similarity matrix toxic_chunk × nontoxic_list."""
    if not toxic_chunk or not nontoxic_list:
        return np.zeros((len(toxic_chunk), len(nontoxic_list)), dtype=np.float32)
    try:
        from rapidfuzz import process
        from rapidfuzz.distance import Levenshtein
    except ImportError:
        return _smiles_similarity_batch_pure_python(toxic_chunk, nontoxic_list)
    dist = process.cdist(
        toxic_chunk,
        nontoxic_list,
        scorer=Levenshtein.distance,
        dtype=np.int32,
        workers=1,
    )
    len_t = np.array([len(s) for s in toxic_chunk], dtype=np.float32)
    len_n = np.array([len(s) for s in nontoxic_list], dtype=np.float32)
    max_len = np.maximum(len_t[:, None], len_n[None, :])
    np.maximum(max_len, 1.0, out=max_len)
    return 1.0 - (dist.astype(np.float32) / max_len)


def process_endpoint(
    dataset: str,
    endpoint: str,
    toxic_smiles: list[str],
    nontoxic_smiles: list[str],
    save_sim_path: Path | None = None,
    canonicalize_smiles: bool = True,
) -> tuple[str, str, list[tuple[str, str]], int]:
    """Candidate pairs for one endpoint: keep a pair if any of the three similarity rules reaches 0.9.

    Returns (dataset, endpoint, pairs, count).
    """
    n_t, n_n = len(toxic_smiles), len(nontoxic_smiles)
    empty: list[tuple[str, str]] = []
    if n_t == 0 or n_n == 0:
        return dataset, endpoint, empty, 0

    if canonicalize_smiles:
        toxic_smiles = canonicalize_smiles_list(toxic_smiles, isomeric=True, keep_invalid=True)
        nontoxic_smiles = canonicalize_smiles_list(nontoxic_smiles, isomeric=True, keep_invalid=True)

    fp_toxic_full = build_ecfp4_list(toxic_smiles, include_chirality=False)
    fp_nontoxic_full = build_ecfp4_list(nontoxic_smiles, include_chirality=False)
    valid_n_full = [(j, fp) for j, fp in enumerate(fp_nontoxic_full) if fp is not None]
    if not valid_n_full:
        return dataset, endpoint, empty, 0
    fp_nontoxic_full_list = [f for _, f in valid_n_full]

    fp_toxic_scaffold = build_scaffold_fp_list(toxic_smiles)
    fp_nontoxic_scaffold = build_scaffold_fp_list(nontoxic_smiles)
    valid_n_scaffold = [(j, fp) for j, fp in enumerate(fp_nontoxic_scaffold) if fp is not None]
    fp_nontoxic_scaffold_list = [f for _, f in valid_n_scaffold]

    ecfp4_full_sim = np.full((n_t, n_n), np.nan, dtype=np.float32)
    scaffold_sim = np.full((n_t, n_n), np.nan, dtype=np.float32)

    for i in range(n_t):
        if fp_toxic_full[i] is not None:
            sims = DataStructs.BulkTanimotoSimilarity(fp_toxic_full[i], fp_nontoxic_full_list)
            for k, sim in enumerate(sims):
                j = valid_n_full[k][0]
                ecfp4_full_sim[i, j] = sim
        if fp_toxic_scaffold[i] is not None:
            sims = DataStructs.BulkTanimotoSimilarity(fp_toxic_scaffold[i], fp_nontoxic_scaffold_list)
            for k, sim in enumerate(sims):
                j = valid_n_scaffold[k][0]
                scaffold_sim[i, j] = sim

    smiles_sim = np.full((n_t, n_n), np.nan, dtype=np.float32)
    if nontoxic_smiles:
        for start in range(0, n_t, CHUNK_TOXIC_SMILES):
            end = min(start + CHUNK_TOXIC_SMILES, n_t)
            smiles_sim[start:end, :] = smiles_similarity_batch(toxic_smiles[start:end], nontoxic_smiles)

    full_ok = ecfp4_full_sim >= SIM_THRESHOLD
    scaffold_ok = scaffold_sim >= SIM_THRESHOLD
    smiles_ok = np.isfinite(smiles_sim) & (smiles_sim >= SIM_THRESHOLD)
    pass_pair = full_ok | scaffold_ok | smiles_ok

    pairs_all = set(zip(*np.where(pass_pair)))
    rows_all = [(toxic_smiles[i], nontoxic_smiles[j]) for i, j in pairs_all]

    if save_sim_path is not None:
        save_sim_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_sim_path,
            ecfp4_full_molecule_sim=ecfp4_full_sim,
            substructure_sim=ecfp4_full_sim,
            scaffold_sim=scaffold_sim,
            smiles_sim=smiles_sim,
            n_toxic=n_t,
            n_nontoxic=n_n,
        )

    return dataset, endpoint, rows_all, len(pairs_all)






# =============================================================================
# SAFE mappings
# =============================================================================




def build_safe_mapping_from_pairs_with_encode(df: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    """Encode every distinct toxic_smiles / nontoxic_smiles into SAFE."""
    smiles_to_safe: dict[str, str] = {}
    canon_to_safe: dict[str, str] = {}
    unique: set[str] = set()
    for col in ("toxic_smiles", "nontoxic_smiles"):
        if col not in df.columns:
            continue
        for v in df[col].astype(str):
            s = str(v).strip()
            if s and s.lower() != "nan":
                unique.add(s)
    for s in tqdm(sorted(unique), desc="SMILES->SAFE (safe_functions)"):
        if (smiles_to_safe.get(s) or "").strip():
            continue
        out = ""
        try:
            out = _encode_smiles_to_safe(s, canonical=True) or ""
        except SAFEEncodeError:
            out = ""
        except Exception:
            out = ""
        smiles_to_safe[s] = out
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            can = Chem.MolToSmiles(mol, isomericSmiles=True)
            canon_to_safe[can] = out
    return smiles_to_safe, canon_to_safe


def lookup_safe_column(
    smiles_series: pd.Series,
    smiles_to_safe: dict[str, str],
    canon_to_safe: dict[str, str],
) -> list[str]:
    out: list[str] = []
    for s in smiles_series:
        s = str(s).strip() if pd.notna(s) else ""
        safe_str = smiles_to_safe.get(s)
        if safe_str is None and s:
            safe_str = canon_to_safe.get(s, "")
        out.append(safe_str if safe_str is not None else "")
    return out


def attach_safe_to_pairs(
    df: pd.DataFrame,
    smiles_to_safe: dict[str, str],
    canon_to_safe: dict[str, str],
) -> pd.DataFrame:
    """Augment dataframe with toxic_safe / nontoxic_safe lookups."""
    return df.assign(
        toxic_safe=lookup_safe_column(df["toxic_smiles"], smiles_to_safe, canon_to_safe),
        nontoxic_safe=lookup_safe_column(df["nontoxic_smiles"], smiles_to_safe, canon_to_safe),
    )




# =============================================================================
# SAFE fragment comparisons (compare_safe)
# =============================================================================


def safe_to_fragments(safe_str: Any) -> set[str]:
    if pd.isna(safe_str) or not str(safe_str).strip():
        return set()
    return {s.strip() for s in str(safe_str).split(SAFE_SEP) if s.strip()}


def compare_fragments(toxic_safe: Any, nontoxic_safe: Any) -> tuple[set[str], set[str], set[str]]:
    t_set = safe_to_fragments(toxic_safe)
    n_set = safe_to_fragments(nontoxic_safe)
    common = t_set & n_set
    only_toxic = t_set - n_set
    only_nontoxic = n_set - t_set
    return common, only_toxic, only_nontoxic




def enrich_pairs_with_safe_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Add fragment analytics columns compatible with attach-safe CSVs."""
    if "toxic_safe" not in df.columns or "nontoxic_safe" not in df.columns:
        raise ValueError("Need toxic_safe, nontoxic_safe columns.")

    common_list: list[str] = []
    only_toxic_list: list[str] = []
    only_nontoxic_list: list[str] = []
    has_safe_diff_list: list[bool] = []
    n_common_list: list[int] = []
    n_only_toxic_list: list[int] = []
    n_only_nontoxic_list: list[int] = []
    toxic_fragments_str_list: list[str] = []
    nontoxic_fragments_str_list: list[str] = []

    for _, row in df.iterrows():
        toxic_safe = row.get("toxic_safe", "")
        nontoxic_safe = row.get("nontoxic_safe", "")
        common, only_toxic, only_nontoxic = compare_fragments(toxic_safe, nontoxic_safe)

        toxic_fragments_str_list.append(SAFE_SEP.join(sorted(safe_to_fragments(toxic_safe))))
        nontoxic_fragments_str_list.append(SAFE_SEP.join(sorted(safe_to_fragments(nontoxic_safe))))
        common_list.append(SAFE_SEP.join(sorted(common)))
        only_toxic_list.append(SAFE_SEP.join(sorted(only_toxic)))
        only_nontoxic_list.append(SAFE_SEP.join(sorted(only_nontoxic)))
        has_safe_diff_list.append(len(only_toxic) > 0 or len(only_nontoxic) > 0)
        n_common_list.append(len(common))
        n_only_toxic_list.append(len(only_toxic))
        n_only_nontoxic_list.append(len(only_nontoxic))

    return df.assign(
        toxic_safe_fragments=toxic_fragments_str_list,
        nontoxic_safe_fragments=nontoxic_fragments_str_list,
        common_safe_fragments=common_list,
        only_toxic_safe_fragments=only_toxic_list,
        only_nontoxic_safe_fragments=only_nontoxic_list,
        n_common_safe=n_common_list,
        n_only_toxic_safe=n_only_toxic_list,
        n_only_nontoxic_safe=n_only_nontoxic_list,
        has_safe_diff=has_safe_diff_list,
    )




# =============================================================================
# SAFE filters (paper Step 4 lengths/counts caps after conserved-core enforcement)
# =============================================================================


def collect_only_fragment_token_lengths(df: pd.DataFrame) -> list[int]:
    """Collect per-token lengths from exclusive-toxic / exclusive-nontoxic fragment columns."""
    lengths: list[int] = []
    for col in (COL_ONLY_TOXIC_FRAG, COL_ONLY_NONTOXIC_FRAG):
        if col not in df.columns:
            continue
        for s in df[col].fillna(""):
            for p in str(s).split(SAFE_SEP):
                p = p.strip()
                if p:
                    lengths.append(len(p))
    return lengths


def tukey_upper_fence_from_values(values: list[int], *, min_points: int = 4) -> float | None:
    if len(values) < min_points:
        return None
    x = pd.Series(values, dtype=float)
    q1, q3 = float(x.quantile(0.25)), float(x.quantile(0.75))
    iqr = q3 - q1
    if iqr <= 0:
        return None
    return float(q3 + 1.5 * iqr)


def row_has_only_fragment_length_outlier(row: pd.Series, upper_fence: float) -> bool:
    """True when any exclusive fragment token exceeds the Tukey upper fence."""
    for col in (COL_ONLY_TOXIC_FRAG, COL_ONLY_NONTOXIC_FRAG):
        s = row.get(col)
        if s is None:
            continue
        t = str(s).strip()
        if not t or t.lower() == "nan":
            continue
        for p in t.split(SAFE_SEP):
            p = p.strip()
            if p and len(p) > upper_fence:
                return True
    return False


def series_tukey_upper_fence(series: pd.Series, *, fallback: float) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if len(x) < 4:
        return fallback
    q1, q3 = float(x.quantile(0.25)), float(x.quantile(0.75))
    iqr = q3 - q1
    if iqr <= 0:
        return fallback
    return float(q3 + 1.5 * iqr)


def _has_any_fragment_ge(s: Any, min_length: int) -> bool:
    """Fixed-threshold mode helper: any exclusive fragment length >= min_length."""
    if s is None:
        return False
    t = str(s).strip()
    if not t or t.lower() == "nan":
        return False
    parts = [p.strip() for p in t.split(SAFE_SEP) if p.strip()]
    return any(len(p) >= min_length for p in parts)


def apply_safe_pair_filters(
    df: pd.DataFrame,
    *,
    use_iqr: bool = True,
    fallback_frag_len_ge: int = 28,
    fallback_max_only_count: int = 4,
) -> pd.DataFrame:
    """Drop pairs whose exclusive fragments are unusually long or numerous.

    Cut-offs come from Tukey fences on the fragment token lengths and counts; when the sample
    is too small or degenerate for that, fallback_frag_len_ge and fallback_max_only_count are
    used instead. use_iqr=False keeps the older fixed thresholds.
    """
    for c in [
        "n_common_safe",
        "n_only_toxic_safe",
        "n_only_nontoxic_safe",
        COL_ONLY_TOXIC_FRAG,
        COL_ONLY_NONTOXIC_FRAG,
    ]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}. Run compare-safe step first.")

    n_start = len(df)
    df = df[df["n_common_safe"].ne(0)].copy()
    mask_both_zero = (df["n_only_nontoxic_safe"] == 0) & (df["n_only_toxic_safe"] == 0)
    df = df[~mask_both_zero].copy()
    n_after_core = len(df)

    if not use_iqr:
        mask_long = df.apply(
            lambda r: _has_any_fragment_ge(r.get(COL_ONLY_TOXIC_FRAG), fallback_frag_len_ge)
            or _has_any_fragment_ge(r.get(COL_ONLY_NONTOXIC_FRAG), fallback_frag_len_ge),
            axis=1,
        )
        df = df[~mask_long].copy()
        df = df[
            (df["n_only_toxic_safe"] <= fallback_max_only_count)
            & (df["n_only_nontoxic_safe"] <= fallback_max_only_count)
        ].copy()
        print(
            f"Filtered (fixed thresholds): {n_start} -> {len(df)} "
            f"(len>={fallback_frag_len_ge}, n_only<={fallback_max_only_count}); "
            f"after core rules: {n_after_core}"
        )
        return df

    lengths = collect_only_fragment_token_lengths(df)
    len_upper = tukey_upper_fence_from_values(lengths)
    if len_upper is None:
        len_upper = float(fallback_frag_len_ge - 1)

    mask_long = df.apply(lambda r: row_has_only_fragment_length_outlier(r, len_upper), axis=1)
    df = df[~mask_long].copy()
    n_after_len = len(df)

    upper_nt_only = series_tukey_upper_fence(
        df["n_only_toxic_safe"], fallback=float(fallback_max_only_count)
    )
    upper_nnt_only = series_tukey_upper_fence(
        df["n_only_nontoxic_safe"], fallback=float(fallback_max_only_count)
    )
    df = df[(df["n_only_toxic_safe"] <= upper_nt_only) & (df["n_only_nontoxic_safe"] <= upper_nnt_only)].copy()

    print(
        f"Filtered (IQR): {n_start} -> {len(df)} "
        f"| after shared/non-trivial: {n_after_core} "
        f"| fragment length upper (Tukey): {len_upper:.4g} "
        f"| n_only upper (toxic, nontoxic): {upper_nt_only:.4g}, {upper_nnt_only:.4g} "
        f"| after length step: {n_after_len}"
    )
    return df




# =============================================================================
# Physicochemical deltas
# =============================================================================


def get_descriptors(smiles: str) -> dict | None:
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return {
            "MW": Descriptors.ExactMolWt(mol),
            "logP": Crippen.MolLogP(mol),
            "TPSA": MolSurf.TPSA(mol),
            "HBD": Lipinski.NumHDonors(mol),
            "HBA": Lipinski.NumHAcceptors(mol),
            "RotB": Lipinski.NumRotatableBonds(mol),
        }
    except Exception:
        return None


def add_property_deltas(
    df: pd.DataFrame,
    toxic_col: str = "toxic_smiles",
    nontoxic_col: str = "nontoxic_smiles",
    cache: dict | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Append descriptor columns plus signed/absolute deltas between toxic and nontoxic SMILES."""
    if toxic_col not in df.columns or nontoxic_col not in df.columns:
        raise ValueError(f"CSV must have '{toxic_col}' and '{nontoxic_col}'.")

    n_pairs = len(df)
    cache = cache if cache is not None else {}
    names = PROPERTY_DESCRIPTOR_NAMES

    toxic_prop = {name: [float("nan")] * n_pairs for name in names}
    nontoxic_prop = {name: [float("nan")] * n_pairs for name in names}
    delta_signed = {name: [float("nan")] * n_pairs for name in names}
    delta_abs = {name: [float("nan")] * n_pairs for name in names}

    it = df.iterrows()
    if verbose:
        it = tqdm(it, total=n_pairs, desc="Property delta")

    for pos, (_, row) in enumerate(it):
        smi_t = row[toxic_col]
        smi_n = row[nontoxic_col]
        if smi_t not in cache:
            cache[smi_t] = get_descriptors(smi_t)
        if smi_n not in cache:
            cache[smi_n] = get_descriptors(smi_n)
        desc_t = cache.get(smi_t)
        desc_n = cache.get(smi_n)
        if desc_t is None or desc_n is None:
            continue
        for name in names:
            vt, vn = desc_t.get(name), desc_n.get(name)
            if vt is None or vn is None or not (np.isfinite(vt) and np.isfinite(vn)):
                continue
            vt_f, vn_f = float(vt), float(vn)
            toxic_prop[name][pos] = vt_f
            nontoxic_prop[name][pos] = vn_f
            d = vt_f - vn_f
            delta_signed[name][pos] = d
            delta_abs[name][pos] = abs(d)

    out = df.copy()
    for name in names:
        out[f"toxic_{name}"] = toxic_prop[name]
        out[f"nontoxic_{name}"] = nontoxic_prop[name]
        out[f"delta_{name}"] = delta_signed[name]
        out[f"delta_abs_{name}"] = delta_abs[name]
    return out




# =============================================================================
# Property outliers (Δ IQR pruning)
# =============================================================================

DEFAULT_OUTLIER_DESCRIPTORS = ["MW", "logP", "TPSA", "HBD", "HBA", "RotB"]


def _iqr_fences(x: pd.Series) -> Tuple[float, float, float, float, float]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return (float("nan"),) * 5
    q1 = float(x.quantile(0.25))
    q3 = float(x.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return q1, q3, iqr, lower, upper


def compute_delta_abs_thresholds(df: pd.DataFrame, delta_abs_cols: List[str]) -> Dict[str, Dict[str, float]]:
    thr: Dict[str, Dict[str, float]] = {}
    for c in delta_abs_cols:
        q1, q3, iqr, lower, upper = _iqr_fences(df[c])
        thr[c] = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_fence": lower,
            "upper_fence": upper,
        }
    return thr


def outlier_any_mask_for_thresholds(df: pd.DataFrame, thresholds: Dict[str, Dict[str, float]]) -> pd.Series:
    masks = []
    for c, t in thresholds.items():
        x = pd.to_numeric(df[c], errors="coerce")
        lower, upper = t["lower_fence"], t["upper_fence"]
        if np.isnan(lower) or np.isnan(upper):
            masks.append(pd.Series(False, index=df.index))
        else:
            masks.append((x < lower) | (x > upper))
    if not masks:
        return pd.Series(False, index=df.index)
    m = masks[0].copy()
    for mm in masks[1:]:
        m |= mm
    return m


def drop_property_outliers_dataframe(
    df: pd.DataFrame,
    descriptors: List[str] | None = None,
) -> pd.DataFrame:
    """Filter delta_abs_* outliers in-memory via Tukey fences (disk writes handled elsewhere)."""
    descriptors = descriptors or list(DEFAULT_OUTLIER_DESCRIPTORS)
    delta_abs_cols: List[str] = []
    for d in descriptors:
        c = f"delta_abs_{d}"
        if c in df.columns:
            delta_abs_cols.append(c)
        else:
            raise ValueError(f"Missing column: {c}")
    thresholds = compute_delta_abs_thresholds(df, delta_abs_cols)
    out_any = outlier_any_mask_for_thresholds(df, thresholds)
    return df.loc[~out_any].copy()




# =============================================================================
# orchestration
# =============================================================================


def pair_molecules(df: pd.DataFrame, n_workers: int | None = None) -> pd.DataFrame:
    """Step 1: pair toxic with non-toxic molecules inside each endpoint."""
    for col in ("dataset_name", "endpoint", "smiles", "label"):
        if col not in df.columns:
            raise ValueError(f"Input CSV must have a '{col}' column.")

    groups = []
    for (dataset, endpoint), g in df.groupby(["dataset_name", "endpoint"], sort=True):
        toxic = g.loc[g["label"].astype(int) == 1, "smiles"].astype(str).tolist()
        nontoxic = g.loc[g["label"].astype(int) == 0, "smiles"].astype(str).tolist()
        if toxic and nontoxic:
            groups.append((str(dataset), str(endpoint), toxic, nontoxic))

    rows: list[dict] = []
    n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(process_endpoint, *g): g for g in groups}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Endpoints"):
            dataset, endpoint = futures[fut][0], futures[fut][1]
            try:
                _, _, pairs, _ = fut.result()
            except Exception as err:
                tqdm.write(f"Error {dataset}/{endpoint}: {err}")
                continue
            for toxic_smiles, nontoxic_smiles in pairs:
                rows.append({
                    "dataset_name": dataset,
                    "endpoint": endpoint,
                    "toxic_smiles": toxic_smiles,
                    "nontoxic_smiles": nontoxic_smiles,
                })
    out = pd.DataFrame(rows, columns=["dataset_name", "endpoint", "toxic_smiles", "nontoxic_smiles"])
    return out.drop_duplicates()


def build_toxicity_cliffs(
    pairs: pd.DataFrame,
    *,
    use_iqr: bool = True,
    fallback_frag_len_ge: int = 28,
    fallback_max_only_count: int = 4,
    descriptors: List[str] | None = None,
    drop_property_outliers: bool = True,
) -> pd.DataFrame:
    """Steps 2-5: SAFE encoding, fragment comparison, and the two filtering stages."""
    smiles_to_safe, canon_to_safe = build_safe_mapping_from_pairs_with_encode(pairs)
    df = attach_safe_to_pairs(pairs, smiles_to_safe, canon_to_safe)
    df = enrich_pairs_with_safe_comparison(df)
    df = apply_safe_pair_filters(
        df,
        use_iqr=use_iqr,
        fallback_frag_len_ge=fallback_frag_len_ge,
        fallback_max_only_count=fallback_max_only_count,
    )
    df = add_property_deltas(df, cache={}, verbose=False)
    if drop_property_outliers:
        df = drop_property_outliers_dataframe(df, descriptors=descriptors)
    return finalize_toxicitycliff_pair_table(df)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="CSV with dataset_name, endpoint, smiles, label")
    parser.add_argument("--out", "-o", type=Path, default=Path("toxicitycliff.csv"))
    parser.add_argument("--workers", type=int, default=None, help="processes used for pairing")
    parser.add_argument("--keep-property-outliers", action="store_true",
                        help="skip the final property-delta filter")
    parser.add_argument("--no-iqr", action="store_true",
                        help="use fixed fragment cut-offs instead of Tukey fences")
    parser.add_argument("--fragment-length", type=int, default=28,
                        help="fixed cut-off on exclusive fragment length, with --no-iqr")
    parser.add_argument("--max-fragments", type=int, default=4,
                        help="fixed cut-off on the number of exclusive fragments, with --no-iqr")
    args = parser.parse_args()

    molecules = pd.read_csv(args.input)
    pairs = pair_molecules(molecules, n_workers=args.workers)
    print(f"{len(pairs):,} candidate pairs")

    cliffs = build_toxicity_cliffs(
        pairs,
        use_iqr=not args.no_iqr,
        fallback_frag_len_ge=args.fragment_length,
        fallback_max_only_count=args.max_fragments,
        drop_property_outliers=not args.keep_property_outliers,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cliffs.to_csv(args.out, index=False)
    print(f"{len(cliffs):,} toxicity cliffs -> {args.out}")


if __name__ == "__main__":
    main()
