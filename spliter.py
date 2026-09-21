from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

RDLogger.DisableLog("rdApp.*")

TEST_FRACTION = 0.1
SMILES_COLUMN = "toxic_smiles"


def scaffold_of(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    return MurckoScaffoldSmiles(mol=mol) if mol is not None else None


def split_indices(smiles: list[str]) -> tuple[list[int], list[int]]:
    groups: dict[str, list[int]] = {}
    unparsed: list[int] = []
    for i, smi in enumerate(smiles):
        scaffold = scaffold_of(smi)
        if scaffold is None:
            unparsed.append(i)
        else:
            groups.setdefault(scaffold, []).append(i)

    # Largest groups first, ties broken by first row, so the split does not depend on dict order.
    ordered = sorted(groups.values(), key=lambda g: (len(g), g[0]), reverse=True)
    budget = TEST_FRACTION * len(smiles)
    train, test = list(unparsed), []
    for group in reversed(ordered):
        if len(test) + len(group) <= budget:
            test += group
        else:
            train += group
    return sorted(train), sorted(test)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", "-i", type=Path, required=True, help="toxicitycliff.csv")
    parser.add_argument("--out-dir", "-o", type=Path, required=True,
                        help="directory to write train.csv and test.csv into")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if SMILES_COLUMN not in df.columns:
        raise SystemExit(f"{args.input} has no {SMILES_COLUMN} column")

    train_idx, test_idx = split_indices(df[SMILES_COLUMN].tolist())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.iloc[train_idx].to_csv(args.out_dir / "train.csv", index=False)
    df.iloc[test_idx].to_csv(args.out_dir / "test.csv", index=False)
    print(f"train {len(train_idx)} rows, test {len(test_idx)} rows -> {args.out_dir}")


if __name__ == "__main__":
    main()
