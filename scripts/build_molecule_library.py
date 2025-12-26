# scripts/build_molecule_library.py
# FragSynth v0.1.0 - Molecule standardization and deduplication pipeline
# Processes COCONUT and ChEMBL CSVs → unique, validated molecules

import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools, Descriptors, rdMolDescriptors, inchi
from rdkit.Chem.MolStandardize import rdMolStandardize  
from tqdm import tqdm
import numpy as np

tqdm.pandas()

# PATHS
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

COCONUT_CSV = os.path.join(RAW_DATA_DIR, "coconut_csv.csv")          # columns: identifier, canonical_smiles
CHEMBL_CSV = os.path.join(RAW_DATA_DIR, "Chembl_csv.csv")            # columns: ChEMBL ID, Smiles
OUTPUT_PARQUET = os.path.join(PROCESSED_DATA_DIR, "unique_molecules.parquet")

# CONSTANTS
ALLOWED_ELEMENTS = {"C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"}
MW_MIN, MW_MAX = 100, 700

# STANDARDIZATION
def standardize_smiles(smiles: str) -> tuple[str | None, str | None]:
    """
    Full standardization per FragSynth spec (compatible with RDKit 2024+).
    Returns (canonical_smiles, inchikey) or (None, None) if parsing fails.
    """
    if pd.isna(smiles):
        return None, None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    # Keep largest fragment (remove salts/solvents)
    frags = Chem.GetMolFrags(mol, asMols=True)
    mol = max(frags, key=lambda m: m.GetNumAtoms())

    # Neutralize charges
    uncharger = rdMolStandardize.Uncharger()
    mol = uncharger.uncharge(mol)

    # Canonical tautomer
    enumerator = rdMolStandardize.TautomerEnumerator()
    mol = enumerator.Canonicalize(mol)

    # Remove explicit Hs and Kekulize
    mol = Chem.RemoveHs(mol)
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except:
        pass  # Some molecules resist Kekulization

    # Generate canonical SMILES and InChIKey
    can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True)
    inchikey = inchi.MolToInchiKey(mol)

    return can_smiles, inchikey

# VALIDATION
def validate_molecule(mol: Chem.Mol) -> bool:
    """Apply strict structural constraints per project spec."""
    if mol is None:
        return False

    # Single connected component
    if rdMolDescriptors.CalcNumFrags(mol) > 1:
        return False

    # Allowed elements only
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ALLOWED_ELEMENTS:
            return False

    # Molecular weight
    mw = Descriptors.MolWt(mol)
    if not (MW_MIN <= mw <= MW_MAX):
        return False

    # Heavy atoms ≤ 100
    if rdMolDescriptors.CalcNumHeavyAtoms(mol) > 100:
        return False

    # Rings ≤ 12
    if rdMolDescriptors.CalcNumRings(mol) > 12:
        return False

    # Final sanitization check
    try:
        Chem.SanitizeMol(mol)
    except:
        return False

    return True

# PROCESS ONE DATASET
def process_dataset(csv_path: str, id_col: str, smiles_col: str, source_name: str) -> pd.DataFrame:
    print(f"\nLoading and processing {source_name}...")
    df = pd.read_csv(csv_path)

    print(f"  Original entries: {len(df):,}")

    # Standardize
    print("  Standardizing SMILES and generating InChIKeys...")
    standardized = df[smiles_col].progress_apply(standardize_smiles)
    df["canonical_smiles"] = [x[0] for x in standardized]
    df["inchikey"] = [x[1] for x in standardized]

    # Drop standardization failures
    before = len(df)
    df = df.dropna(subset=["inchikey"]).copy()
    print(f"  Failed standardization: {before - len(df):,} → Valid: {len(df):,}")

    # Add RDKit Mol column for validation
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol="canonical_smiles", molCol="ROMol")

    # Validate
    print("  Validating molecular constraints...")
    df["valid"] = df["ROMol"].progress_apply(validate_molecule)
    df = df[df["valid"]].copy()
    print(f"  Invalid after constraints: {len(df):,} remaining")

    # Keep essential columns
    df = df[["inchikey", "canonical_smiles", "ROMol"]].copy()
    df["source"] = source_name
    df["source_id"] = pd.read_csv(csv_path)[id_col]  # Re-read ID column to align

    return df

# MAIN
def main():
    print("=== FragSynth Molecule Library Builder ===\n")

    if not os.path.exists(COCONUT_CSV):
        raise FileNotFoundError(f"COCONUT CSV not found: {COCONUT_CSV}")
    if not os.path.exists(CHEMBL_CSV):
        raise FileNotFoundError(f"ChEMBL CSV not found: {CHEMBL_CSV}")

    # Process both datasets
    coconut_df = process_dataset(COCONUT_CSV, "identifier", "canonical_smiles", "COCONUT")
    chembl_df = process_dataset(CHEMBL_CSV, "ChEMBL ID", "Smiles", "ChEMBL")

    # Combine
    combined = pd.concat([coconut_df, chembl_df], ignore_index=True)
    print(f"\nTotal before deduplication: {len(combined):,}")

    # Deduplicate by InChIKey, track provenance
    provenance = combined.groupby("inchikey")["source"].apply(lambda x: "/".join(sorted(set(x))))
    unique = combined.drop_duplicates(subset="inchikey").copy()
    unique["provenance"] = provenance.values

    # Final cleanup
    final_df = unique[["inchikey", "canonical_smiles", "provenance"]].copy()
    final_df = final_df.reset_index(drop=True)

    print(f"\nFinal unique validated molecules: {len(final_df):,}")
    print("\nProvenance breakdown:")
    print(final_df["provenance"].value_counts())

    # Save
    final_df.to_parquet(OUTPUT_PARQUET, compression="gzip")
    print(f"\nSaved to: {OUTPUT_PARQUET}")

    print("\nMolecule library built successfully! Ready for fragmentation.")

if __name__ == "__main__":
    main()