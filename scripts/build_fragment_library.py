# scripts/build_fragment_library.py
# FragSynth v0.1.0 - Fragment library builder
# Input: unique_molecules.parquet
# Output: bundled fragments.parquet and attachment_points.parquet

import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS, PandasTools, Descriptors, rdMolDescriptors, inchi
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm
import json

tqdm.pandas()

# PATHS
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
BUNDLE_DIR = os.path.join(PROJECT_ROOT, "fragsynth", "data")
os.makedirs(BUNDLE_DIR, exist_ok=True)

INPUT_PARQUET = os.path.join(PROCESSED_DIR, "unique_molecules.parquet")
FRAGMENTS_PARQUET = os.path.join(BUNDLE_DIR, "fragments.parquet")
APS_PARQUET = os.path.join(BUNDLE_DIR, "attachment_points.parquet")

# CONSTANTS
ALLOWED_ELEMENTS = {"C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"}

# Forbidden reactive motifs (SMARTS)
FORBIDDEN_SMARTS = [
    "[N+]#N",           # diazonium
    "[C-]", "[N-]", "[O-]",  # charged species
    "[N+](=O)[O-]",     # nitro
    "[SX1]",            # sulfenyl halide
]

# ATTACHMENT POINT ONTOLOGY
class AttachmentPoint:
    def __init__(self, anchor_atom, bond_order):
        self.element = anchor_atom.GetSymbol()
        self.hybridization = str(anchor_atom.GetHybridization()).lower()
        self.is_aromatic = anchor_atom.GetIsAromatic()
        self.bond_order = bond_order
        self.ap_type = self._get_type()
        self.allowed_bond_orders = self._get_allowed_bond_orders()
        self.allowed_partners = self._get_allowed_partners()

    def _get_type(self):
        if self.is_aromatic:
            return f"{self.element}_aromatic"
        return f"{self.element}_{self.hybridization}"

    def _get_allowed_bond_orders(self):
        if self.ap_type == "C_sp3":
            return [1]
        if self.ap_type == "C_sp2":
            return [1, 2]
        if self.ap_type in ["C_aromatic", "N_aromatic"]:
            return [1]
        return [1]  # default

    def _get_allowed_partners(self):
        # Semantic compatibility for recombination
        if "aromatic" in self.ap_type or self.ap_type == "C_sp2":
            return ["Linker", "Side_chain", "Functional_group"]
        return ["Scaffold", "Linker", "Side_chain", "Cap"]

    def to_dict(self):
        return {
            "ap_type": self.ap_type,
            "element": self.element,
            "hybridization": self.hybridization,
            "is_aromatic": self.is_aromatic,
            "bond_order": self.bond_order,
            "allowed_bond_orders": self.allowed_bond_orders,
            "allowed_partners": self.allowed_partners
        }

# FRAGMENTATION
def fragment_molecule(mol: Chem.Mol):
    """Use BRICS decomposition and annotate attachment points."""
    if mol is None:
        return []

    fragments = []
    brics_frags = list(BRICS.BRICSDecompose(mol, returnMols=True))

    for frag in brics_frags:
        # BRICS uses dummy atoms (*) for attachment points
        editable = Chem.EditableMol(frag)
        ap_list = []

        for atom in frag.GetAtoms():
            if atom.GetSymbol() == "*":  # dummy atom
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:
                    real_atom = neighbors[0]
                    bond = frag.GetBondBetweenAtoms(atom.GetIdx(), real_atom.GetIdx())
                    bond_order = bond.GetBondTypeAsDouble()

                    ap = AttachmentPoint(real_atom, bond_order)
                    ap_list.append(ap.to_dict())

                    # Remove dummy atom
                    editable.RemoveAtom(atom.GetIdx())

        clean_mol = editable.GetMol()
        try:
            Chem.SanitizeMol(clean_mol)
        except:
            continue  # skip unsanitizable

        fragments.append({
            "mol": clean_mol,
            "attachment_points": ap_list
        })

    return fragments

# CLASSIFICATION
def classify_fragment(mol: Chem.Mol, aps: list):
    num_aps = len(aps)
    mw = Descriptors.MolWt(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    heavy_atoms = rdMolDescriptors.CalcNumHeavyAtoms(mol)

    if aromatic_rings >= 1 and heavy_atoms >= 5:
        return "Scaffold"
    if num_aps >= 2:
        return "Linker"
    if mw <= 150 and rings <= 1:
        return "Side_chain"
    if num_aps == 1 and mw <= 100:
        return "Cap"
    return "Functional_group"

# VALIDATION
def validate_fragment(mol: Chem.Mol, aps: list):
    if mol is None or len(aps) == 0:
        return False

    mw = Descriptors.MolWt(mol)
    if not (30 <= mw <= 350):
        return False

    # No forbidden motifs
    for smarts in FORBIDDEN_SMARTS:
        if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
            return False

    # Allowed elements
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ALLOWED_ELEMENTS:
            return False

    return True

# MAIN
def main():
    print("=== FragSynth Fragment Library Builder ===\n")

    if not os.path.exists(INPUT_PARQUET):
        raise FileNotFoundError(f"Run build_molecule_library.py first! Missing: {INPUT_PARQUET}")

    print("Loading unique molecules...")
    molecules = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded {len(molecules):,} molecules")

    PandasTools.AddMoleculeColumnToFrame(molecules, "canonical_smiles", "ROMol")

    fragment_records = []
    ap_records = []

    print("Fragmenting molecules...")
    for idx, row in tqdm(molecules.iterrows(), total=len(molecules)):
        frags = fragment_molecule(row["ROMol"])
        for frag_idx, frag_data in enumerate(frags):
            mol = frag_data["mol"]
            aps = frag_data["attachment_points"]

            if not validate_fragment(mol, aps):
                continue

            can_smiles = Chem.MolToSmiles(mol)
            inchikey = inchi.MolToInchiKey(mol)
            frag_class = classify_fragment(mol, aps)

            frag_id = f"{row['inchikey']}_{frag_idx}"

            frag_record = {
                "fragment_id": frag_id,
                "inchikey": inchikey,
                "canonical_smiles": can_smiles,
                "class": frag_class,
                "mw": Descriptors.MolWt(mol),
                "ring_count": rdMolDescriptors.CalcNumRings(mol),
                "num_attachment_points": len(aps),
                "provenance": row["provenance"]
            }
            fragment_records.append(frag_record)

            for ap_idx, ap in enumerate(aps):
                ap_record = {
                    "fragment_id": frag_id,
                    "ap_index": ap_idx,
                    "ap_type": ap["ap_type"],
                    "element": ap["element"],
                    "hybridization": ap["hybridization"],
                    "is_aromatic": ap["is_aromatic"],
                    "allowed_bond_orders": json.dumps(ap["allowed_bond_orders"]),
                    "allowed_partners": json.dumps(ap["allowed_partners"])
                }
                ap_records.append(ap_record)

    # Save
    fragments_df = pd.DataFrame(fragment_records)
    aps_df = pd.DataFrame(ap_records)

    print(f"\nGenerated {len(fragments_df):,} validated fragments")
    print("Class distribution:")
    print(fragments_df["class"].value_counts())

    fragments_df.to_parquet(FRAGMENTS_PARQUET, compression="gzip")
    aps_df.to_parquet(APS_PARQUET, compression="gzip")

    print(f"\nFragment library saved to {BUNDLE_DIR}")
    print("FragSynth is now ready for recombination and pip packaging!")

if __name__ == "__main__":
    main()