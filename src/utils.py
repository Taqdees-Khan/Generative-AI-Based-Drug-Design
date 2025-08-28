"""
Utility functions for BindingForge
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
from typing import List, Dict, Optional

def smiles_to_tokens(smiles: str, vocab: Dict[str, int]) -> List[int]:
    """Convert SMILES string to token sequence"""
    tokens = []
    for char in smiles:
        if char in vocab:
            tokens.append(vocab[char])
        else:
            tokens.append(vocab['<UNK>'])
    return tokens

def calculate_molecular_properties(smiles: str) -> Dict[str, float]:
    """Calculate molecular descriptors from SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    properties = {
        'MW': Descriptors.MolWt(mol),
        'LogP': Crippen.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'HBD': Descriptors.NumHDonors(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'RB': Descriptors.NumRotatableBonds(mol)
    }

    return properties

def prepare_target_features(pdb_file: str) -> np.ndarray:
    """Extract target protein features from PDB file"""
    # Implementation for target feature extraction
    # This would include binding site characterization
    pass

def validate_drug_likeness(smiles: str) -> bool:
    """Check if molecule satisfies Lipinski's Rule of Five"""
    props = calculate_molecular_properties(smiles)

    if not props:
        return False

    lipinski_violations = 0
    if props['MW'] > 500:
        lipinski_violations += 1
    if props['LogP'] > 5:
        lipinski_violations += 1
    if props['HBD'] > 5:
        lipinski_violations += 1
    if props['HBA'] > 10:
        lipinski_violations += 1

    return lipinski_violations <= 1
