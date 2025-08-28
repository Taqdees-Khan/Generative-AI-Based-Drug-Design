"""
Evaluation metrics for BindingForge
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from typing import List, Dict, Tuple

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics for binding affinity prediction"""

    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) == 0:
        return {}

    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
        'MAE': mean_absolute_error(y_true_clean, y_pred_clean),
        'R2': r2_score(y_true_clean, y_pred_clean),
        'Pearson_r': pearsonr(y_true_clean, y_pred_clean)[0],
        'Spearman_rho': spearmanr(y_true_clean, y_pred_clean)[0]
    }

    return metrics

def evaluate_molecular_diversity(smiles_list: List[str]) -> Dict[str, float]:
    """Evaluate diversity of generated molecules"""
    from rdkit import Chem
    from rdkit.Chem import DataStructs
    from rdkit.Chem.Fingerprints import FingerprintMols

    valid_mols = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_mols.append(mol)

    if len(valid_mols) < 2:
        return {'validity': len(valid_mols) / len(smiles_list)}

    # Calculate pairwise Tanimoto similarities
    fps = [FingerprintMols.FingerprintMol(mol) for mol in valid_mols]
    similarities = []

    for i in range(len(fps)):
        for j in range(i+1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarities.append(sim)

    diversity_metrics = {
        'validity': len(valid_mols) / len(smiles_list),
        'uniqueness': len(set(smiles_list)) / len(smiles_list),
        'mean_similarity': np.mean(similarities),
        'diversity': 1 - np.mean(similarities)
    }

    return diversity_metrics

def docking_validation_metrics(rmsd_values: List[float], threshold: float = 2.0) -> Dict[str, float]:
    """Calculate docking validation metrics"""

    rmsd_array = np.array(rmsd_values)

    metrics = {
        'mean_rmsd': np.mean(rmsd_array),
        'median_rmsd': np.median(rmsd_array),
        'success_rate': np.sum(rmsd_array <= threshold) / len(rmsd_array),
        'best_rmsd': np.min(rmsd_array),
        'worst_rmsd': np.max(rmsd_array)
    }

    return metrics
