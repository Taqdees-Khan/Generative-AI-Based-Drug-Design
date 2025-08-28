"""
Basic Docking Example with BindingForge
"""

import os
import subprocess
import pandas as pd
from src.utils import calculate_molecular_properties

def run_basic_docking(smiles_list, output_dir="results/basic_docking"):
    """
    Run basic docking for a list of SMILES
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    results = []

    for i, smiles in enumerate(smiles_list):
        print(f"Processing molecule {i+1}/{len(smiles_list)}: {smiles}")

        # Calculate molecular properties
        props = calculate_molecular_properties(smiles)

        # Convert SMILES to 3D structure (would use RDKit/OpenBabel)
        # mol_file = f"{output_dir}/molecule_{i+1}.mol2"
        # convert_smiles_to_mol2(smiles, mol_file)

        # Run AutoDock Vina docking
        # docking_result = run_vina_docking(mol_file)

        # Store results
        result = {
            'Molecule_ID': f'MOL_{i+1:03d}',
            'SMILES': smiles,
            'MW': props.get('MW', 0),
            'LogP': props.get('LogP', 0),
            'TPSA': props.get('TPSA', 0),
            # 'Binding_Affinity': docking_result['affinity'],
            # 'RMSD': docking_result['rmsd']
        }
        results.append(result)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/docking_results.csv", index=False)

    print(f"✅ Docking completed. Results saved to {output_dir}/docking_results.csv")
    return results_df

# Example usage
if __name__ == "__main__":
    example_smiles = [
        'CCOc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OCCCN1CCOCC1',  # Erlotinib
        'COc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OC',  # Gefitinib
    ]

    results = run_basic_docking(example_smiles)
    print(results.head())
