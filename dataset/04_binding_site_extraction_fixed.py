#!/usr/bin/env python3
"""
EGFR Binding Site Extraction Script
Extracts and analyzes EGFR binding site features from crystal structure.

Based on your final approach:
- Uses EGFR structure 1M17 (with erlotinib)
- Extracts 20 key binding site residues within 5Å of ligand
- Calculates physicochemical properties for each residue
- Generates aggregate binding site features for model conditioning

This creates the structural foundation for target-aware molecule generation.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# BioPython imports
try:
    from Bio.PDB import PDBParser, NeighborSearch, PDBIO, Select
    from Bio.PDB.Polypeptide import protein_letters_3to1
    print("✅ BioPython imported successfully")
except ImportError:
    print("❌ BioPython not found. Please install: pip install biopython")
    exit(1)

# Create three_to_one function using the dictionary
def three_to_one(residue_name):
    """Convert three-letter amino acid code to one-letter code."""
    return protein_letters_3to1.get(residue_name, 'X')

class BindingSiteSelect(Select):
    """Custom selector for binding site residues."""
    
    def __init__(self, residue_list):
        self.residue_list = residue_list
        
    def accept_residue(self, residue):
        """Accept only residues in the binding site list."""
        return residue.get_full_id() in self.residue_list

def extract_binding_site_residues(pdb_file, ligand_name="AQ4", cutoff=5.0):
    """
    Extract binding site residues within specified cutoff of ligand.
    
    Args:
        pdb_file (str): Path to PDB file
        ligand_name (str): Ligand identifier (AQ4 = erlotinib in 1M17)
        cutoff (float): Distance cutoff in Angstroms
        
    Returns:
        list: List of binding site residues
    """
    
    print(f"🔄 Extracting binding site residues from {pdb_file}")
    print(f"🎯 Ligand: {ligand_name}, Cutoff: {cutoff}Å")
    
    # Parse PDB structure
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('EGFR', pdb_file)
    except Exception as e:
        print(f"❌ Error parsing PDB file: {e}")
        return None
    
    # Find the ligand
    ligand_atoms = []
    ligand_residue = None
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() == ligand_name:
                    ligand_residue = residue
                    ligand_atoms = [atom for atom in residue.get_atoms()]
                    print(f"✅ Found ligand {ligand_name} in chain {chain.id}")
                    break
            if ligand_atoms:
                break
        if ligand_atoms:
            break
    
    if not ligand_atoms:
        print(f"❌ Ligand {ligand_name} not found in structure!")
        return None
    
    print(f"🧪 Ligand has {len(ligand_atoms)} atoms")
    
    # Get all protein atoms for neighbor search
    protein_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                # Skip ligands, water, and other heteroatoms
                if residue.get_id()[0] == ' ':  # Standard amino acid
                    for atom in residue.get_atoms():
                        protein_atoms.append(atom)
    
    print(f"🧬 Found {len(protein_atoms)} protein atoms")
    
    # Create neighbor search object
    neighbor_search = NeighborSearch(protein_atoms)
    
    # Find binding site residues
    binding_site_residues = set()
    
    for ligand_atom in ligand_atoms:
        # Find protein atoms within cutoff
        nearby_atoms = neighbor_search.search(ligand_atom.get_coord(), cutoff)
        
        for atom in nearby_atoms:
            residue = atom.get_parent()
            binding_site_residues.add(residue)
    
    # Convert to list and sort
    binding_site_list = list(binding_site_residues)
    binding_site_list.sort(key=lambda x: (x.get_parent().id, x.get_id()[1]))
    
    print(f"✅ Found {len(binding_site_list)} binding site residues")
    
    return binding_site_list, ligand_residue

def calculate_residue_properties(residue):
    """
    Calculate physicochemical properties for a residue.
    
    Args:
        residue: Bio.PDB residue object
        
    Returns:
        dict: Residue properties
    """
    
    # Amino acid property lookup tables
    hydrophobicity_scale = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }
    
    charge_scale = {
        'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
        'Q': 0, 'E': -1, 'G': 0, 'H': 0, 'I': 0,
        'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
        'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
    }
    
    h_donor_scale = {
        'A': 0, 'R': 1, 'N': 1, 'D': 0, 'C': 0,
        'Q': 1, 'E': 0, 'G': 0, 'H': 1, 'I': 0,
        'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
        'S': 1, 'T': 1, 'W': 1, 'Y': 1, 'V': 0
    }
    
    h_acceptor_scale = {
        'A': 0, 'R': 0, 'N': 1, 'D': 1, 'C': 0,
        'Q': 1, 'E': 1, 'G': 0, 'H': 1, 'I': 0,
        'L': 0, 'K': 0, 'M': 0, 'F': 0, 'P': 0,
        'S': 1, 'T': 1, 'W': 0, 'Y': 1, 'V': 0
    }
    
    size_scale = {
        'A': 1, 'R': 4, 'N': 2, 'D': 2, 'C': 2,
        'Q': 3, 'E': 3, 'G': 1, 'H': 3, 'I': 3,
        'L': 3, 'K': 3, 'M': 3, 'F': 4, 'P': 2,
        'S': 1, 'T': 2, 'W': 5, 'Y': 4, 'V': 2
    }
    
    # Get residue information
    chain_id = residue.get_parent().id
    res_id = residue.get_id()[1]
    res_name = residue.get_resname()
    
    # Convert to one-letter code
    try:
        one_letter = three_to_one(res_name)
    except KeyError:
        one_letter = 'X'  # Unknown residue
    
    # Calculate center of mass
    atoms = list(residue.get_atoms())
    if atoms:
        coords = np.array([atom.get_coord() for atom in atoms])
        center = np.mean(coords, axis=0)
        x, y, z = center
    else:
        x, y, z = 0.0, 0.0, 0.0
    
    # Get properties from lookup tables
    hydrophobicity = hydrophobicity_scale.get(one_letter, 0.0)
    charge = charge_scale.get(one_letter, 0)
    h_donor = h_donor_scale.get(one_letter, 0)
    h_acceptor = h_acceptor_scale.get(one_letter, 0)
    size = size_scale.get(one_letter, 1)
    
    # Categorize residue
    if one_letter in ['R', 'K', 'H']:
        category = 'Charged'
    elif one_letter in ['D', 'E']:
        category = 'Charged'
    elif one_letter in ['S', 'T', 'N', 'Q', 'Y']:
        category = 'Polar'
    elif one_letter in ['A', 'V', 'I', 'L', 'M', 'F', 'W', 'P']:
        category = 'Hydrophobic'
    elif one_letter == 'G':
        category = 'Glycine'
    elif one_letter == 'C':
        category = 'Cysteine'
    else:
        category = 'Other'
    
    return {
        'chain': chain_id,
        'residue_id': res_id,
        'residue_name': res_name,
        'one_letter_code': one_letter,
        'x': float(x),
        'y': float(y),
        'z': float(z),
        'hydrophobicity': float(hydrophobicity),
        'charge': int(charge),
        'h_donor': int(h_donor),
        'h_acceptor': int(h_acceptor),
        'size': int(size),
        'category': category
    }

def analyze_binding_site(pdb_file, output_dir):
    """
    Complete binding site analysis workflow.
    
    Args:
        pdb_file (str): Path to EGFR PDB file
        output_dir (str): Output directory for results
        
    Returns:
        tuple: (binding_site_df, aggregate_features)
    """
    
    print("🔄 Starting EGFR binding site analysis...")
    print(f"📂 PDB file: {pdb_file}")
    print(f"📁 Output directory: {output_dir}")
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Extract binding site residues
    print("\n⿡ Extracting binding site residues...")
    
    result = extract_binding_site_residues(pdb_file, ligand_name="AQ4", cutoff=5.0)
    if result is None:
        print("❌ Failed to extract binding site!")
        return None, None
    
    binding_site_residues, ligand_residue = result
    
    # Step 2: Calculate residue properties
    print("\n⿢ Calculating residue properties...")
    
    residue_data = []
    for residue in binding_site_residues:
        properties = calculate_residue_properties(residue)
        residue_data.append(properties)
    
    df_residues = pd.DataFrame(residue_data)
    print(f"✅ Calculated properties for {len(df_residues)} residues")
    
    # Step 3: Generate aggregate binding site features
    print("\n⿣ Generating aggregate binding site features...")
    
    # Calculate aggregate statistics
    aggregate_features = {
        'total_residues': len(df_residues),
        'average_hydrophobicity': float(df_residues['hydrophobicity'].mean()),
        'net_charge': int(df_residues['charge'].sum()),
        'total_h_donors': int(df_residues['h_donor'].sum()),
        'total_h_acceptors': int(df_residues['h_acceptor'].sum()),
        'average_size': float(df_residues['size'].mean()),
        'center_of_mass': {
            'x': float(df_residues['x'].mean()),
            'y': float(df_residues['y'].mean()),
            'z': float(df_residues['z'].mean())
        },
        'residue_composition': df_residues['category'].value_counts().to_dict(),
        'hydrophobicity_distribution': {
            'min': float(df_residues['hydrophobicity'].min()),
            'max': float(df_residues['hydrophobicity'].max()),
            'std': float(df_residues['hydrophobicity'].std())
        }
    }
    
    # Step 4: Save detailed residue data
    print("\n⿤ Saving binding site data...")
    
    # Save residue features
    residue_file = Path(output_dir) / "binding_site_features.csv"
    df_residues.to_csv(residue_file, index=False)
    print(f"💾 Saved residue features: {residue_file}")
    
    # Save aggregate features
    aggregate_file = Path(output_dir) / "binding_site_aggregate_features.json"
    with open(aggregate_file, 'w') as f:
        json.dump(aggregate_features, f, indent=2)
    print(f"💾 Saved aggregate features: {aggregate_file}")
    
    # Save residue list (simple format)
    residue_list_file = Path(output_dir) / "1M17_binding_site_residues.txt"
    with open(residue_list_file, 'w') as f:
        f.write("# EGFR Binding Site Residues (within 5Å of erlotinib)\n")
        f.write("# Chain:ResID:ResName:Distance\n")
        for _, row in df_residues.iterrows():
            f.write(f"Chain: {row['chain']}, ResID: {row['residue_id']}, "
                   f"ResName: {row['residue_name']}, OneLetterCode: {row['one_letter_code']}\n")
    print(f"📝 Saved residue list: {residue_list_file}")
    
    # Step 5: Create clean PDB files
    print("\n⿥ Creating clean PDB files...")
    
    try:
        # Parse original structure
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('EGFR', pdb_file)
        io = PDBIO()
        
        # Save clean protein structure (no ligands, water)
        class ProteinSelect(Select):
            def accept_residue(self, residue):
                return residue.get_id()[0] == ' '  # Only standard amino acids
        
        io.set_structure(structure)
        clean_protein_file = Path(output_dir) / "1M17_protein_clean.pdb"
        io.save(str(clean_protein_file), ProteinSelect())
        print(f"🧬 Saved clean protein: {clean_protein_file}")
        
        # Save binding site only
        binding_site_ids = []
        for residue in binding_site_residues:
            binding_site_ids.append(residue.get_full_id())
        
        binding_site_selector = BindingSiteSelect(binding_site_ids)
        binding_site_file = Path(output_dir) / "1M17_binding_site.pdb"
        io.save(str(binding_site_file), binding_site_selector)
        print(f"🎯 Saved binding site: {binding_site_file}")
        
    except Exception as e:
        print(f"⚠ Warning: Could not create clean PDB files: {e}")
    
    # Step 6: Generate analysis summary
    print("\n⿦ Generating analysis summary...")
    
    summary = {
        'binding_site_analysis': {
            'pdb_file': str(pdb_file),
            'ligand_name': 'AQ4 (erlotinib)',
            'cutoff_distance': 5.0,
            'total_binding_site_residues': len(df_residues),
            'residue_composition': aggregate_features['residue_composition'],
            'physicochemical_properties': {
                'average_hydrophobicity': aggregate_features['average_hydrophobicity'],
                'net_charge': aggregate_features['net_charge'],
                'h_bond_donors': aggregate_features['total_h_donors'],
                'h_bond_acceptors': aggregate_features['total_h_acceptors'],
                'average_residue_size': aggregate_features['average_size']
            },
            'key_binding_residues': df_residues[['chain', 'residue_id', 'residue_name', 'category']].to_dict('records')
        },
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Save summary
    summary_file = Path(output_dir) / "binding_site_analysis_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"📊 Saved analysis summary: {summary_file}")
    
    # Print detailed results
    print("\n" + "="*60)
    print("🎉 EGFR Binding Site Analysis Complete!")
    print("="*60)
    print(f"🎯 Binding site residues: {len(df_residues)}")
    print(f"💧 Average hydrophobicity: {aggregate_features['average_hydrophobicity']:.2f}")
    print(f"⚡ Net charge: {aggregate_features['net_charge']}")
    print(f"🔗 H-bond donors: {aggregate_features['total_h_donors']}")
    print(f"🔗 H-bond acceptors: {aggregate_features['total_h_acceptors']}")
    print("\n📊 Residue composition:")
    for category, count in aggregate_features['residue_composition'].items():
        percentage = (count / len(df_residues)) * 100
        print(f"   {category}: {count} ({percentage:.1f}%)")
    
    print("\n🧬 Key binding residues:")
    key_residues = df_residues.sort_values('hydrophobicity', ascending=False).head(10)
    for _, residue in key_residues.iterrows():
        print(f"   {residue['chain']}:{residue['residue_id']}:{residue['residue_name']} "
              f"({residue['category']}, hydrophobicity: {residue['hydrophobicity']:.1f})")
    
    print("="*60)
    
    return df_residues, aggregate_features

def validate_binding_site_data(output_dir):
    """
    Validate that binding site extraction completed successfully.
    
    Args:
        output_dir (str): Directory to validate
        
    Returns:
        bool: True if validation passes
    """
    
    print("\n🔍 Validating binding site data...")
    
    required_files = [
        "binding_site_features.csv",
        "binding_site_aggregate_features.json",
        "1M17_binding_site_residues.txt",
        "binding_site_analysis_summary.json"
    ]
    
    all_good = True
    
    for filename in required_files:
        filepath = Path(output_dir) / filename
        if filepath.exists():
            print(f"✅ {filename} - OK")
        else:
            print(f"❌ {filename} - MISSING")
            all_good = False
    
    # Check data content
    try:
        features_file = Path(output_dir) / "binding_site_features.csv"
        if features_file.exists():
            df = pd.read_csv(features_file)
            if len(df) >= 15:  # Should have ~20 residues
                print(f"✅ Binding site features: {len(df)} residues - OK")
            else:
                print(f"⚠ Binding site features: Only {len(df)} residues (expected ~20)")
                
            # Check required columns
            required_cols = ['chain', 'residue_id', 'residue_name', 'hydrophobicity', 'charge']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if not missing_cols:
                print("✅ All required columns present - OK")
            else:
                print(f"❌ Missing columns: {missing_cols}")
                all_good = False
        
    except Exception as e:
        print(f"❌ Error validating data content: {e}")
        all_good = False
    
    if all_good:
        print("✅ Binding site data validation PASSED!")
    else:
        print("❌ Binding site data validation FAILED!")
    
    return all_good

if __name__ == "__main__":
    # Set paths
    pdb_file = os.path.join(os.getcwd(), "raw_data", "1M17.pdb")
    output_dir = os.path.join(os.getcwd(), "binding_site_data")
    
    print("🚀 BindingForge EGFR Binding Site Extraction")
    print("="*50)
    
    # Check if PDB file exists
    if not Path(pdb_file).exists():
        print(f"❌ PDB file not found: {pdb_file}")
        print("Please run 02_chembl_data_collection.py first to download the structure.")
        exit(1)
    
    # Analyze binding site
    df_residues, aggregate_features = analyze_binding_site(pdb_file, output_dir)
    
    if df_residues is not None and aggregate_features is not None:
        # Validate results
        if validate_binding_site_data(output_dir):
            print("\n✅ Binding site extraction completed successfully!")
            print("📁 Check the binding_site_data/ directory for:")
            print("   - binding_site_features.csv (detailed residue properties)")
            print("   - binding_site_aggregate_features.json (aggregate features)")
            print("   - 1M17_binding_site_residues.txt (residue list)")
            print("   - 1M17_protein_clean.pdb (clean protein structure)")
            print("   - 1M17_binding_site.pdb (binding site only)")
            
            print(f"\n🎯 Key Results:")
            print(f"   - {len(df_residues)} binding site residues identified")
            print(f"   - Average hydrophobicity: {aggregate_features['average_hydrophobicity']:.2f}")
            print(f"   - Net charge: {aggregate_features['net_charge']}")
            print(f"   - H-bond capacity: {aggregate_features['total_h_donors']} donors, {aggregate_features['total_h_acceptors']} acceptors")
            
            print("\n🚀 Next step: Run 05_albumin_binding_analysis.py")
        else:
            print("❌ Validation failed!")
            exit(1)
    else:
        print("❌ Binding site extraction failed!")
        exit(1)
