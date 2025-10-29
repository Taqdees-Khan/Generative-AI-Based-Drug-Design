#!/usr/bin/env python3
"""
Comprehensive Molecule Validator
Author: TAQDEES
Description: Validate molecular accuracy and novelty through multiple methods
"""

import os
import pandas as pd
import numpy as np
import requests
import time
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED, Crippen, Lipinski
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem import DataStructs
    from rdkit.Chem.Fingerprints import FingerprintMols
    RDKIT_AVAILABLE = True
    print("✅ RDKit available for validation")
except ImportError:
    print("❌ RDKit not available. Please install: conda install -c conda-forge rdkit")
    RDKIT_AVAILABLE = False
    exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MoleculeValidator:
    """Comprehensive molecule validation class"""
    
    def __init__(self):
        self.known_egfr_inhibitors = self.load_known_egfr_inhibitors()
        self.training_molecules = self.load_training_molecules()
        
    def load_known_egfr_inhibitors(self):
        """Load known EGFR inhibitors for comparison"""
        known_inhibitors = {
            # FDA-approved EGFR inhibitors
            'Erlotinib': 'COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCOCCN',
            'Gefitinib': 'COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1',
            'Afatinib': 'CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OCCCN',
            'Osimertinib': 'COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc1nccc(n1)c1c(C)cccc1Cl',
            'Dacomitinib': 'CN(C)C/C=C/C(=O)Nc1ccc2ncnc(Nc3cccc(Cl)c3)c2c1',
            'Lapatinib': 'CS(=O)(=O)CCNCc1oc(cc1)c1ccc(Nc2nccc(n2)c2ccc(F)cc2OCC2CCCO2)cc1',
            
            # Common research compounds
            'AG1478': 'CNc1ncnc2c1ccc(Br)c2Nc1cccc(Cl)c1',
            'PD153035': 'Brc1cc2c(Nc3cccc(Br)c3)ncnc2s1',
            'Canertinib': 'CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1O',
            
            # Basic scaffolds
            'Quinazoline_core': 'c1ccc2ncncc2c1',
            'Anilinoquinazoline': 'c1ccc(Nc2ncnc3ccccc23)cc1',
        }
        return known_inhibitors
    
    def load_training_molecules(self):
        """Load training molecules if available"""
        try:
            # Try to load your processed training data
            df = pd.read_csv('processed_data/egfr_type1_filtered.csv', on_bad_lines='skip')
            if 'smiles' in df.columns:
                return df['smiles'].dropna().tolist()
            return []
        except:
            return []

    def validate_structure(self, smiles):
        """Validate molecular structure"""
        results = {
            'smiles': smiles,
            'is_valid': False,
            'mol_object': None,
            'error_message': None,
            'num_atoms': 0,
            'num_bonds': 0,
            'has_metals': False,
            'has_unusual_atoms': False,
            'is_organic': True
        }
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                results['error_message'] = "Invalid SMILES - RDKit could not parse"
                return results
            
            results['is_valid'] = True
            results['mol_object'] = mol
            results['num_atoms'] = mol.GetNumAtoms()
            results['num_bonds'] = mol.GetNumBonds()
            
            # Check for unusual atoms
            atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
            organic_atoms = {'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H'}
            metal_atoms = {'Fe', 'Cu', 'Zn', 'Mg', 'Ca', 'Mn', 'Co', 'Ni'}
            
            unusual_atoms = set(atom_symbols) - organic_atoms - metal_atoms
            results['has_unusual_atoms'] = len(unusual_atoms) > 0
            results['has_metals'] = len(set(atom_symbols) & metal_atoms) > 0
            results['is_organic'] = len(unusual_atoms) == 0 and not results['has_metals']
            
            if unusual_atoms:
                results['error_message'] = f"Contains unusual atoms: {unusual_atoms}"
            
        except Exception as e:
            results['error_message'] = f"RDKit error: {str(e)}"
        
        return results

    def calculate_similarity(self, smiles1, smiles2):
        """Calculate Tanimoto similarity between two molecules"""
        try:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
            
            if mol1 is None or mol2 is None:
                return 0.0
            
            fp1 = FingerprintMols.FingerprintMol(mol1)
            fp2 = FingerprintMols.FingerprintMol(mol2)
            
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        except:
            return 0.0

    def check_novelty(self, smiles):
        """Check novelty against known compounds"""
        results = {
            'is_novel': True,
            'max_similarity_known': 0.0,
            'most_similar_known': None,
            'max_similarity_training': 0.0,
            'most_similar_training': None,
            'is_exact_match': False,
            'exact_match_compound': None
        }
        
        # Check against known EGFR inhibitors
        for name, known_smiles in self.known_egfr_inhibitors.items():
            similarity = self.calculate_similarity(smiles, known_smiles)
            
            if similarity > results['max_similarity_known']:
                results['max_similarity_known'] = similarity
                results['most_similar_known'] = name
            
            # Check for exact match
            if similarity > 0.99:  # Very high similarity indicates likely exact match
                results['is_exact_match'] = True
                results['exact_match_compound'] = name
        
        # Check against training molecules
        for train_smiles in self.training_molecules[:1000]:  # Limit for performance
            similarity = self.calculate_similarity(smiles, train_smiles)
            
            if similarity > results['max_similarity_training']:
                results['max_similarity_training'] = similarity
                results['most_similar_training'] = train_smiles
            
            if similarity > 0.99:
                results['is_exact_match'] = True
                results['exact_match_compound'] = "Training molecule"
        
        # Define novelty threshold (typically 0.7-0.8 for drug discovery)
        novelty_threshold = 0.8
        results['is_novel'] = (results['max_similarity_known'] < novelty_threshold and 
                              results['max_similarity_training'] < novelty_threshold)
        
        return results

    def get_scaffold(self, smiles):
        """Get Murcko scaffold"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold)
        except:
            return None

    def check_drug_likeness(self, smiles):
        """Check drug-likeness using multiple criteria"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            results = {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Crippen.MolLogP(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'tpsa': Descriptors.TPSA(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'aromatic_rings': Descriptors.NumAromaticRings(mol),
                'qed': QED.qed(mol),
                'num_heavy_atoms': Descriptors.HeavyAtomCount(mol)
            }
            
            # Lipinski Rule of Five
            results['lipinski_violations'] = sum([
                results['molecular_weight'] > 500,
                results['logp'] > 5,
                results['hbd'] > 5,
                results['hba'] > 10
            ])
            
            # Additional drug-like filters
            results['passes_lipinski'] = results['lipinski_violations'] == 0
            results['passes_ghose'] = (160 <= results['molecular_weight'] <= 480 and
                                     -0.4 <= results['logp'] <= 5.6 and
                                     40 <= results['num_heavy_atoms'] <= 70)
            results['passes_veber'] = (results['rotatable_bonds'] <= 10 and
                                     results['tpsa'] <= 140)
            results['is_drug_like'] = (results['qed'] >= 0.5 and 
                                     results['passes_lipinski'] and
                                     results['passes_veber'])
            
            return results
        except:
            return None

    def search_chembl(self, smiles, max_results=5):
        """Search ChEMBL database for similar compounds"""
        results = {
            'found_in_chembl': False,
            'chembl_compounds': [],
            'search_error': None
        }
        
        try:
            # ChEMBL similarity search (you'd need ChEMBL web services)
            # This is a placeholder - would require actual ChEMBL API integration
            logger.info(f"ChEMBL search not implemented - would search for: {smiles}")
            results['search_error'] = "ChEMBL search not implemented"
            
        except Exception as e:
            results['search_error'] = str(e)
        
        return results

def validate_molecules_comprehensive(csv_file):
    """Comprehensive validation of molecules from CSV"""
    
    logger.info("🔍 Starting Comprehensive Molecule Validation")
    logger.info("=" * 60)
    
    # Load molecules
    try:
        df = pd.read_csv(csv_file)
        molecules = df['smiles'].tolist()
        logger.info(f"📋 Loaded {len(molecules)} molecules from {csv_file}")
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return
    
    # Initialize validator
    validator = MoleculeValidator()
    
    # Validation results
    all_results = []
    
    logger.info("🧪 Validating each molecule...")
    
    for i, smiles in enumerate(molecules, 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"🧬 Molecule {i}: {smiles}")
        logger.info('='*50)
        
        result = {'molecule_id': i, 'smiles': smiles}
        
        # 1. Structure validation
        logger.info("1️⃣ Structure Validation:")
        structure_results = validator.validate_structure(smiles)
        result.update(structure_results)
        
        if structure_results['is_valid']:
            logger.info(f"   ✅ Valid structure ({structure_results['num_atoms']} atoms, {structure_results['num_bonds']} bonds)")
            if structure_results['has_unusual_atoms']:
                logger.warning(f"   ⚠️  Contains unusual atoms")
            if structure_results['has_metals']:
                logger.warning(f"   ⚠️  Contains metal atoms")
        else:
            logger.error(f"   ❌ Invalid: {structure_results['error_message']}")
            all_results.append(result)
            continue
        
        # 2. Novelty check
        logger.info("2️⃣ Novelty Assessment:")
        novelty_results = validator.check_novelty(smiles)
        result.update(novelty_results)
        
        if novelty_results['is_exact_match']:
            logger.warning(f"   ❌ EXACT MATCH found: {novelty_results['exact_match_compound']}")
        elif novelty_results['is_novel']:
            logger.info(f"   ✅ NOVEL molecule")
            logger.info(f"   📊 Max similarity to known: {novelty_results['max_similarity_known']:.3f} ({novelty_results['most_similar_known']})")
        else:
            logger.warning(f"   ⚠️  HIGH SIMILARITY to known compounds")
            logger.info(f"   📊 Max similarity: {novelty_results['max_similarity_known']:.3f} ({novelty_results['most_similar_known']})")
        
        # 3. Drug-likeness
        logger.info("3️⃣ Drug-likeness Assessment:")
        drug_results = validator.check_drug_likeness(smiles)
        if drug_results:
            result.update(drug_results)
            logger.info(f"   💊 QED Score: {drug_results['qed']:.3f}")
            logger.info(f"   ⚖️  Lipinski violations: {drug_results['lipinski_violations']}")
            logger.info(f"   🎯 Drug-like: {'Yes' if drug_results['is_drug_like'] else 'No'}")
        
        # 4. Scaffold analysis
        logger.info("4️⃣ Scaffold Analysis:")
        scaffold = validator.get_scaffold(smiles)
        result['scaffold'] = scaffold
        if scaffold:
            logger.info(f"   🧬 Scaffold: {scaffold}")
        
        all_results.append(result)
    
    # Summary analysis
    logger.info("\n" + "="*60)
    logger.info("📊 COMPREHENSIVE VALIDATION SUMMARY")
    logger.info("="*60)
    
    valid_molecules = [r for r in all_results if r.get('is_valid', False)]
    novel_molecules = [r for r in valid_molecules if r.get('is_novel', False)]
    drug_like = [r for r in valid_molecules if r.get('is_drug_like', False)]
    exact_matches = [r for r in valid_molecules if r.get('is_exact_match', False)]
    
    logger.info(f"🧪 Total molecules analyzed: {len(molecules)}")
    logger.info(f"✅ Valid structures: {len(valid_molecules)}/{len(molecules)} ({len(valid_molecules)/len(molecules)*100:.1f}%)")
    logger.info(f"🆕 Novel molecules: {len(novel_molecules)}/{len(valid_molecules)} ({len(novel_molecules)/len(valid_molecules)*100:.1f}%)")
    logger.info(f"💊 Drug-like molecules: {len(drug_like)}/{len(valid_molecules)} ({len(drug_like)/len(valid_molecules)*100:.1f}%)")
    logger.info(f"🔄 Exact matches found: {len(exact_matches)}")
    
    if novel_molecules:
        logger.info(f"\n🏆 TOP NOVEL CANDIDATES:")
        novel_sorted = sorted(novel_molecules, key=lambda x: x.get('qed', 0), reverse=True)
        for i, mol in enumerate(novel_sorted[:5], 1):
            logger.info(f"  {i}. {mol['smiles']}")
            logger.info(f"     QED: {mol.get('qed', 0):.3f} | Max similarity: {mol.get('max_similarity_known', 0):.3f}")
    
    if exact_matches:
        logger.info(f"\n⚠️  EXACT MATCHES FOUND:")
        for mol in exact_matches:
            logger.info(f"  {mol['smiles']} → {mol['exact_match_compound']}")
    
    # Save detailed results
    results_df = pd.DataFrame(all_results)
    output_file = csv_file.replace('.csv', '_validation_results.csv')
    results_df.to_csv(output_file, index=False)
    logger.info(f"\n💾 Detailed results saved to: {output_file}")
    
    return results_df

def main():
    """Main validation function"""
    
    # Check for results file
    csv_files = [
        'modification_results/molecular_properties.csv',
        'constrained_results/molecular_properties.csv',
        'working_results/molecular_properties.csv',
        'results/molecular_properties.csv'
    ]
    
    csv_file = None
    for file in csv_files:
        if os.path.exists(file):
            csv_file = file
            break
    
    if csv_file is None:
        logger.error("❌ No molecular properties CSV file found!")
        logger.info("Expected files:")
        for file in csv_files:
            logger.info(f"  - {file}")
        return
    
    logger.info(f"📁 Using file: {csv_file}")
    
    # Run comprehensive validation
    results = validate_molecules_comprehensive(csv_file)
    
    logger.info("\n🎉 Validation Complete!")
    logger.info("Check the validation results CSV for detailed analysis.")

if __name__ == "__main__":
    main()