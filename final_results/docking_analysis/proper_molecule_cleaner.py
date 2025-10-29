#!/usr/bin/env python3
"""
Proper Molecule Dataset Cleaner
Author: TAQDEES
Description: Clean and validate molecular datasets properly - no more mess!
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Try to import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED
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

class ProperMoleculeValidator:
    """Proper validation with no mistakes"""
    
    def __init__(self):
        # Known molecules that should be excluded
        self.known_basic_molecules = {
            'c1ccc2ccccc2c1': 'Naphthalene - basic aromatic',
            'c1cnc2ccccc2c1': 'Quinoline - basic heterocycle', 
            'c1ccc2ncncc2c1': 'Quinazoline - basic scaffold',
            'c1ccc(Nc2ncnc3ccccc23)cc1': 'Basic anilinoquinazoline',
            'Nc1ccc(Cl)cc1': 'p-Chloroaniline - basic building block',
            'COc1ccc(N)cc1': 'p-Anisidine - basic building block',
            'Nc1ccc(F)c(Cl)c1': 'Basic dihalogenated aniline'
        }
        
        # FDA-approved EGFR inhibitors for high similarity check
        self.known_egfr_drugs = {
            'Erlotinib': 'COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCOCCN',
            'Gefitinib': 'COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1',
            'Afatinib': 'CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OCCCN',
            'Osimertinib': 'COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc1nccc(n1)c1c(C)cccc1Cl'
        }

    def calculate_tanimoto_similarity(self, smiles1, smiles2):
        """Calculate Tanimoto similarity"""
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

    def clean_bindingforge_molecules(self):
        """Clean the BindingForge molecules properly"""
        
        logger.info("🧹 Cleaning BindingForge molecules...")
        
        # From your validation results - ONLY the truly novel ones
        raw_molecules = [
            'c1ccc(CC2ncCc(c3ccccc3)n2)cc1',        # Novel benzyl-pyrimidine (QED: 0.794)
            'COc1ccc(Nc2nccc3ccccc23)cc1',          # Novel quinoline variant (QED: 0.763) 
            'Cc1ccc2ncnc(Nc3ccc(F)c(Cl)c3)c2c1',    # Novel methyl-quinazoline (QED: 0.755)
            'c1ccc(Nc2nccc(c3ccccc3)n2)cc1',        # Novel pyrimidine core (QED: 0.763)
            'COc1ccc(Nc2nccc(c3ccccc3)n2)cc1',      # Novel methoxy-pyrimidine (QED: 0.785)
            
            # Remove these - they were identified as basic/known:
            # 'Nc1ccc(Cl)cc1',          # Basic building block
            # 'COc1ccc(N)cc1',          # Basic building block  
            # 'Nc1ccc(F)c(Cl)c1',       # Basic building block
            # 'c1ccc2ccccc2c1',         # Naphthalene
            # 'c1cnc2ccccc2c1'          # Quinoline
        ]
        
        logger.info(f"📋 Candidate molecules: {len(raw_molecules)}")
        
        cleaned_molecules = []
        removed_molecules = []
        
        for i, smiles in enumerate(raw_molecules, 1):
            # Check if it's a known basic molecule
            is_basic = False
            for known_smiles, description in self.known_basic_molecules.items():
                similarity = self.calculate_tanimoto_similarity(smiles, known_smiles)
                if similarity > 0.95:  # Very high similarity
                    removed_molecules.append({
                        'smiles': smiles,
                        'reason': f'Basic molecule: {description}',
                        'similarity': similarity
                    })
                    is_basic = True
                    break
            
            if is_basic:
                continue
            
            # Check similarity to known EGFR drugs
            is_too_similar = False
            for drug_name, drug_smiles in self.known_egfr_drugs.items():
                similarity = self.calculate_tanimoto_similarity(smiles, drug_smiles)
                if similarity > 0.85:  # High similarity threshold
                    removed_molecules.append({
                        'smiles': smiles,
                        'reason': f'Too similar to {drug_name}',
                        'similarity': similarity
                    })
                    is_too_similar = True
                    break
            
            if is_too_similar:
                continue
            
            # Check for duplicates within the dataset
            is_duplicate = False
            for existing in cleaned_molecules:
                similarity = self.calculate_tanimoto_similarity(smiles, existing['smiles'])
                if similarity > 0.99:  # Exact match
                    removed_molecules.append({
                        'smiles': smiles,
                        'reason': 'Duplicate within dataset',
                        'similarity': similarity
                    })
                    is_duplicate = True
                    break
            
            if is_duplicate:
                continue
            
            # Calculate properties
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                cleaned_molecules.append({
                    'mol_id': len(cleaned_molecules) + 1,
                    'smiles': smiles,
                    'source': 'BindingForge_Validated',
                    'molecular_weight': Descriptors.MolWt(mol),
                    'logp': Descriptors.MolLogP(mol),
                    'qed': QED.qed(mol),
                    'hbd': Descriptors.NumHDonors(mol),
                    'hba': Descriptors.NumHAcceptors(mol),
                    'tpsa': Descriptors.TPSA(mol),
                    'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                    'aromatic_rings': Descriptors.NumAromaticRings(mol)
                })
        
        logger.info(f"✅ Cleaned molecules: {len(cleaned_molecules)}")
        logger.info(f"❌ Removed molecules: {len(removed_molecules)}")
        
        if removed_molecules:
            logger.info("\n🗑️  REMOVED MOLECULES:")
            for mol in removed_molecules:
                logger.info(f"   - {mol['smiles']} | {mol['reason']} | Similarity: {mol['similarity']:.3f}")
        
        return cleaned_molecules, removed_molecules

    def add_previous_project_molecules(self):
        """Add the 13 molecules from previous project"""
        
        logger.info("📋 Adding previous project molecules...")
        
        previous_molecules = [
            'c1ccc(cc1Nc1ccccc1Nc1ccccc1OC)C',
            'n1c2c(ncnc2c(Nc2cccc(NC(C)C)c2)cc1)C1CCC1',
            'c1ccc(cc2c1Nc1ncnc(c1)NCCCCC1)Nc1ncnc2C1CCOCC1',
            'c1ccccc1Nc2cc(ncn2)NC1CC1CCC(C)C',
            'c1cc(cc(Nc2cc(NCCC(C)C)c2)ccc1)Cl',
            'c1ccc(cc2c1cc(cc2)NCCC(O)CC(O)CC1)ncn1',
            'c1ccc(cc2c1Nc1ccccc(NCCC(O)CO)nc2)c1',
            'c1ccc(cc1Nc1cc(ccc1)Br)N',
            'c1ccc(Nc2cccc2c1)Nc1cc(N)ncn1',
            'c1c(ccc(Nc2cc(NC(C)C)cc1)Nc1cccc(NC(C)C)nc2)c1',
            'c1ccc(cc2c1Nc1cccc(NC(C)C)c2)c1',
            'c1ccc(cc1Nc1cccc(Cl)c1)N',
            'c1ccc(cc1Nc1cc(ccc1)NCCC(C)CC1)cccc1'
        ]
        
        previous_data = []
        for i, smiles in enumerate(previous_molecules, 1):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                previous_data.append({
                    'mol_id': 100 + i,  # Different ID range
                    'smiles': smiles,
                    'source': 'Previous_Project',
                    'molecular_weight': Descriptors.MolWt(mol),
                    'logp': Descriptors.MolLogP(mol),
                    'qed': QED.qed(mol),
                    'hbd': Descriptors.NumHDonors(mol),
                    'hba': Descriptors.NumHAcceptors(mol),
                    'tpsa': Descriptors.TPSA(mol),
                    'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                    'aromatic_rings': Descriptors.NumAromaticRings(mol)
                })
        
        logger.info(f"✅ Added {len(previous_data)} previous project molecules")
        return previous_data

    def create_final_clean_dataset(self):
        """Create the final clean dataset"""
        
        logger.info("🎯 Creating final clean dataset...")
        
        # Get cleaned BindingForge molecules
        bindingforge_molecules, removed = self.clean_bindingforge_molecules()
        
        # Get previous project molecules
        previous_molecules = self.add_previous_project_molecules()
        
        # Combine
        all_molecules = bindingforge_molecules + previous_molecules
        
        # Create DataFrame
        df = pd.DataFrame(all_molecules)
        
        # Add additional drug-like properties
        df['lipinski_violations'] = (
            (df['molecular_weight'] > 500).astype(int) +
            (df['logp'] > 5).astype(int) +
            (df['hbd'] > 5).astype(int) +
            (df['hba'] > 10).astype(int)
        )
        
        df['is_drug_like'] = (df['qed'] >= 0.5) & (df['lipinski_violations'] == 0)
        
        # Sort by QED score (best first)
        df = df.sort_values('qed', ascending=False).reset_index(drop=True)
        
        return df, removed

    def save_clean_dataset(self, df, removed_molecules):
        """Save the clean dataset"""
        
        output_dir = "clean_molecular_dataset"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main dataset
        main_file = os.path.join(output_dir, 'clean_novel_molecules.csv')
        df.to_csv(main_file, index=False)
        
        # Save removed molecules log
        if removed_molecules:
            removed_df = pd.DataFrame(removed_molecules)
            removed_file = os.path.join(output_dir, 'removed_molecules_log.csv')
            removed_df.to_csv(removed_file, index=False)
        
        # Create summary report
        self.create_clean_summary_report(df, removed_molecules, output_dir)
        
        logger.info(f"💾 Clean dataset saved to: {output_dir}")
        
        return main_file

    def create_clean_summary_report(self, df, removed_molecules, output_dir):
        """Create summary report"""
        
        report_file = os.path.join(output_dir, 'clean_dataset_report.txt')
        
        with open(report_file, 'w') as f:
            f.write("CLEAN MOLECULAR DATASET REPORT\n")
            f.write("=" * 40 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("DATASET COMPOSITION:\n")
            f.write(f"Total clean molecules: {len(df)}\n")
            
            source_counts = df['source'].value_counts()
            for source, count in source_counts.items():
                f.write(f"  - {source}: {count} molecules\n")
            
            f.write(f"\nREMOVED MOLECULES: {len(removed_molecules)}\n")
            for mol in removed_molecules:
                f.write(f"  - {mol['smiles']} | {mol['reason']}\n")
            
            f.write(f"\nQUALITY METRICS:\n")
            f.write(f"Average QED score: {df['qed'].mean():.3f}\n")
            f.write(f"Drug-like molecules: {df['is_drug_like'].sum()}/{len(df)} ({df['is_drug_like'].mean()*100:.1f}%)\n")
            f.write(f"Lipinski compliant: {(df['lipinski_violations'] == 0).sum()}/{len(df)} ({(df['lipinski_violations'] == 0).mean()*100:.1f}%)\n")
            
            f.write(f"\nTOP 10 MOLECULES BY QED:\n")
            for idx, row in df.head(10).iterrows():
                f.write(f"{idx+1:2d}. QED: {row['qed']:.3f} | MW: {row['molecular_weight']:.0f} | {row['smiles']}\n")

def main():
    """Main cleaning function"""
    
    logger.info("🧹 PROPER MOLECULE DATASET CLEANING")
    logger.info("=" * 50)
    logger.info("No more mess - doing this right!")
    
    try:
        # Initialize validator
        validator = ProperMoleculeValidator()
        
        # Create clean dataset
        clean_df, removed_molecules = validator.create_final_clean_dataset()
        
        # Save results
        main_file = validator.save_clean_dataset(clean_df, removed_molecules)
        
        # Print summary
        logger.info("\n🎉 CLEANING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 40)
        logger.info(f"✅ Clean molecules: {len(clean_df)}")
        logger.info(f"   - BindingForge validated: {len(clean_df[clean_df['source'] == 'BindingForge_Validated'])}")
        logger.info(f"   - Previous project: {len(clean_df[clean_df['source'] == 'Previous_Project'])}")
        logger.info(f"❌ Removed molecules: {len(removed_molecules)}")
        logger.info(f"📁 Clean dataset: {main_file}")
        
        # Show top molecules
        logger.info(f"\n🏆 TOP 5 MOLECULES (by QED):")
        for idx, row in clean_df.head(5).iterrows():
            logger.info(f"  {idx+1}. QED: {row['qed']:.3f} | {row['source'][:15]} | {row['smiles']}")
        
        logger.info(f"\n✨ Ready for docking analysis!")
        
    except Exception as e:
        logger.error(f"❌ Cleaning failed: {e}")
        raise

if __name__ == "__main__":
    main()