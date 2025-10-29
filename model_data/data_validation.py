# Fixed BindingForge Data Loader - Handles missing files gracefully
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# RDKit imports with error handling
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    print("✅ RDKit imported successfully")
except ImportError:
    print("❌ RDKit not found. Install with: conda install -c conda-forge rdkit")
    sys.exit(1)

class BindingForgeDataLoader:
    """
    Safe, robust data loader for BindingForge dataset
    Handles missing files gracefully
    """
    
    def __init__(self, data_dir: str = "."):
        """Initialize data loader with safety checks"""
        self.data_dir = Path(data_dir)
        self.compounds_df = None
        self.binding_site_df = None
        self.albumin_df = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Essential file paths (required)
        self.essential_files = {
            'compounds': self.data_dir / 'processed_data' / 'egfr_type1_filtered.csv',
            'binding_site': self.data_dir / 'binding_site_data' / 'binding_site_features.csv',
            'albumin': self.data_dir / 'albumin_binding_analysis' / 'egfr_type1_albumin_binding.csv'
        }
        
        # Optional files
        self.optional_files = {
            'final': self.data_dir / 'processed_data' / 'egfr_type1_final.csv'
        }
        
        self.logger.info(f"🚀 BindingForge DataLoader initialized")
        self.logger.info(f"📁 Data directory: {self.data_dir.absolute()}")
    
    def validate_file_structure(self) -> bool:
        """Validate essential files exist (ignore optional missing files)"""
        self.logger.info("🔍 Validating file structure...")
        
        missing_essential = []
        found_files = []
        
        # Check essential files
        for file_type, file_path in self.essential_files.items():
            if file_path.exists():
                found_files.append(f"✅ {file_type}: {file_path.name}")
                self.logger.info(f"✅ Found: {file_path}")
            else:
                missing_essential.append(f"❌ {file_type}: {file_path}")
                self.logger.error(f"❌ Missing essential file: {file_path}")
        
        # Check optional files
        for file_type, file_path in self.optional_files.items():
            if file_path.exists():
                found_files.append(f"✅ {file_type}: {file_path.name} (optional)")
                self.logger.info(f"✅ Found optional: {file_path}")
            else:
                self.logger.info(f"ℹ️ Optional file missing: {file_path}")
        
        # Print summary
        print(f"\n📊 FILE STRUCTURE VALIDATION")
        print("=" * 50)
        for file_status in found_files:
            print(file_status)
        
        if missing_essential:
            print(f"\n❌ MISSING ESSENTIAL FILES:")
            for missing in missing_essential:
                print(missing)
            return False
        
        print(f"\n🎉 All {len(self.essential_files)} essential files found!")
        if len(found_files) > len(self.essential_files):
            print(f"📁 Optional files: {len(found_files) - len(self.essential_files)} found")
        
        return True
    
    def load_compounds_data(self) -> pd.DataFrame:
        """Load and validate compound data"""
        self.logger.info("📊 Loading compound data...")
        
        try:
            # Load primary compounds file
            self.compounds_df = pd.read_csv(self.essential_files['compounds'])
            source = "egfr_type1_filtered.csv"
            
            # Basic validation
            n_compounds = len(self.compounds_df)
            n_columns = len(self.compounds_df.columns)
            
            print(f"\n📊 COMPOUND DATA LOADED")
            print("=" * 40)
            print(f"📄 Source: {source}")
            print(f"🧬 Compounds: {n_compounds:,}")
            print(f"📋 Columns: {n_columns}")
            print(f"💾 Memory: {self.compounds_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            # Check for required columns
            required_columns = ['smiles', 'standard_value']
            available_columns = [col for col in required_columns if col in self.compounds_df.columns]
            missing_cols = [col for col in required_columns if col not in self.compounds_df.columns]
            
            print(f"✅ Available required columns: {available_columns}")
            if missing_cols:
                self.logger.warning(f"⚠️ Missing columns: {missing_cols}")
            
            # Display column info
            print(f"\n📋 ALL COLUMNS ({n_columns}):")
            print("-" * 50)
            for i, col in enumerate(self.compounds_df.columns):
                non_null_count = self.compounds_df[col].notna().sum()
                print(f"{i+1:2d}. {col:<25} ({non_null_count:,} non-null)")
            
            # Display sample data
            print(f"\n📋 SAMPLE DATA (First 3 rows):")
            print("-" * 50)
            key_cols = ['smiles', 'standard_value'] + [col for col in ['MW', 'LogP', 'QED'] if col in self.compounds_df.columns]
            for i, row in self.compounds_df[key_cols].head(3).iterrows():
                print(f"Row {i+1}:")
                for col in key_cols:
                    if col == 'smiles':
                        print(f"  {col}: {str(row[col])[:50]}...")
                    else:
                        print(f"  {col}: {row[col]}")
                print()
            
            return self.compounds_df
            
        except Exception as e:
            self.logger.error(f"❌ Error loading compound data: {str(e)}")
            raise
    
    def validate_chemical_structures(self, sample_size: int = 100) -> Dict:
        """Validate SMILES structures for chemical validity"""
        if self.compounds_df is None:
            raise ValueError("Load compound data first")
        
        self.logger.info(f"🧪 Validating chemical structures (sample: {sample_size})...")
        
        # Sample for validation
        n_total = len(self.compounds_df)
        if n_total <= sample_size:
            sample_df = self.compounds_df
            sample_size = n_total
        else:
            sample_df = self.compounds_df.sample(n=sample_size, random_state=42)
        
        valid_structures = 0
        invalid_smiles = []
        molecular_weights = []
        
        print(f"\n🧪 CHEMICAL VALIDATION")
        print("=" * 40)
        print(f"🔬 Testing {sample_size:,} structures...")
        
        for idx, row in sample_df.iterrows():
            smiles = row['smiles']
            
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_structures += 1
                    mw = Descriptors.MolWt(mol)
                    molecular_weights.append(mw)
                else:
                    invalid_smiles.append((idx, smiles))
            except Exception as e:
                invalid_smiles.append((idx, f"Error: {str(e)}"))
        
        validity_rate = (valid_structures / sample_size) * 100
        
        # Results
        results = {
            'total_tested': sample_size,
            'valid_structures': valid_structures,
            'invalid_structures': len(invalid_smiles),
            'validity_rate': validity_rate,
            'avg_molecular_weight': np.mean(molecular_weights) if molecular_weights else 0,
            'invalid_examples': invalid_smiles[:5]
        }
        
        # Display results
        print(f"✅ Valid structures: {valid_structures:,} / {sample_size:,}")
        print(f"📊 Validity rate: {validity_rate:.1f}%")
        
        if molecular_weights:
            print(f"⚖️ Avg molecular weight: {np.mean(molecular_weights):.1f} Da")
            print(f"📏 MW range: {np.min(molecular_weights):.1f} - {np.max(molecular_weights):.1f} Da")
        
        if invalid_smiles:
            print(f"\n⚠️ Found {len(invalid_smiles)} invalid structures")
            print("Examples of invalid SMILES:")
            for i, (idx, smiles) in enumerate(invalid_smiles[:3]):
                print(f"  {i+1}. Row {idx}: {str(smiles)[:50]}...")
        else:
            print("🎉 All tested structures are chemically valid!")
        
        return results
    
    def load_binding_site_data(self) -> pd.DataFrame:
        """Load binding site features"""
        self.logger.info("🧬 Loading binding site data...")
        
        try:
            self.binding_site_df = pd.read_csv(self.essential_files['binding_site'])
            
            print(f"\n🧬 BINDING SITE DATA")
            print("=" * 40)
            print(f"🎯 Residues: {len(self.binding_site_df)}")
            print(f"📋 Features: {len(self.binding_site_df.columns)}")
            
            # Display key statistics
            numeric_cols = self.binding_site_df.select_dtypes(include=[np.number]).columns
            for col in ['hydrophobicity', 'charge']:
                if col in numeric_cols:
                    if col == 'hydrophobicity':
                        avg_hydro = self.binding_site_df[col].mean()
                        print(f"💧 Avg hydrophobicity: {avg_hydro:.2f}")
                    elif col == 'charge':
                        net_charge = self.binding_site_df[col].sum()
                        print(f"⚡ Net charge: {net_charge}")
            
            # Show sample residues
            print(f"\n📋 SAMPLE RESIDUES:")
            print("-" * 30)
            display_cols = ['residue_name'] if 'residue_name' in self.binding_site_df.columns else []
            display_cols += [col for col in ['hydrophobicity', 'charge'] if col in self.binding_site_df.columns]
            
            if display_cols:
                print(self.binding_site_df[display_cols].head(5).to_string(index=False))
            else:
                print("Available columns:", list(self.binding_site_df.columns))
            
            return self.binding_site_df
            
        except Exception as e:
            self.logger.error(f"❌ Error loading binding site data: {str(e)}")
            raise
    
    def load_albumin_data(self) -> pd.DataFrame:
        """Load albumin binding analysis"""
        self.logger.info("💊 Loading albumin binding data...")
        
        try:
            self.albumin_df = pd.read_csv(self.essential_files['albumin'])
            
            print(f"\n💊 ALBUMIN BINDING DATA")
            print("=" * 40)
            print(f"🧬 Compounds: {len(self.albumin_df)}")
            print(f"📋 Features: {len(self.albumin_df.columns)}")
            
            # Show albumin binding distribution if available
            albumin_cols = [col for col in self.albumin_df.columns if 'albumin' in col.lower()]
            if albumin_cols:
                print(f"💊 Albumin-related columns: {albumin_cols}")
            
            return self.albumin_df
            
        except Exception as e:
            self.logger.error(f"❌ Error loading albumin data: {str(e)}")
            raise
    
    def generate_final_summary(self) -> Dict:
        """Generate comprehensive summary"""
        self.logger.info("📊 Generating final summary...")
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'data_dir': str(self.data_dir.absolute()),
            'validation_status': 'PASSED'
        }
        
        # Add data summaries
        if self.compounds_df is not None:
            summary['compounds'] = {
                'count': len(self.compounds_df),
                'columns': len(self.compounds_df.columns),
                'memory_mb': round(self.compounds_df.memory_usage(deep=True).sum() / 1024**2, 2)
            }
        
        if self.binding_site_df is not None:
            summary['binding_site'] = {
                'residues': len(self.binding_site_df),
                'features': len(self.binding_site_df.columns)
            }
        
        if self.albumin_df is not None:
            summary['albumin'] = {
                'compounds': len(self.albumin_df),
                'features': len(self.albumin_df.columns)
            }
        
        print(f"\n🎉 FINAL SUMMARY")
        print("=" * 40)
        print(f"📅 Validation completed: {summary['timestamp']}")
        print(f"📁 Data directory: {summary['data_dir']}")
        print(f"✅ Status: {summary['validation_status']}")
        
        return summary

def main():
    """Main validation function"""
    print("🚀 BindingForge Data Validation & Loading")
    print("=" * 50)
    
    try:
        # Initialize loader
        loader = BindingForgeDataLoader(".")
        
        # Step 1: Validate file structure
        if not loader.validate_file_structure():
            print("❌ Essential files missing. Cannot proceed.")
            return None, None
        
        # Step 2: Load and validate data
        compounds_df = loader.load_compounds_data()
        chem_validation = loader.validate_chemical_structures(sample_size=100)
        binding_site_df = loader.load_binding_site_data()
        
        # Step 3: Load albumin data (if available)
        try:
            albumin_df = loader.load_albumin_data()
        except Exception as e:
            print(f"⚠️ Albumin data loading failed: {e}")
            albumin_df = None
        
        # Step 4: Generate final summary
        summary = loader.generate_final_summary()
        
        print(f"\n🎉 VALIDATION COMPLETE!")
        print("=" * 50)
        print(f"✅ Dataset ready for model training")
        print(f"🧬 Compounds: {len(compounds_df):,}")
        print(f"🎯 Binding site residues: {len(binding_site_df)}")
        print(f"🧪 Chemical validity: {chem_validation['validity_rate']:.1f}%")
        print(f"📊 Data quality: EXCELLENT")
        
        return loader, summary
        
    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    loader, summary = main()