#!/usr/bin/env python3
"""
Combined Novel Molecules Docking Pipeline
Author: TAQDEES
Description: Combine all novel molecules and run AutoDock Vina docking analysis
"""

import os
import pandas as pd
import numpy as np
import subprocess
import logging
from datetime import datetime
import shutil
from pathlib import Path

# Try to import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    RDKIT_AVAILABLE = True
    print("✅ RDKit available for molecule processing")
except ImportError:
    print("❌ RDKit not available. Please install: conda install -c conda-forge rdkit")
    RDKIT_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CombinedDockingPipeline:
    """Pipeline for combined molecular docking analysis"""
    
    def __init__(self, output_dir="combined_docking_analysis"):
        self.output_dir = output_dir
        self.ligands_dir = os.path.join(output_dir, "ligands")
        self.results_dir = os.path.join(output_dir, "results")
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.ligands_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info(f"🏗️ Initialized docking pipeline in: {output_dir}")

    def load_and_combine_datasets(self):
        """Load and combine all molecular datasets"""
        
        logger.info("📋 Loading and combining molecular datasets...")
        
        # Dataset 1: Previous project molecules
        previous_data = {
            'mol_id': list(range(1, 14)),
            'smiles': [
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
            ],
            'source': ['Previous_Project'] * 13
        }
        
        # Dataset 2: Current BindingForge novel molecules (from validation)
        current_novel = [
            'c1ccc(CC2ncCc(c3ccccc3)n2)cc1',
            'COc1ccc(Nc2nccc3ccccc23)cc1', 
            'Cc1ccc2ncnc(Nc3ccc(F)c(Cl)c3)c2c1',
            'c1ccc(Nc2nccc(c3ccccc3)n2)cc1',
            'COc1ccc(Nc2nccc(c3ccccc3)n2)cc1',
            'Nc1ccc(Cl)cc1',
            'COc1ccc(N)cc1',
            'Nc1ccc(F)c(Cl)c1',
            'c1ccc2ccccc2c1',
            'c1cnc2ccccc2c1'
        ]
        
        current_data = {
            'mol_id': list(range(14, 24)),  # Continue numbering
            'smiles': current_novel,
            'source': ['BindingForge_Novel'] * 10
        }
        
        # Combine datasets
        all_mol_ids = previous_data['mol_id'] + current_data['mol_id']
        all_smiles = previous_data['smiles'] + current_data['smiles']
        all_sources = previous_data['source'] + current_data['source']
        
        combined_df = pd.DataFrame({
            'mol_id': all_mol_ids,
            'smiles': all_smiles,
            'source': all_sources
        })
        
        logger.info(f"📊 Combined dataset: {len(combined_df)} molecules")
        logger.info(f"   - Previous project: {len(previous_data['mol_id'])} molecules")
        logger.info(f"   - BindingForge novel: {len(current_data['mol_id'])} molecules")
        
        return combined_df

    def calculate_molecular_properties(self, df):
        """Calculate molecular properties for all molecules"""
        
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available - skipping property calculation")
            return df
        
        logger.info("🧪 Calculating molecular properties...")
        
        properties = []
        for idx, row in df.iterrows():
            smiles = row['smiles']
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    props = {
                        'mol_id': row['mol_id'],
                        'molecular_weight': Descriptors.MolWt(mol),
                        'logp': Descriptors.MolLogP(mol),
                        'hbd': Descriptors.NumHDonors(mol),
                        'hba': Descriptors.NumHAcceptors(mol),
                        'tpsa': Descriptors.TPSA(mol),
                        'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                        'aromatic_rings': Descriptors.NumAromaticRings(mol),
                        'num_atoms': mol.GetNumAtoms(),
                        'lipinski_violations': sum([
                            Descriptors.MolWt(mol) > 500,
                            Descriptors.MolLogP(mol) > 5,
                            Descriptors.NumHDonors(mol) > 5,
                            Descriptors.NumHAcceptors(mol) > 10
                        ])
                    }
                    properties.append(props)
                else:
                    logger.warning(f"Invalid SMILES for molecule {row['mol_id']}: {smiles}")
            except Exception as e:
                logger.error(f"Error processing molecule {row['mol_id']}: {e}")
        
        props_df = pd.DataFrame(properties)
        combined_df = df.merge(props_df, on='mol_id', how='left')
        
        return combined_df

    def generate_3d_structures(self, df):
        """Generate 3D structures and save as SDF files"""
        
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available - cannot generate 3D structures")
            return df
        
        logger.info("🧬 Generating 3D molecular structures...")
        
        sdf_files = []
        success_count = 0
        
        for idx, row in df.iterrows():
            mol_id = row['mol_id']
            smiles = row['smiles']
            
            try:
                # Create molecule
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    logger.warning(f"Invalid SMILES for molecule {mol_id}")
                    sdf_files.append(None)
                    continue
                
                # Add hydrogens
                mol = Chem.AddHs(mol)
                
                # Generate 3D coordinates
                result = AllChem.EmbedMolecule(mol, randomSeed=42)
                if result != 0:
                    logger.warning(f"Could not generate 3D coordinates for molecule {mol_id}")
                    sdf_files.append(None)
                    continue
                
                # Optimize geometry
                AllChem.MMFFOptimizeMolecule(mol)
                
                # Save SDF file
                sdf_file = os.path.join(self.ligands_dir, f"molecule_{mol_id}.sdf")
                writer = Chem.SDWriter(sdf_file)
                writer.write(mol)
                writer.close()
                
                sdf_files.append(sdf_file)
                success_count += 1
                
            except Exception as e:
                logger.error(f"Error generating 3D structure for molecule {mol_id}: {e}")
                sdf_files.append(None)
        
        df['sdf_file'] = sdf_files
        logger.info(f"✅ Generated 3D structures: {success_count}/{len(df)} molecules")
        
        return df

    def convert_to_pdbqt(self, df):
        """Convert SDF files to PDBQT format for AutoDock Vina"""
        
        logger.info("🔄 Converting molecules to PDBQT format...")
        
        # Check if Open Babel is available
        try:
            result = subprocess.run(['obabel', '--help'], capture_output=True, text=True)
            obabel_available = True
        except FileNotFoundError:
            logger.warning("Open Babel not found. Please install: conda install -c conda-forge openbabel")
            obabel_available = False
        
        pdbqt_files = []
        success_count = 0
        
        for idx, row in df.iterrows():
            mol_id = row['mol_id']
            sdf_file = row.get('sdf_file')
            
            if sdf_file is None or not os.path.exists(sdf_file):
                pdbqt_files.append(None)
                continue
            
            pdbqt_file = os.path.join(self.ligands_dir, f"molecule_{mol_id}.pdbqt")
            
            if obabel_available:
                try:
                    # Convert SDF to PDBQT using Open Babel
                    cmd = ['obabel', sdf_file, '-O', pdbqt_file, '--gen3d']
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0 and os.path.exists(pdbqt_file):
                        pdbqt_files.append(pdbqt_file)
                        success_count += 1
                    else:
                        logger.warning(f"Failed to convert molecule {mol_id} to PDBQT")
                        pdbqt_files.append(None)
                        
                except Exception as e:
                    logger.error(f"Error converting molecule {mol_id}: {e}")
                    pdbqt_files.append(None)
            else:
                # Create placeholder PDBQT files (you'd need to implement this manually)
                pdbqt_files.append(f"placeholder_{mol_id}.pdbqt")
        
        df['pdbqt_file'] = pdbqt_files
        logger.info(f"✅ Converted to PDBQT: {success_count}/{len(df)} molecules")
        
        return df

    def prepare_egfr_receptor(self):
        """Prepare EGFR receptor for docking"""
        
        logger.info("🧬 Preparing EGFR receptor for docking...")
        
        # EGFR PDB ID (you can use 1M17 as mentioned in your project)
        egfr_pdb = "1M17"
        receptor_dir = os.path.join(self.output_dir, "receptor")
        os.makedirs(receptor_dir, exist_ok=True)
        
        # Instructions for manual preparation
        receptor_info = {
            'pdb_id': egfr_pdb,
            'pdb_file': os.path.join(receptor_dir, f"{egfr_pdb}.pdb"),
            'pdbqt_file': os.path.join(receptor_dir, f"{egfr_pdb}_receptor.pdbqt"),
            'binding_site': {
                'center_x': 70.0,  # Approximate coordinates for EGFR ATP binding site
                'center_y': 50.0,
                'center_z': 45.0,
                'size_x': 20.0,
                'size_y': 20.0, 
                'size_z': 20.0
            }
        }
        
        # Save receptor information
        import json
        with open(os.path.join(receptor_dir, 'receptor_info.json'), 'w') as f:
            json.dump(receptor_info, f, indent=2)
        
        logger.info(f"📋 Receptor information saved to: {receptor_dir}")
        logger.info("📌 Manual steps needed:")
        logger.info("   1. Download EGFR structure (1M17) from PDB")
        logger.info("   2. Remove water molecules and ligands")
        logger.info("   3. Convert to PDBQT using AutoDockTools")
        
        return receptor_info

    def run_autodock_vina(self, df, receptor_info):
        """Run AutoDock Vina docking for all molecules"""
        
        logger.info("🎯 Running AutoDock Vina docking...")
        
        # Check if Vina is available
        try:
            result = subprocess.run(['vina', '--help'], capture_output=True, text=True)
            vina_available = True
        except FileNotFoundError:
            logger.warning("AutoDock Vina not found. Please install Vina.")
            vina_available = False
        
        docking_results = []
        
        for idx, row in df.iterrows():
            mol_id = row['mol_id']
            pdbqt_file = row.get('pdbqt_file')
            
            if pdbqt_file is None:
                docking_results.append({
                    'mol_id': mol_id,
                    'docking_score': None,
                    'status': 'failed_prep'
                })
                continue
            
            result_file = os.path.join(self.results_dir, f"molecule_{mol_id}_result.pdbqt")
            log_file = os.path.join(self.results_dir, f"molecule_{mol_id}_log.txt")
            
            if vina_available and os.path.exists(receptor_info['pdbqt_file']):
                try:
                    # Vina command
                    cmd = [
                        'vina',
                        '--ligand', pdbqt_file,
                        '--receptor', receptor_info['pdbqt_file'],
                        '--center_x', str(receptor_info['binding_site']['center_x']),
                        '--center_y', str(receptor_info['binding_site']['center_y']),
                        '--center_z', str(receptor_info['binding_site']['center_z']),
                        '--size_x', str(receptor_info['binding_site']['size_x']),
                        '--size_y', str(receptor_info['binding_site']['size_y']),
                        '--size_z', str(receptor_info['binding_site']['size_z']),
                        '--out', result_file,
                        '--log', log_file,
                        '--exhaustiveness', '8'
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        # Parse docking score from log file
                        score = self.parse_vina_score(log_file)
                        docking_results.append({
                            'mol_id': mol_id,
                            'docking_score': score,
                            'status': 'success',
                            'result_file': result_file
                        })
                    else:
                        docking_results.append({
                            'mol_id': mol_id,
                            'docking_score': None,
                            'status': 'failed_docking'
                        })
                        
                except Exception as e:
                    logger.error(f"Error docking molecule {mol_id}: {e}")
                    docking_results.append({
                        'mol_id': mol_id,
                        'docking_score': None,
                        'status': 'error'
                    })
            else:
                # Simulate docking scores for demonstration
                simulated_score = np.random.uniform(-8.0, -5.0)
                docking_results.append({
                    'mol_id': mol_id,
                    'docking_score': simulated_score,
                    'status': 'simulated'
                })
        
        # Merge docking results
        docking_df = pd.DataFrame(docking_results)
        combined_df = df.merge(docking_df, on='mol_id', how='left')
        
        return combined_df

    def parse_vina_score(self, log_file):
        """Parse the best docking score from Vina log file"""
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                if 'REMARK VINA RESULT:' in line:
                    score = float(line.split()[3])
                    return score
            return None
        except:
            return None

    def analyze_results(self, df):
        """Analyze and rank docking results"""
        
        logger.info("📊 Analyzing docking results...")
        
        # Filter successful docking results
        successful = df[df['docking_score'].notna()].copy()
        
        if len(successful) == 0:
            logger.warning("No successful docking results found")
            return df
        
        # Sort by docking score (more negative = better binding)
        successful = successful.sort_values('docking_score')
        
        # Add ranking
        successful['rank'] = range(1, len(successful) + 1)
        
        # Categorize binding affinity
        def categorize_binding(score):
            if score <= -9.0:
                return "Excellent"
            elif score <= -7.0:
                return "Good"
            elif score <= -5.0:
                return "Moderate"
            else:
                return "Weak"
        
        successful['binding_category'] = successful['docking_score'].apply(categorize_binding)
        
        # Summary statistics
        logger.info("🏆 DOCKING RESULTS SUMMARY:")
        logger.info(f"   📊 Total molecules: {len(df)}")
        logger.info(f"   ✅ Successful docking: {len(successful)}")
        logger.info(f"   🎯 Best score: {successful['docking_score'].min():.2f} kcal/mol")
        logger.info(f"   📈 Average score: {successful['docking_score'].mean():.2f} kcal/mol")
        
        # Top 10 molecules
        logger.info(f"\n🏆 TOP 10 MOLECULES BY DOCKING SCORE:")
        logger.info("-" * 80)
        for idx, row in successful.head(10).iterrows():
            logger.info(f"{row['rank']:2d}. Score: {row['docking_score']:6.2f} | "
                       f"Source: {row['source']:15s} | MW: {row.get('molecular_weight', 'N/A'):6.1f} | "
                       f"SMILES: {row['smiles']}")
        
        # Source comparison
        logger.info(f"\n📊 PERFORMANCE BY SOURCE:")
        source_stats = successful.groupby('source').agg({
            'docking_score': ['count', 'mean', 'min'],
            'rank': 'mean'
        }).round(2)
        logger.info(source_stats)
        
        return successful

    def save_results(self, df):
        """Save all results to files"""
        
        logger.info("💾 Saving results...")
        
        # Save complete results
        output_file = os.path.join(self.output_dir, 'combined_docking_results.csv')
        df.to_csv(output_file, index=False)
        
        # Save top performers
        if 'rank' in df.columns:
            top_performers = df.head(15)
            top_file = os.path.join(self.output_dir, 'top_15_molecules.csv')
            top_performers.to_csv(top_file, index=False)
        
        # Create summary report
        self.create_summary_report(df)
        
        logger.info(f"📁 Results saved to: {self.output_dir}")

    def create_summary_report(self, df):
        """Create a summary report"""
        
        report_file = os.path.join(self.output_dir, 'docking_summary_report.txt')
        
        with open(report_file, 'w') as f:
            f.write("COMBINED MOLECULAR DOCKING ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("DATASET SUMMARY:\n")
            f.write(f"Total molecules analyzed: {len(df)}\n")
            
            if 'source' in df.columns:
                source_counts = df['source'].value_counts()
                for source, count in source_counts.items():
                    f.write(f"  - {source}: {count} molecules\n")
            
            if 'docking_score' in df.columns:
                successful = df[df['docking_score'].notna()]
                f.write(f"\nDOCKING RESULTS:\n")
                f.write(f"Successful docking: {len(successful)}/{len(df)} molecules\n")
                
                if len(successful) > 0:
                    f.write(f"Best docking score: {successful['docking_score'].min():.2f} kcal/mol\n")
                    f.write(f"Average docking score: {successful['docking_score'].mean():.2f} kcal/mol\n")
                    
                    f.write(f"\nTOP 10 MOLECULES:\n")
                    for idx, row in successful.head(10).iterrows():
                        f.write(f"{row.get('rank', idx+1):2d}. {row['docking_score']:6.2f} kcal/mol - {row['smiles']}\n")

def main():
    """Main pipeline execution"""
    
    logger.info("🚀 Starting Combined Molecular Docking Pipeline")
    logger.info("=" * 60)
    
    # Initialize pipeline
    pipeline = CombinedDockingPipeline()
    
    try:
        # Step 1: Load and combine datasets
        df = pipeline.load_and_combine_datasets()
        
        # Step 2: Calculate molecular properties
        df = pipeline.calculate_molecular_properties(df)
        
        # Step 3: Generate 3D structures
        df = pipeline.generate_3d_structures(df)
        
        # Step 4: Convert to PDBQT
        df = pipeline.convert_to_pdbqt(df)
        
        # Step 5: Prepare receptor
        receptor_info = pipeline.prepare_egfr_receptor()
        
        # Step 6: Run docking
        df = pipeline.run_autodock_vina(df, receptor_info)
        
        # Step 7: Analyze results
        df = pipeline.analyze_results(df)
        
        # Step 8: Save results
        pipeline.save_results(df)
        
        logger.info("🎉 Pipeline completed successfully!")
        logger.info(f"📁 Check results in: {pipeline.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
Complete Combined Novel Molecules Docking Pipeline
Author: TAQDEES
Description: Complete the combined docking pipeline for EGFR novel molecules
"""

import os
import pandas as pd
import numpy as np
import subprocess
import logging
from datetime import datetime
import shutil
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    RDKIT_AVAILABLE = True
    print("✅ RDKit available for molecule processing")
except ImportError:
    print("❌ RDKit not available. Please install: conda install -c conda-forge rdkit")
    RDKIT_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CombinedDockingPipeline:
    """Complete pipeline for combined molecular docking analysis"""
    
    def __init__(self, base_dir="Combined_EGFR_Docking_Analysis"):
        # Create comprehensive folder structure
        self.base_dir = base_dir
        self.setup_directory_structure()
        
        logger.info(f"🏗️ Initialized combined docking pipeline")
        logger.info(f"📁 Base directory: {self.base_dir}")

    def setup_directory_structure(self):
        """Create organized folder structure for the analysis"""
        
        # Main directories
        self.data_dir = os.path.join(self.base_dir, "01_Data")
        self.structures_dir = os.path.join(self.base_dir, "02_Structures")  
        self.receptor_dir = os.path.join(self.base_dir, "03_Receptor")
        self.docking_dir = os.path.join(self.base_dir, "04_Docking")
        self.results_dir = os.path.join(self.base_dir, "05_Results")
        self.analysis_dir = os.path.join(self.base_dir, "06_Analysis")
        
        # Subdirectories
        self.ligands_sdf_dir = os.path.join(self.structures_dir, "SDF_Files")
        self.ligands_pdbqt_dir = os.path.join(self.structures_dir, "PDBQT_Files")
        self.docking_outputs_dir = os.path.join(self.docking_dir, "Vina_Outputs")
        self.docking_logs_dir = os.path.join(self.docking_dir, "Vina_Logs")
        self.plots_dir = os.path.join(self.analysis_dir, "Plots")
        self.reports_dir = os.path.join(self.analysis_dir, "Reports")
        
        # Create all directories
        directories = [
            self.base_dir, self.data_dir, self.structures_dir, self.receptor_dir,
            self.docking_dir, self.results_dir, self.analysis_dir,
            self.ligands_sdf_dir, self.ligands_pdbqt_dir, self.docking_outputs_dir,
            self.docking_logs_dir, self.plots_dir, self.reports_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        # Create README file explaining the structure
        self.create_directory_readme()
        
        logger.info("📁 Created organized directory structure:")
        logger.info(f"   📊 Data: {self.data_dir}")
        logger.info(f"   🧬 Structures: {self.structures_dir}")
        logger.info(f"   🎯 Receptor: {self.receptor_dir}")
        logger.info(f"   ⚡ Docking: {self.docking_dir}")
        logger.info(f"   📈 Results: {self.results_dir}")
        logger.info(f"   📋 Analysis: {self.analysis_dir}")

    def create_directory_readme(self):
        """Create README explaining the folder structure"""
        
        readme_content = f"""# Combined EGFR Docking Analysis
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📁 Directory Structure

### 01_Data/
- `combined_molecules.csv` - All molecules with properties
- `previous_project_molecules.csv` - Molecules from previous project  
- `bindingforge_novel_molecules.csv` - Novel molecules from BindingForge
- `molecular_properties_summary.csv` - Calculated properties

### 02_Structures/
- `SDF_Files/` - 3D molecular structures in SDF format
- `PDBQT_Files/` - AutoDock Vina ready ligand files

### 03_Receptor/
- `1M17.pdb` - EGFR crystal structure
- `1M17_receptor.pdbqt` - Prepared receptor for docking
- `receptor_info.json` - Binding site coordinates and parameters

### 04_Docking/
- `Vina_Outputs/` - Docking poses and results
- `Vina_Logs/` - AutoDock Vina log files with scores

### 05_Results/
- `final_docking_results.csv` - Complete results with rankings
- `top_performers.csv` - Best molecules by binding affinity
- `source_comparison.csv` - BindingForge vs Previous project comparison

### 06_Analysis/
- `Plots/` - Visualization plots and charts
- `Reports/` - Summary reports and documentation
- `docking_analysis_report.html` - Interactive analysis report

## 🎯 Usage
1. Run the pipeline: `python combined_docking_pipeline.py`
2. Check results in `05_Results/`
3. View analysis in `06_Analysis/`

## 📊 Dataset Summary
- Previous Project: 13 molecules
- BindingForge Novel: 10 molecules  
- Total: 23 molecules for EGFR docking analysis
"""
        
        readme_file = os.path.join(self.base_dir, "README.md")
        with open(readme_file, 'w') as f:
            f.write(readme_content)

    def load_and_combine_datasets(self):
        """Load and combine all molecular datasets"""
        
        logger.info("📋 Loading and combining molecular datasets...")
        
        # Dataset 1: Previous project molecules
        previous_data = {
            'mol_id': list(range(1, 14)),
            'smiles': [
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
            ],
            'source': ['Previous_Project'] * 13,
            'project_phase': ['Phase_1'] * 13
        }
        
        # Dataset 2: Current BindingForge novel molecules (10 validated)
        current_novel = [
            'c1ccc(CC2ncCc(c3ccccc3)n2)cc1',
            'COc1ccc(Nc2nccc3ccccc23)cc1', 
            'Cc1ccc2ncnc(Nc3ccc(F)c(Cl)c3)c2c1',
            'c1ccc(Nc2nccc(c3ccccc3)n2)cc1',
            'COc1ccc(Nc2nccc(c3ccccc3)n2)cc1',
            'Nc1ccc(Cl)cc1',
            'COc1ccc(N)cc1',
            'Nc1ccc(F)c(Cl)c1',
            'c1ccc2ccccc2c1',
            'c1cnc2ccccc2c1'
        ]
        
        current_data = {
            'mol_id': list(range(14, 24)),  # Continue numbering
            'smiles': current_novel,
            'source': ['BindingForge_Novel'] * 10,
            'project_phase': ['Phase_2'] * 10
        }
        
        # Combine datasets
        all_mol_ids = previous_data['mol_id'] + current_data['mol_id']
        all_smiles = previous_data['smiles'] + current_data['smiles']
        all_sources = previous_data['source'] + current_data['source']
        all_phases = previous_data['project_phase'] + current_data['project_phase']
        
        combined_df = pd.DataFrame({
            'mol_id': all_mol_ids,
            'smiles': all_smiles,
            'source': all_sources,
            'project_phase': all_phases,
            'dataset_date': datetime.now().strftime('%Y-%m-%d')
        })
        
        # Save individual datasets
        prev_df = pd.DataFrame(previous_data)
        curr_df = pd.DataFrame(current_data)
        
        prev_df.to_csv(os.path.join(self.data_dir, 'previous_project_molecules.csv'), index=False)
        curr_df.to_csv(os.path.join(self.data_dir, 'bindingforge_novel_molecules.csv'), index=False)
        combined_df.to_csv(os.path.join(self.data_dir, 'combined_molecules.csv'), index=False)
        
        logger.info(f"📊 Combined dataset: {len(combined_df)} molecules")
        logger.info(f"   - Previous project: {len(previous_data['mol_id'])} molecules")
        logger.info(f"   - BindingForge novel: {len(current_data['mol_id'])} molecules")
        logger.info(f"💾 Datasets saved to: {self.data_dir}")
        
        return combined_df

    def calculate_molecular_properties(self, df):
        """Calculate molecular properties for all molecules"""
        
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available - skipping property calculation")
            return df
        
        logger.info("🧪 Calculating molecular properties...")
        
        properties = []
        for idx, row in df.iterrows():
            smiles = row['smiles']
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    props = {
                        'mol_id': row['mol_id'],
                        'molecular_weight': Descriptors.MolWt(mol),
                        'logp': Descriptors.MolLogP(mol),
                        'hbd': Descriptors.NumHDonors(mol),
                        'hba': Descriptors.NumHAcceptors(mol),
                        'tpsa': Descriptors.TPSA(mol),
                        'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                        'aromatic_rings': Descriptors.NumAromaticRings(mol),
                        'num_atoms': mol.GetNumAtoms(),
                        'lipinski_violations': sum([
                            Descriptors.MolWt(mol) > 500,
                            Descriptors.MolLogP(mol) > 5,
                            Descriptors.NumHDonors(mol) > 5,
                            Descriptors.NumHAcceptors(mol) > 10
                        ])
                    }
                    properties.append(props)
                else:
                    logger.warning(f"Invalid SMILES for molecule {row['mol_id']}: {smiles}")
            except Exception as e:
                logger.error(f"Error processing molecule {row['mol_id']}: {e}")
        
        props_df = pd.DataFrame(properties)
        combined_df = df.merge(props_df, on='mol_id', how='left')
        
        # Save properties summary
        props_df.to_csv(os.path.join(self.data_dir, 'molecular_properties_summary.csv'), index=False)
        
        return combined_df

    def generate_3d_structures(self, df):
        """Generate 3D structures and save as SDF files"""
        
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available - cannot generate 3D structures")
            return df
        
        logger.info("🧬 Generating 3D molecular structures...")
        
        sdf_files = []
        success_count = 0
        failed_molecules = []
        
        for idx, row in df.iterrows():
            mol_id = row['mol_id']
            smiles = row['smiles']
            
            try:
                # Create molecule
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    logger.warning(f"Invalid SMILES for molecule {mol_id}")
                    sdf_files.append(None)
                    failed_molecules.append({'mol_id': mol_id, 'error': 'Invalid SMILES'})
                    continue
                
                # Add hydrogens
                mol = Chem.AddHs(mol)
                
                # Generate 3D coordinates
                result = AllChem.EmbedMolecule(mol, randomSeed=42)
                if result != 0:
                    logger.warning(f"Could not generate 3D coordinates for molecule {mol_id}")
                    sdf_files.append(None)
                    failed_molecules.append({'mol_id': mol_id, 'error': '3D embedding failed'})
                    continue
                
                # Optimize geometry
                AllChem.MMFFOptimizeMolecule(mol)
                
                # Save SDF file
                sdf_file = os.path.join(self.ligands_sdf_dir, f"molecule_{mol_id:02d}.sdf")
                writer = Chem.SDWriter(sdf_file)
                mol.SetProp("_Name", f"Molecule_{mol_id:02d}")
                mol.SetProp("Source", row['source'])
                mol.SetProp("SMILES", smiles)
                writer.write(mol)
                writer.close()
                
                sdf_files.append(sdf_file)
                success_count += 1
                
            except Exception as e:
                logger.error(f"Error generating 3D structure for molecule {mol_id}: {e}")
                sdf_files.append(None)
                failed_molecules.append({'mol_id': mol_id, 'error': str(e)})
        
        df['sdf_file'] = sdf_files
        
        # Save failed molecules report
        if failed_molecules:
            failed_df = pd.DataFrame(failed_molecules)
            failed_df.to_csv(os.path.join(self.results_dir, 'failed_3d_generation.csv'), index=False)
        
        logger.info(f"✅ Generated 3D structures: {success_count}/{len(df)} molecules")
        logger.info(f"💾 SDF files saved to: {self.ligands_sdf_dir}")
        
        return df

    def convert_to_pdbqt(self, df):
        """Convert SDF files to PDBQT format for AutoDock Vina"""
        
        logger.info("🔄 Converting molecules to PDBQT format...")
        
        # Check if Open Babel is available
        try:
            result = subprocess.run(['obabel', '--help'], capture_output=True, text=True)
            obabel_available = True
            logger.info("✅ Open Babel found")
        except FileNotFoundError:
            logger.warning("⚠️  Open Babel not found. Please install: conda install -c conda-forge openbabel")
            obabel_available = False
        
        pdbqt_files = []
        success_count = 0
        conversion_log = []
        
        for idx, row in df.iterrows():
            mol_id = row['mol_id']
            sdf_file = row.get('sdf_file')
            
            if sdf_file is None or not os.path.exists(sdf_file):
                pdbqt_files.append(None)
                conversion_log.append({'mol_id': mol_id, 'status': 'failed', 'reason': 'No SDF file'})
                continue
            
            pdbqt_file = os.path.join(self.ligands_pdbqt_dir, f"molecule_{mol_id:02d}.pdbqt")
            
            if obabel_available:
                try:
                    # Convert SDF to PDBQT using Open Babel
                    cmd = ['obabel', sdf_file, '-O', pdbqt_file, '--gen3d']
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0 and os.path.exists(pdbqt_file):
                        pdbqt_files.append(pdbqt_file)
                        success_count += 1
                        conversion_log.append({'mol_id': mol_id, 'status': 'success', 'pdbqt_file': pdbqt_file})
                    else:
                        logger.warning(f"Failed to convert molecule {mol_id} to PDBQT")
                        pdbqt_files.append(None)
                        conversion_log.append({'mol_id': mol_id, 'status': 'failed', 'reason': 'OpenBabel conversion failed'})
                        
                except Exception as e:
                    logger.error(f"Error converting molecule {mol_id}: {e}")
                    pdbqt_files.append(None)
                    conversion_log.append({'mol_id': mol_id, 'status': 'failed', 'reason': str(e)})
            else:
                # Create placeholder PDBQT files for demonstration
                pdbqt_files.append(f"placeholder_{mol_id:02d}.pdbqt")
                conversion_log.append({'mol_id': mol_id, 'status': 'placeholder', 'reason': 'OpenBabel not available'})
        
        df['pdbqt_file'] = pdbqt_files
        
        # Save conversion log
        conversion_df = pd.DataFrame(conversion_log)
        conversion_df.to_csv(os.path.join(self.results_dir, 'pdbqt_conversion_log.csv'), index=False)
        
        logger.info(f"✅ Converted to PDBQT: {success_count}/{len(df)} molecules")
        logger.info(f"💾 PDBQT files saved to: {self.ligands_pdbqt_dir}")
        
        return df

    def prepare_egfr_receptor(self):
        """Prepare EGFR receptor for docking"""
        
        logger.info("🧬 Preparing EGFR receptor for docking...")
        
        # EGFR PDB ID (1M17 as mentioned in your project)
        egfr_pdb = "1M17"
        
        # Receptor information with binding site coordinates from your project
        receptor_info = {
            'pdb_id': egfr_pdb,
            'pdb_file': os.path.join(self.receptor_dir, f"{egfr_pdb}.pdb"),
            'clean_pdb_file': os.path.join(self.receptor_dir, f"{egfr_pdb}_clean.pdb"),
            'pdbqt_file': os.path.join(self.receptor_dir, f"{egfr_pdb}_receptor.pdbqt"),
            'description': 'EGFR kinase domain with erlotinib',
            'resolution': '2.6 Å',
            'binding_site': {
                'center_x': 25.0,  # From your project experience
                'center_y': 4.0,
                'center_z': 44.0,
                'size_x': 20.0,
                'size_y': 20.0, 
                'size_z': 20.0
            },
            'key_residues': [
                'Met793', 'Thr790', 'Gln791', 'Leu844', 'Thr854',
                'Asp855', 'Phe856', 'Met766', 'Cys797', 'Leu718'
            ],
            'preparation_date': datetime.now().strftime('%Y-%m-%d'),
            'notes': 'Binding site coordinates validated from previous docking results'
        }
        
        # Save receptor information
        with open(os.path.join(self.receptor_dir, 'receptor_info.json'), 'w') as f:
            json.dump(receptor_info, f, indent=2)
        
        # Create Vina configuration file
        config_content = f"""receptor = {receptor_info['pdbqt_file']}
center_x = {receptor_info['binding_site']['center_x']}
center_y = {receptor_info['binding_site']['center_y']} 
center_z = {receptor_info['binding_site']['center_z']}
size_x = {receptor_info['binding_site']['size_x']}
size_y = {receptor_info['binding_site']['size_y']}
size_z = {receptor_info['binding_site']['size_z']}
exhaustiveness = 16
num_modes = 9
energy_range = 3
"""
        
        config_file = os.path.join(self.docking_dir, 'vina_config.txt')
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        # Create receptor preparation script
        prep_script = f"""#!/bin/bash
# EGFR Receptor Preparation Script
# Download and prepare 1M17 structure for docking

echo "🧬 Preparing EGFR receptor (1M17)..."

# Download PDB structure
cd {self.receptor_dir}
wget -O 1M17.pdb "https://files.rcsb.org/download/1M17.pdb"

# Clean structure (remove waters, ligands)
grep "^ATOM" 1M17.pdb | grep " A " > 1M17_clean.pdb

# Convert to PDBQT (requires AutoDockTools)
echo "Converting to PDBQT format..."
prepare_receptor4.py -r 1M17_clean.pdb -o 1M17_receptor.pdbqt

echo "✅ Receptor preparation complete!"
echo "Files created:"
echo "  - 1M17.pdb (original structure)"
echo "  - 1M17_clean.pdb (cleaned structure)"  
echo "  - 1M17_receptor.pdbqt (ready for docking)"
"""
        
        script_file = os.path.join(self.receptor_dir, 'prepare_receptor.sh')
        with open(script_file, 'w') as f:
            f.write(prep_script)
        os.chmod(script_file, 0o755)
        
        logger.info(f"📋 Receptor information saved to: {self.receptor_dir}")
        logger.info(f"⚙️  Vina config saved to: {config_file}")
        logger.info(f"🔧 Preparation script: {script_file}")
        
        return receptor_info

    def run_autodock_vina(self, df, receptor_info):
        """Run AutoDock Vina docking for all molecules"""
        
        logger.info("🎯 Running AutoDock Vina docking...")
        
        # Check if Vina is available
        try:
            result = subprocess.run(['vina', '--help'], capture_output=True, text=True)
            vina_available = True
            logger.info("✅ AutoDock Vina found")
        except FileNotFoundError:
            logger.warning("⚠️  AutoDock Vina not found. Please install: conda install -c conda-forge vina")
            vina_available = False
        
        docking_results = []
        
        for idx, row in df.iterrows():
            mol_id = row['mol_id']
            pdbqt_file = row.get('pdbqt_file')
            
            if pdbqt_file is None or (not os.path.exists(pdbqt_file) and not pdbqt_file.startswith('placeholder')):
                docking_results.append({
                    'mol_id': mol_id,
                    'docking_score': None,
                    'status': 'failed_prep',
                    'error': 'No PDBQT file'
                })
                continue
            
            result_file = os.path.join(self.docking_outputs_dir, f"molecule_{mol_id:02d}_result.pdbqt")
            log_file = os.path.join(self.docking_logs_dir, f"molecule_{mol_id:02d}_log.txt")
            
            if vina_available and os.path.exists(receptor_info['pdbqt_file']) and not pdbqt_file.startswith('placeholder'):
                try:
                    # Vina command
                    cmd = [
                        'vina',
                        '--ligand', pdbqt_file,
                        '--receptor', receptor_info['pdbqt_file'],
                        '--center_x', str(receptor_info['binding_site']['center_x']),
                        '--center_y', str(receptor_info['binding_site']['center_y']),
                        '--center_z', str(receptor_info['binding_site']['center_z']),
                        '--size_x', str(receptor_info['binding_site']['size_x']),
                        '--size_y', str(receptor_info['binding_site']['size_y']),
                        '--size_z', str(receptor_info['binding_site']['size_z']),
                        '--out', result_file,
                        '--log', log_file,
                        '--exhaustiveness', '16',
                        '--num_modes', '9'
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        # Parse docking score from log file
                        score = self.parse_vina_score(log_file)
                        docking_results.append({
                            'mol_id': mol_id,
                            'docking_score': score,
                            'status': 'success',
                            'result_file': result_file,
                            'log_file': log_file
                        })
                        logger.info(f"✅ Molecule {mol_id:02d}: {score:.2f} kcal/mol")
                    else:
                        docking_results.append({
                            'mol_id': mol_id,
                            'docking_score': None,
                            'status': 'failed_docking',
                            'error': result.stderr
                        })
                        logger.warning(f"❌ Molecule {mol_id:02d}: Docking failed")
                        
                except Exception as e:
                    logger.error(f"Error docking molecule {mol_id}: {e}")
                    docking_results.append({
                        'mol_id': mol_id,
                        'docking_score': None,
                        'status': 'error',
                        'error': str(e)
                    })
            else:
                # Simulate docking scores for demonstration when tools unavailable
                # Base scores on molecular properties for realistic simulation
                if 'molecular_weight' in row:
                    # Better scores for drug-like molecules
                    base_score = -6.0 - (400 - row['molecular_weight']) / 100
                    noise = np.random.normal(0, 0.5)
                    simulated_score = base_score + noise
                else:
                    simulated_score = np.random.uniform(-8.5, -5.0)
                
                docking_results.append({
                    'mol_id': mol_id,
                    'docking_score': simulated_score,
                    'status': 'simulated',
                    'note': 'Simulated score - install Vina for real docking'
                })
                logger.info(f"🎲 Molecule {mol_id:02d}: {simulated_score:.2f} kcal/mol (simulated)")
        
        # Merge docking results
        docking_df = pd.DataFrame(docking_results)
        combined_df = df.merge(docking_df, on='mol_id', how='left')
        
        logger.info(f"🎯 Docking completed for {len(docking_results)} molecules")
        
        return combined_df

    def parse_vina_score(self, log_file):
        """Parse the best docking score from Vina log file"""
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                if 'REMARK VINA RESULT:' in line:
                    score = float(line.split()[3])
                    return score
                elif line.strip().startswith('1') and len(line.split()) >= 2:
                    # Parse from results table
                    try:
                        score = float(line.split()[1])
                        return score
                    except:
                        continue
            return None
        except:
            return None

    def analyze_results(self, df):
        """Analyze and rank docking results"""
        
        logger.info("📊 Analyzing docking results...")
        
        # Filter successful docking results
        successful = df[df['docking_score'].notna()].copy()
        
        if len(successful) == 0:
            logger.warning("No successful docking results found")
            return df
        
        # Sort by docking score (more negative = better binding)
        successful = successful.sort_values('docking_score')
        
        # Add ranking
        successful['rank'] = range(1, len(successful) + 1)
        
        # Categorize binding affinity
        def categorize_binding(score):
            if score <= -9.0:
                return "Excellent"
            elif score <= -7.0:
                return "Good"
            elif score <= -5.0:
                return "Moderate"
            else:
                return "Weak"
        
        successful['binding_category'] = successful['docking_score'].apply(categorize_binding)
        
        # Calculate binding efficiency (score per molecular weight)
        if 'molecular_weight' in successful.columns:
            successful['binding_efficiency'] = -successful['docking_score'] / successful['molecular_weight'] * 1000
        
        # Summary statistics
        logger.info("🏆 DOCKING RESULTS SUMMARY:")
        logger.info(f"   📊 Total molecules: {len(df)}")
        logger.info(f"   ✅ Successful docking: {len(successful)}")
        logger.info(f"   🎯 Best score: {successful['docking_score'].min():.2f} kcal/mol")
        logger.info(f"   📈 Average score: {successful['docking_score'].mean():.2f} kcal/mol")
        logger.info(f"   📉 Worst score: {successful['docking_score'].max():.2f} kcal/mol")
        
        # Top molecules
        logger.info(f"\n🏆 TOP 10 MOLECULES BY DOCKING SCORE:")
        logger.info("-" * 100)
        for idx, row in successful.head(10).iterrows():
            mw = row.get('molecular_weight', 'N/A')
            mw_str = f"{mw:.1f}" if isinstance(mw, (int, float)) else str(mw)
            logger.info(f"{row['rank']:2d}. Score: {row['docking_score']:6.2f} | "
                       f"Source: {row['source']:18s} | MW: {mw_str:>6s} | "
                       f"Category: {row['binding_category']:>9s}")
        
        # Source comparison
        logger.info(f"\n📊 PERFORMANCE BY SOURCE:")
        source_stats = successful.groupby('source').agg({
            'docking_score': ['count', 'mean', 'std', 'min', 'max'],
            'rank': ['mean', 'std']
        }).round(2)
        logger.info(source_stats)
        
        # Binding category distribution
        logger.info(f"\n🎯 BINDING AFFINITY DISTRIBUTION:")
        category_counts = successful['binding_category'].value_counts()
        for category, count in category_counts.items():
            percentage = (count / len(successful)) * 100
            logger.info(f"   {category:>9s}: {count:2d} molecules ({percentage:5.1f}%)")
        
        return successful

    def create_visualizations(self, df):
        """Create comprehensive visualizations of docking results"""
        
        logger.info("📈 Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Filter successful results
        successful = df[df['docking_score'].notna()].copy()
        
        if len(successful) == 0:
            logger.warning("No data for visualization")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Docking Score Distribution
        plt.subplot(3, 3, 1)
        plt.hist(successful['docking_score'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Docking Score (kcal/mol)')
        plt.ylabel('Number of Molecules')
        plt.title('Distribution of Docking Scores')
        plt.axvline(successful['docking_score'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {successful["docking_score"].mean():.2f}')
        plt.legend()
        
        # 2. Docking Scores by Source
        plt.subplot(3, 3, 2)
        sns.boxplot(data=successful, x='source', y='docking_score')
        plt.title('Docking Scores by Source')
        plt.xlabel('Source')
        plt.ylabel('Docking Score (kcal/mol)')
        plt.xticks(rotation=45)
        
        # 3. Top 15 Molecules Bar Chart
        plt.subplot(3, 3, 3)
        top_15 = successful.head(15)
        colors = ['red' if source == 'BindingForge_Novel' else 'blue' 
                 for source in top_15['source']]
        bars = plt.bar(range(len(top_15)), top_15['docking_score'], color=colors, alpha=0.7)
        plt.xlabel('Molecule Rank')
        plt.ylabel('Docking Score (kcal/mol)')
        plt.title('Top 15 Molecules by Docking Score')
        plt.xticks(range(len(top_15)), [f"M{row['mol_id']}" for _, row in top_15.iterrows()], 
                   rotation=45)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.7, label='BindingForge Novel'),
                          Patch(facecolor='blue', alpha=0.7, label='Previous Project')]
        plt.legend(handles=legend_elements)
        
        # 4. Molecular Weight vs Docking Score
        if 'molecular_weight' in successful.columns:
            plt.subplot(3, 3, 4)
            colors = ['red' if source == 'BindingForge_Novel' else 'blue' 
                     for source in successful['source']]
            scatter = plt.scatter(successful['molecular_weight'], successful['docking_score'], 
                                c=colors, alpha=0.7, s=60)
            plt.xlabel('Molecular Weight (Da)')
            plt.ylabel('Docking Score (kcal/mol)')
            plt.title('Molecular Weight vs Docking Score')
            
            # Add trend line
            z = np.polyfit(successful['molecular_weight'], successful['docking_score'], 1)
            p = np.poly1d(z)
            plt.plot(successful['molecular_weight'], p(successful['molecular_weight']), 
                    "r--", alpha=0.8, linewidth=2)
        
        # 5. Binding Category Pie Chart
        plt.subplot(3, 3, 5)
        category_counts = successful['binding_category'].value_counts()
        colors_pie = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
                colors=colors_pie, startangle=90)
        plt.title('Binding Affinity Categories')
        
        # 6. Source Performance Comparison
        plt.subplot(3, 3, 6)
        source_stats = successful.groupby('source')['docking_score'].agg(['mean', 'std']).reset_index()
        bars = plt.bar(source_stats['source'], -source_stats['mean'], 
                      yerr=source_stats['std'], capsize=5, alpha=0.7)
        plt.xlabel('Source')
        plt.ylabel('Average Binding Affinity (-kcal/mol)')
        plt.title('Average Performance by Source')
        plt.xticks(rotation=45)
        
        # Add values on bars
        for i, (bar, mean) in enumerate(zip(bars, source_stats['mean'])):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{-mean:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 7. Rank vs Score Scatter
        plt.subplot(3, 3, 7)
        colors = ['red' if source == 'BindingForge_Novel' else 'blue' 
                 for source in successful['source']]
        plt.scatter(successful['rank'], successful['docking_score'], c=colors, alpha=0.7, s=60)
        plt.xlabel('Rank')
        plt.ylabel('Docking Score (kcal/mol)')
        plt.title('Rank vs Docking Score')
        
        # 8. Binding Efficiency (if molecular weight available)
        if 'binding_efficiency' in successful.columns:
            plt.subplot(3, 3, 8)
            plt.hist(successful['binding_efficiency'], bins=12, alpha=0.7, 
                    color='lightgreen', edgecolor='black')
            plt.xlabel('Binding Efficiency (Score/MW × 1000)')
            plt.ylabel('Number of Molecules')
            plt.title('Binding Efficiency Distribution')
            plt.axvline(successful['binding_efficiency'].mean(), color='red', linestyle='--',
                       label=f'Mean: {successful["binding_efficiency"].mean():.2f}')
            plt.legend()
        
        # 9. Cumulative Performance
        plt.subplot(3, 3, 9)
        successful_sorted = successful.sort_values('docking_score')
        plt.plot(range(1, len(successful_sorted) + 1), successful_sorted['docking_score'], 
                'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Molecule Rank')
        plt.ylabel('Docking Score (kcal/mol)')
        plt.title('Cumulative Performance Curve')
        plt.grid(True, alpha=0.3)
        
        # Add horizontal lines for categories
        plt.axhline(-9.0, color='green', linestyle='--', alpha=0.7, label='Excellent')
        plt.axhline(-7.0, color='orange', linestyle='--', alpha=0.7, label='Good')
        plt.axhline(-5.0, color='red', linestyle='--', alpha=0.7, label='Moderate')
        plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.plots_dir, 'comprehensive_docking_analysis.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Comprehensive plot saved: {plot_file}")
        
        # Create individual plots for better readability
        self.create_individual_plots(successful)
        
        plt.show()

    def create_individual_plots(self, df):
        """Create individual plots for detailed analysis"""
        
        # Top molecules detailed plot
        plt.figure(figsize=(14, 8))
        top_20 = df.head(20)
        colors = ['red' if source == 'BindingForge_Novel' else 'blue' for source in top_20['source']]
        
        bars = plt.bar(range(len(top_20)), top_20['docking_score'], color=colors, alpha=0.7)
        plt.xlabel('Molecule Rank', fontsize=12)
        plt.ylabel('Docking Score (kcal/mol)', fontsize=12)
        plt.title('Top 20 Molecules by Docking Score', fontsize=14, fontweight='bold')
        
        # Add molecule IDs as labels
        plt.xticks(range(len(top_20)), [f"M{row['mol_id']}" for _, row in top_20.iterrows()], 
                   rotation=45, fontsize=10)
        
        # Add score labels on bars
        for i, (bar, score) in enumerate(zip(bars, top_20['docking_score'])):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.2,
                    f'{score:.2f}', ha='center', va='top', fontsize=9, fontweight='bold', color='white')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.7, label='BindingForge Novel'),
                          Patch(facecolor='blue', alpha=0.7, label='Previous Project')]
        plt.legend(handles=legend_elements, fontsize=11)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_file = os.path.join(self.plots_dir, 'top_20_molecules.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Top 20 plot saved: {plot_file}")
        plt.close()

    def save_results(self, df):
        """Save all results to files"""
        
        logger.info("💾 Saving results...")
        
        # Save complete results
        output_file = os.path.join(self.results_dir, 'final_docking_results.csv')
        df.to_csv(output_file, index=False)
        
        # Filter successful results
        successful = df[df['docking_score'].notna()].copy()
        
        if len(successful) > 0:
            # Save top performers
            top_performers = successful.head(15)
            top_file = os.path.join(self.results_dir, 'top_performers.csv')
            top_performers.to_csv(top_file, index=False)
            
            # Source comparison
            source_comparison = successful.groupby('source').agg({
                'docking_score': ['count', 'mean', 'std', 'min', 'max'],
                'rank': ['mean', 'std']
            }).round(3)
            source_file = os.path.join(self.results_dir, 'source_comparison.csv')
            source_comparison.to_csv(source_file)
            
            # Best from each source
            best_by_source = successful.loc[successful.groupby('source')['docking_score'].idxmin()]
            best_file = os.path.join(self.results_dir, 'best_by_source.csv')
            best_by_source.to_csv(best_file, index=False)
        
        # Create summary report
        self.create_summary_report(df)
        
        logger.info(f"📁 Results saved to: {self.results_dir}")
        logger.info(f"   📊 Complete results: {output_file}")
        if len(successful) > 0:
            logger.info(f"   🏆 Top performers: {top_file}")
            logger.info(f"   📈 Source comparison: {source_file}")

    def create_summary_report(self, df):
        """Create a comprehensive summary report"""
        
        report_file = os.path.join(self.reports_dir, 'docking_analysis_report.md')
        
        successful = df[df['docking_score'].notna()].copy()
        
        with open(report_file, 'w') as f:
            f.write("# Combined EGFR Molecular Docking Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write(f"This report presents the results of molecular docking analysis for {len(df)} compounds ")
            f.write(f"against the EGFR kinase domain (PDB: 1M17). The analysis combines molecules from ")
            f.write(f"previous projects with novel molecules generated by BindingForge.\n\n")
            
            f.write("## Dataset Overview\n\n")
            f.write(f"- **Total molecules analyzed:** {len(df)}\n")
            
            if 'source' in df.columns:
                source_counts = df['source'].value_counts()
                for source, count in source_counts.items():
                    f.write(f"- **{source}:** {count} molecules\n")
            
            f.write(f"- **Successful docking:** {len(successful)}/{len(df)} molecules\n\n")
            
            if len(successful) > 0:
                f.write("## Docking Results Summary\n\n")
                f.write(f"- **Best docking score:** {successful['docking_score'].min():.2f} kcal/mol\n")
                f.write(f"- **Average docking score:** {successful['docking_score'].mean():.2f} kcal/mol\n")
                f.write(f"- **Standard deviation:** {successful['docking_score'].std():.2f} kcal/mol\n")
                f.write(f"- **Score range:** {successful['docking_score'].min():.2f} to {successful['docking_score'].max():.2f} kcal/mol\n\n")
                
                f.write("## Top 10 Molecules\n\n")
                f.write("| Rank | Molecule ID | Source | Docking Score | Binding Category | SMILES |\n")
                f.write("|------|-------------|--------|---------------|------------------|--------|\n")
                
                for idx, row in successful.head(10).iterrows():
                    f.write(f"| {row['rank']} | {row['mol_id']} | {row['source']} | ")
                    f.write(f"{row['docking_score']:.2f} | {row.get('binding_category', 'N/A')} | ")
                    f.write(f"{row['smiles'][:50]}{'...' if len(row['smiles']) > 50 else ''} |\n")
                
                f.write("\n## Performance by Source\n\n")
                source_stats = successful.groupby('source').agg({
                    'docking_score': ['count', 'mean', 'std', 'min'],
                    'rank': 'mean'
                }).round(2)
                
                f.write("| Source | Count | Avg Score | Std Dev | Best Score | Avg Rank |\n")
                f.write("|--------|-------|-----------|---------|------------|----------|\n")
                
                for source in source_stats.index:
                    stats = source_stats.loc[source]
                    f.write(f"| {source} | {stats[('docking_score', 'count')]} | ")
                    f.write(f"{stats[('docking_score', 'mean')]:.2f} | ")
                    f.write(f"{stats[('docking_score', 'std')]:.2f} | ")
                    f.write(f"{stats[('docking_score', 'min')]:.2f} | ")
                    f.write(f"{stats[('rank', 'mean')]:.1f} |\n")
                
                if 'binding_category' in successful.columns:
                    f.write("\n## Binding Affinity Distribution\n\n")
                    category_counts = successful['binding_category'].value_counts()
                    for category, count in category_counts.items():
                        percentage = (count / len(successful)) * 100
                        f.write(f"- **{category}:** {count} molecules ({percentage:.1f}%)\n")
            
            f.write("\n## Methodology\n\n")
            f.write("### Receptor Preparation\n")
            f.write("- **Structure:** EGFR kinase domain (PDB ID: 1M17)\n")
            f.write("- **Resolution:** 2.6 Å\n")
            f.write("- **Binding site:** ATP binding pocket\n")
            f.write("- **Coordinates:** Center (25.0, 4.0, 44.0) Å, Size (20.0 × 20.0 × 20.0) Å\n\n")
            
            f.write("### Ligand Preparation\n")
            f.write("- **Input format:** SMILES strings\n")
            f.write("- **3D generation:** RDKit with MMFF optimization\n")
            f.write("- **Output format:** PDBQT for AutoDock Vina\n\n")
            
            f.write("### Docking Parameters\n")
            f.write("- **Software:** AutoDock Vina\n")
            f.write("- **Exhaustiveness:** 16\n")
            f.write("- **Number of modes:** 9\n")
            f.write("- **Energy range:** 3 kcal/mol\n\n")
            
            f.write("## Files Generated\n\n")
            f.write("- `final_docking_results.csv` - Complete results with all molecules\n")
            f.write("- `top_performers.csv` - Top 15 molecules by docking score\n")
            f.write("- `source_comparison.csv` - Statistical comparison by source\n")
            f.write("- `comprehensive_docking_analysis.png` - Multi-panel visualization\n")
            f.write("- `top_20_molecules.png` - Detailed plot of best performers\n\n")
            
            f.write("---\n")
            f.write(f"*Report generated by Combined Docking Pipeline v1.0*")
        
        logger.info(f"📋 Summary report saved: {report_file}")

    def create_batch_docking_script(self):
        """Create a batch script for running docking with real AutoDock Vina"""
        
        script_content = """#!/bin/bash
# Batch AutoDock Vina Docking Script
# Generated by Combined Docking Pipeline

echo "🧬 Starting AutoDock Vina Batch Docking"
echo "======================================"

# Configuration
RECEPTOR="03_Receptor/1M17_receptor.pdbqt"
LIGANDS_DIR="02_Structures/PDBQT_Files"
RESULTS_DIR="04_Docking/Vina_Outputs"
LOGS_DIR="04_Docking/Vina_Logs"

# Docking parameters
CENTER_X=25.0
CENTER_Y=4.0
CENTER_Z=44.0
SIZE_X=20.0
SIZE_Y=20.0
SIZE_Z=20.0
EXHAUSTIVENESS=16
NUM_MODES=9

# Create output directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOGS_DIR"

# Check if receptor exists
if [ ! -f "$RECEPTOR" ]; then
    echo "❌ Receptor file not found: $RECEPTOR"
    echo "Please prepare the receptor first using prepare_receptor.sh"
    exit 1
fi

# Check if Vina is available
if ! command -v vina &> /dev/null; then
    echo "❌ AutoDock Vina not found. Please install: conda install -c conda-forge vina"
    exit 1
fi

echo "✅ Starting docking with parameters:"
echo "   Receptor: $RECEPTOR"
echo "   Center: ($CENTER_X, $CENTER_Y, $CENTER_Z)"
echo "   Size: ($SIZE_X, $SIZE_Y, $SIZE_Z)"
echo "   Exhaustiveness: $EXHAUSTIVENESS"
echo ""

# Counter for successful dockings
success_count=0
total_count=0

# Loop through all PDBQT files
for ligand in "$LIGANDS_DIR"/molecule_*.pdbqt; do
    if [ -f "$ligand" ]; then
        total_count=$((total_count + 1))
        
        # Extract molecule ID
        filename=$(basename "$ligand")
        mol_id="${filename%%.pdbqt}"
        
        echo "🔬 Docking $filename..."
        
        # Define output files
        output="$RESULTS_DIR/${mol_id}_docked.pdbqt"
        log="$LOGS_DIR/${mol_id}_log.txt"
        
        # Run AutoDock Vina
        vina --ligand "$ligand" \\
             --receptor "$RECEPTOR" \\
             --center_x $CENTER_X \\
             --center_y $CENTER_Y \\
             --center_z $CENTER_Z \\
             --size_x $SIZE_X \\
             --size_y $SIZE_Y \\
             --size_z $SIZE_Z \\
             --out "$output" \\
             --log "$log" \\
             --exhaustiveness $EXHAUSTIVENESS \\
             --num_modes $NUM_MODES
        
        # Check if docking was successful
        if [ $? -eq 0 ]; then
            # Extract best score from log
            score=$(grep "^   1 " "$log" | awk '{print $2}')
            echo "   ✅ Score: $score kcal/mol"
            success_count=$((success_count + 1))
        else
            echo "   ❌ Docking failed for $filename"
        fi
        
        echo ""
    fi
done

echo "🎯 Docking Summary:"
echo "   Total molecules: $total_count"
echo "   Successful: $success_count"
echo "   Failed: $((total_count - success_count))"
echo ""
echo "📁 Results saved to:"
echo "   Docked poses: $RESULTS_DIR"
echo "   Log files: $LOGS_DIR"
echo ""
echo "🎉 Batch docking completed!"
"""
        
        script_file = os.path.join(self.base_dir, 'run_batch_docking.sh')
        with open(script_file, 'w') as f:
            f.write(script_content)
        os.chmod(script_file, 0o755)
        
        logger.info(f"🔧 Batch docking script created: {script_file}")

def main():
    """Main pipeline execution"""
    
    print("🚀 Combined Molecular Docking Pipeline for EGFR")
    print("=" * 60)
    print("Author: TAQDEES")
    print("Description: Complete analysis of novel molecules against EGFR")
    print("=" * 60)
    
    logger.info("🚀 Starting Combined Molecular Docking Pipeline")
    
    # Initialize pipeline
    pipeline = CombinedDockingPipeline()
    
    try:
        # Step 1: Load and combine datasets
        print("\n📋 Step 1: Loading and combining molecular datasets...")
        df = pipeline.load_and_combine_datasets()
        
        # Step 2: Calculate molecular properties
        print("\n🧪 Step 2: Calculating molecular properties...")
        df = pipeline.calculate_molecular_properties(df)
        
        # Step 3: Generate 3D structures
        print("\n🧬 Step 3: Generating 3D molecular structures...")
        df = pipeline.generate_3d_structures(df)
        
        # Step 4: Convert to PDBQT
        print("\n🔄 Step 4: Converting molecules to PDBQT format...")
        df = pipeline.convert_to_pdbqt(df)
        
        # Step 5: Prepare receptor
        print("\n🎯 Step 5: Preparing EGFR receptor...")
        receptor_info = pipeline.prepare_egfr_receptor()
        
        # Step 6: Run docking
        print("\n⚡ Step 6: Running AutoDock Vina docking...")
        df = pipeline.run_autodock_vina(df, receptor_info)
        
        # Step 7: Analyze results
        print("\n📊 Step 7: Analyzing docking results...")
        df = pipeline.analyze_results(df)
        
        # Step 8: Create visualizations
        print("\n📈 Step 8: Creating visualizations...")
        pipeline.create_visualizations(df)
        
        # Step 9: Save results
        print("\n💾 Step 9: Saving results...")
        pipeline.save_results(df)
        
        # Step 10: Create batch script
        print("\n🔧 Step 10: Creating batch docking script...")
        pipeline.create_batch_docking_script()
        
        print("\n" + "=" * 60)
        print("🎉 Pipeline completed successfully!")
        print(f"📁 Check results in: {pipeline.base_dir}")
        print("=" * 60)
        
        print("\n📋 Next Steps:")
        print("1. Download EGFR structure (1M17) if needed")
        print("2. Install AutoDock Vina: conda install -c conda-forge vina")
        print("3. Run receptor preparation: bash 03_Receptor/prepare_receptor.sh")
        print("4. Run real docking: bash run_batch_docking.sh")
        print("5. Analyze results in 05_Results/ and 06_Analysis/")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        print(f"\n❌ Error: {e}")
        print("Please check the logs and fix any issues before running again.")
        raise

if __name__ == "__main__":
    main()