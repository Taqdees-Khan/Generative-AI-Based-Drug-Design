#!/usr/bin/env python3
"""
AutoDock Vina Runner for Combined Molecular Analysis
Author: TAQDEES
Description: Execute AutoDock Vina docking for all molecules with EGFR
Updated for directory: C:/Users/admin/BF-final-version/Combined_EGFR_Docking_Analysis
"""

import os
import pandas as pd
import subprocess
import logging
import json
import time
from pathlib import Path
from datetime import datetime
import shutil
import glob

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutoDockVinaRunner:
    """AutoDock Vina execution and analysis"""
    
    def __init__(self, analysis_dir=r"C:\Users\admin\BF-final-version\Combined_EGFR_Docking_Analysis"):
        self.analysis_dir = Path(analysis_dir)
        self.ligands_dir = self.analysis_dir / "02_Structures" / "PDBQT_Files"
        self.receptor_dir = self.analysis_dir / "03_Receptor"
        self.results_dir = self.analysis_dir / "04_Docking" / "Vina_Outputs"
        self.logs_dir = self.analysis_dir / "04_Docking" / "Vina_Logs"
        
        # Create results directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # EGFR binding site coordinates (from your project experience)
        self.binding_site = {
            'center_x': 25.0,   # From your previous docking results
            'center_y': 4.0,
            'center_z': 44.0,
            'size_x': 20.0,     # Search space size
            'size_y': 20.0,
            'size_z': 20.0
        }
        
        logger.info(f"🎯 AutoDock Vina Runner initialized")
        logger.info(f"📁 Analysis directory: {self.analysis_dir}")
        logger.info(f"📁 Ligands directory: {self.ligands_dir}")
        logger.info(f"📁 Receptor directory: {self.receptor_dir}")

    def check_prerequisites(self):
        """Check if all required files and software are available"""
        
        logger.info("🔍 Checking prerequisites...")
        
        issues = []
        
        # Check if Vina is installed
        try:
            result = subprocess.run(['vina', '--version'], capture_output=True, text=True)
            logger.info(f"✅ AutoDock Vina found")
        except FileNotFoundError:
            issues.append("AutoDock Vina not found in PATH")
        
        # Check if Open Babel is available
        try:
            result = subprocess.run(['obabel', '--version'], capture_output=True, text=True)
            logger.info(f"✅ Open Babel found")
        except FileNotFoundError:
            issues.append("Open Babel not found in PATH")
        
        # Check directories
        if not self.analysis_dir.exists():
            issues.append(f"Analysis directory not found: {self.analysis_dir}")
        
        # Check for EGFR receptor
        receptor_files = list(self.receptor_dir.glob("*.pdbqt"))
        if not receptor_files:
            logger.warning("⚠️ No EGFR receptor PDBQT file found - will try to prepare one")
            # Look for PDB files
            pdb_files = list(self.receptor_dir.glob("*.pdb"))
            if pdb_files:
                logger.info(f"📁 Found PDB file: {pdb_files[0]}")
                return "need_conversion"
            else:
                issues.append("No EGFR receptor file found (PDB or PDBQT)")
        else:
            self.receptor_file = receptor_files[0]
            logger.info(f"✅ EGFR receptor found: {self.receptor_file}")
        
        if issues:
            logger.error("❌ Prerequisites not met:")
            for issue in issues:
                logger.error(f"   - {issue}")
            return False
        
        logger.info("✅ All prerequisites met!")
        return True

    def prepare_egfr_receptor(self):
        """Prepare EGFR receptor structure"""
        
        logger.info("📥 Preparing EGFR receptor structure...")
        
        # Create receptor directory
        self.receptor_dir.mkdir(parents=True, exist_ok=True)
        
        # Look for existing PDB files
        pdb_files = list(self.receptor_dir.glob("*.pdb"))
        pdbqt_files = list(self.receptor_dir.glob("*.pdbqt"))
        
        if pdbqt_files:
            self.receptor_file = pdbqt_files[0]
            logger.info(f"✅ EGFR receptor ready: {self.receptor_file}")
            return True
        
        # If we have PDB but no PDBQT, convert it
        if pdb_files:
            pdb_file = pdb_files[0]
            pdbqt_file = self.receptor_dir / "egfr_receptor.pdbqt"
            
            logger.info(f"🔄 Converting {pdb_file.name} to PDBQT...")
            
            try:
                # Convert to PDBQT using Open Babel
                cmd = [
                    'obabel', str(pdb_file), '-O', str(pdbqt_file),
                    '-d',  # Delete hydrogens (will be added by AutoDock)
                    '-xr'  # Remove non-protein atoms
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and pdbqt_file.exists():
                    logger.info(f"✅ Created PDBQT: {pdbqt_file}")
                    self.receptor_file = pdbqt_file
                    return True
                else:
                    logger.error(f"❌ PDBQT conversion failed: {result.stderr}")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Error converting to PDBQT: {e}")
                return False
        
        # Download EGFR structure if nothing is available
        logger.info("📥 Downloading EGFR structure (1M17)...")
        
        pdb_file = self.receptor_dir / "1M17.pdb"
        pdbqt_file = self.receptor_dir / "egfr_receptor.pdbqt"
        
        try:
            import urllib.request
            pdb_url = "https://files.rcsb.org/download/1M17.pdb"
            urllib.request.urlretrieve(pdb_url, pdb_file)
            logger.info(f"✅ Downloaded: {pdb_file}")
            
            # Convert to PDBQT
            cmd = ['obabel', str(pdb_file), '-O', str(pdbqt_file), '-d', '-xr']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.receptor_file = pdbqt_file
                logger.info(f"✅ EGFR receptor ready: {pdbqt_file}")
                return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download/prepare EGFR: {e}")
            
            # Create manual setup instructions
            instructions_file = self.receptor_dir / "MANUAL_SETUP_REQUIRED.txt"
            with open(instructions_file, 'w') as f:
                f.write("MANUAL EGFR RECEPTOR PREPARATION REQUIRED\n")
                f.write("=" * 45 + "\n\n")
                f.write("1. Download EGFR structure:\n")
                f.write("   - Go to: https://www.rcsb.org/structure/1M17\n")
                f.write("   - Download PDB file as '1M17.pdb'\n")
                f.write(f"   - Place in: {self.receptor_dir}\n\n")
                f.write("2. Prepare PDBQT file:\n")
                f.write("   - Remove water molecules and ligands\n")
                f.write("   - Convert to PDBQT using:\n")
                f.write("   - obabel 1M17.pdb -O egfr_receptor.pdbqt -xr\n\n")
                f.write("3. Re-run this script\n")
            
            logger.info(f"📄 Created setup instructions: {instructions_file}")
            return False

    def find_ligand_files(self):
        """Find all ligand files in the structures directory"""
        
        logger.info("🔍 Finding ligand files...")
        
        # Check PDBQT files first
        pdbqt_files = list(self.ligands_dir.glob("*.pdbqt"))
        
        # Also check in SDF directory
        sdf_dir = self.analysis_dir / "02_Structures" / "SDF_Files"
        sdf_files = []
        if sdf_dir.exists():
            sdf_files = list(sdf_dir.glob("*.sdf"))
        
        logger.info(f"📊 Found {len(pdbqt_files)} PDBQT files, {len(sdf_files)} SDF files")
        
        # If no PDBQT files, convert SDF files
        if not pdbqt_files and sdf_files:
            logger.info("🔄 Converting SDF files to PDBQT...")
            
            # Ensure PDBQT directory exists
            self.ligands_dir.mkdir(parents=True, exist_ok=True)
            
            for sdf_file in sdf_files:
                pdbqt_file = self.ligands_dir / sdf_file.with_suffix('.pdbqt').name
                
                try:
                    cmd = ['obabel', str(sdf_file), '-O', str(pdbqt_file), '--gen3d']
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        pdbqt_files.append(pdbqt_file)
                        logger.info(f"✅ Converted: {sdf_file.name} → {pdbqt_file.name}")
                    else:
                        logger.warning(f"⚠️ Failed to convert: {sdf_file.name}")
                        
                except Exception as e:
                    logger.error(f"❌ Error converting {sdf_file.name}: {e}")
        
        return sorted(pdbqt_files)

    def run_vina_docking(self, ligand_files):
        """Run AutoDock Vina for all ligand files"""
        
        logger.info(f"🎯 Starting AutoDock Vina docking for {len(ligand_files)} molecules...")
        logger.info(f"🎯 EGFR binding site: center=({self.binding_site['center_x']}, {self.binding_site['center_y']}, {self.binding_site['center_z']})")
        logger.info(f"🎯 Search box size: {self.binding_site['size_x']} x {self.binding_site['size_y']} x {self.binding_site['size_z']} Å")
        
        docking_results = []
        successful_docking = 0
        
        for i, ligand_file in enumerate(ligand_files, 1):
            mol_name = ligand_file.stem
            logger.info(f"🧬 [{i}/{len(ligand_files)}] Docking {mol_name}...")
            
            # Output files
            output_file = self.results_dir / f"{mol_name}_docked.pdbqt"
            log_file = self.logs_dir / f"{mol_name}_log.txt"
            
            # Vina command
            vina_cmd = [
                'vina',
                '--ligand', str(ligand_file),
                '--receptor', str(self.receptor_file),
                '--center_x', str(self.binding_site['center_x']),
                '--center_y', str(self.binding_site['center_y']),
                '--center_z', str(self.binding_site['center_z']),
                '--size_x', str(self.binding_site['size_x']),
                '--size_y', str(self.binding_site['size_y']),
                '--size_z', str(self.binding_site['size_z']),
                '--out', str(output_file),
                '--log', str(log_file),
                '--exhaustiveness', '8',
                '--num_modes', '9'
            ]
            
            try:
                # Run Vina
                start_time = time.time()
                result = subprocess.run(vina_cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
                end_time = time.time()
                
                if result.returncode == 0:
                    # Parse results
                    docking_score, rmsd = self.parse_vina_results(log_file)
                    
                    if docking_score is not None:
                        docking_results.append({
                            'molecule': mol_name,
                            'ligand_file': str(ligand_file),
                            'docking_score': docking_score,
                            'rmsd': rmsd,
                            'runtime': round(end_time - start_time, 2),
                            'status': 'success',
                            'output_file': str(output_file),
                            'log_file': str(log_file)
                        })
                        
                        successful_docking += 1
                        logger.info(f"   ✅ Score: {docking_score:.2f} kcal/mol (RMSD: {rmsd:.2f} Å)")
                    else:
                        logger.warning(f"   ⚠️ Could not parse docking score")
                        docking_results.append({
                            'molecule': mol_name,
                            'ligand_file': str(ligand_file),
                            'docking_score': None,
                            'status': 'parse_failed'
                        })
                else:
                    logger.error(f"   ❌ Vina failed: {result.stderr}")
                    docking_results.append({
                        'molecule': mol_name,
                        'ligand_file': str(ligand_file),
                        'docking_score': None,
                        'status': 'vina_failed',
                        'error': result.stderr
                    })
                    
            except subprocess.TimeoutExpired:
                logger.error(f"   ❌ Timeout (>5 minutes)")
                docking_results.append({
                    'molecule': mol_name,
                    'ligand_file': str(ligand_file),
                    'docking_score': None,
                    'status': 'timeout'
                })
                
            except Exception as e:
                logger.error(f"   ❌ Error: {e}")
                docking_results.append({
                    'molecule': mol_name,
                    'ligand_file': str(ligand_file),
                    'docking_score': None,
                    'status': 'error',
                    'error': str(e)
                })
        
        logger.info(f"🎉 Docking completed: {successful_docking}/{len(ligand_files)} successful")
        
        return docking_results

    def parse_vina_results(self, log_file):
        """Parse AutoDock Vina log file to extract best score"""
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Look for the results table
            lines = content.split('\n')
            
            for i, line in enumerate(lines):
                if 'mode |   affinity | dist from best mode' in line:
                    # Next line should be the header separator
                    # Line after that should be the best result
                    if i + 2 < len(lines):
                        result_line = lines[i + 2].strip()
                        if result_line and not result_line.startswith('-'):
                            parts = result_line.split()
                            if len(parts) >= 3:
                                try:
                                    score = float(parts[1])  # Affinity score
                                    rmsd = float(parts[2]) if len(parts) > 2 else 0.0
                                    return score, rmsd
                                except ValueError:
                                    pass
            
            return None, None
            
        except Exception as e:
            logger.error(f"Error parsing {log_file}: {e}")
            return None, None

    def analyze_results(self, docking_results):
        """Analyze and rank docking results"""
        
        logger.info("📊 Analyzing docking results...")
        
        # Convert to DataFrame
        df = pd.DataFrame(docking_results)
        
        # Filter successful results
        successful = df[df['status'] == 'success'].copy()
        
        if len(successful) == 0:
            logger.warning("❌ No successful docking results to analyze")
            return df
        
        # Sort by docking score (more negative = better)
        successful = successful.sort_values('docking_score')
        successful['rank'] = range(1, len(successful) + 1)
        
        # Categorize binding strength based on your project criteria
        def categorize_binding(score):
            if score <= -9.0:
                return "Excellent (≤ -9.0)"
            elif score <= -7.0:
                return "Good (-7.0 to -9.0)"
            elif score <= -5.0:
                return "Moderate (-5.0 to -7.0)"
            else:
                return "Weak (> -5.0)"
        
        successful['binding_category'] = successful['docking_score'].apply(categorize_binding)
        
        # Statistics
        logger.info("📈 DOCKING ANALYSIS RESULTS:")
        logger.info("=" * 60)
        logger.info(f"🧪 Total molecules: {len(df)}")
        logger.info(f"✅ Successful docking: {len(successful)}")
        logger.info(f"❌ Failed docking: {len(df) - len(successful)}")
        
        if len(successful) > 0:
            logger.info(f"🏆 Best score: {successful['docking_score'].min():.2f} kcal/mol")
            logger.info(f"📊 Average score: {successful['docking_score'].mean():.2f} kcal/mol")
            logger.info(f"📉 Worst score: {successful['docking_score'].max():.2f} kcal/mol")
            
            # Binding categories
            category_counts = successful['binding_category'].value_counts()
            logger.info(f"\n🎯 BINDING AFFINITY DISTRIBUTION:")
            for category, count in category_counts.items():
                logger.info(f"   {category}: {count} molecules")
        
        # Top molecules
        logger.info(f"\n🥇 TOP 10 MOLECULES BY BINDING AFFINITY:")
        logger.info("-" * 70)
        logger.info("Rank | Score (kcal/mol) | RMSD (Å) | Molecule")
        logger.info("-" * 70)
        
        for _, row in successful.head(10).iterrows():
            logger.info(f"{row['rank']:4d} | {row['docking_score']:13.2f} | {row.get('rmsd', 0):8.2f} | {row['molecule']}")
        
        return successful

    def save_results(self, results_df):
        """Save docking results to files"""
        
        logger.info("💾 Saving docking results...")
        
        # Save to 05_Results directory
        results_dir = self.analysis_dir / "05_Results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save complete results
        output_file = results_dir / "final_docking_results.csv"
        results_df.to_csv(output_file, index=False)
        logger.info(f"📊 Complete results: {output_file}")
        
        # Save top performers
        if 'rank' in results_df.columns:
            top_file = results_dir / "top_performers.csv"
            results_df.head(15).to_csv(top_file, index=False)
            logger.info(f"🏆 Top performers: {top_file}")
        
        # Create summary report
        self.create_summary_report(results_df, results_dir)
        
        logger.info(f"📁 All results saved to: {results_dir}")

    def create_summary_report(self, df, results_dir):
        """Create detailed summary report"""
        
        report_file = results_dir / "EGFR_Docking_Summary_Report.txt"
        
        with open(report_file, 'w') as f:
            f.write("COMBINED EGFR AUTODOCK VINA DOCKING ANALYSIS REPORT\n")
            f.write("=" * 65 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Project: BindingForge EGFR Novel Molecules\n")
            f.write(f"Receptor: EGFR (PDB: 1M17) - ATP binding site\n")
            f.write(f"Directory: {self.analysis_dir}\n\n")
            
            # Binding site information
            f.write("BINDING SITE PARAMETERS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Center: ({self.binding_site['center_x']}, {self.binding_site['center_y']}, {self.binding_site['center_z']})\n")
            f.write(f"Size: {self.binding_site['size_x']} x {self.binding_site['size_y']} x {self.binding_site['size_z']} Å\n")
            f.write(f"Target: ATP binding pocket (Met793, Thr790, Gln791, etc.)\n\n")
            
            # Summary statistics
            successful = df[df['status'] == 'success'] if 'status' in df.columns else df
            
            f.write("SUMMARY STATISTICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total molecules analyzed: {len(df)}\n")
            f.write(f"Successful docking: {len(successful)}\n")
            f.write(f"Success rate: {len(successful)/len(df)*100:.1f}%\n\n")
            
            if len(successful) > 0:
                f.write(f"Best binding score: {successful['docking_score'].min():.2f} kcal/mol\n")
                f.write(f"Average binding score: {successful['docking_score'].mean():.2f} kcal/mol\n")
                f.write(f"Standard deviation: {successful['docking_score'].std():.2f} kcal/mol\n\n")
                
                # Interpretation
                f.write("BINDING AFFINITY INTERPRETATION:\n")
                f.write("-" * 40 + "\n")
                f.write("Excellent (≤ -9.0 kcal/mol): Strong binding, drug-like potential\n")
                f.write("Good (-7.0 to -9.0 kcal/mol): Moderate binding, lead optimization\n")
                f.write("Moderate (-5.0 to -7.0 kcal/mol): Weak binding, structural modification needed\n")
                f.write("Weak (> -5.0 kcal/mol): Poor binding, major redesign required\n\n")
                
                # Top molecules
                f.write("TOP 15 MOLECULES BY BINDING AFFINITY:\n")
                f.write("-" * 60 + "\n")
                f.write("Rank | Score (kcal/mol) | RMSD (Å) | Runtime (s) | Molecule\n")
                f.write("-" * 60 + "\n")
                
                for _, row in successful.head(15).iterrows():
                    f.write(f"{row.get('rank', 0):4d} | {row['docking_score']:13.2f} | {row.get('rmsd', 0):8.2f} | {row.get('runtime', 0):10.1f} | {row['molecule']}\n")
                
                # Binding categories
                if 'binding_category' in successful.columns:
                    f.write(f"\nBINDING AFFINITY CATEGORIES:\n")
                    f.write("-" * 35 + "\n")
                    category_counts = successful['binding_category'].value_counts()
                    for category, count in category_counts.items():
                        f.write(f"{category}: {count} molecules\n")
                
                # Failed molecules
                failed = df[df['status'] != 'success'] if 'status' in df.columns else pd.DataFrame()
                if len(failed) > 0:
                    f.write(f"\nFAILED DOCKING:\n")
                    f.write("-" * 20 + "\n")
                    for _, row in failed.iterrows():
                        f.write(f"{row['molecule']}: {row['status']}\n")
        
        logger.info(f"📄 Summary report: {report_file}")

def main():
    """Main execution function"""
    
    logger.info("🎯 Combined EGFR AutoDock Vina Binding Affinity Analysis")
    logger.info("=" * 70)
    logger.info(f"📁 Target directory: C:\\Users\\admin\\BF-final-version\\Combined_EGFR_Docking_Analysis")
    
    # Initialize runner with your specific directory
    runner = AutoDockVinaRunner()
    
    try:
        # Check prerequisites
        prereq_status = runner.check_prerequisites()
        
        if prereq_status == "need_conversion" or prereq_status == False:
            # Try to prepare EGFR structure
            if not runner.prepare_egfr_receptor():
                logger.error("❌ Cannot proceed without EGFR receptor structure")
                logger.info("💡 Please check the MANUAL_SETUP_REQUIRED.txt file in the receptor directory")
                return
        
        # Find ligand files
        ligand_files = runner.find_ligand_files()
        
        if not ligand_files:
            logger.error("❌ No ligand files found!")
            logger.info("💡 Expected locations:")
            logger.info(f"   - PDBQT files: {runner.ligands_dir}")
            logger.info(f"   - SDF files: {runner.analysis_dir / '02_Structures' / 'SDF_Files'}")
            return
        
        logger.info(f"🧬 Found {len(ligand_files)} ligand files for docking")
        
        # Run docking
        docking_results = runner.run_vina_docking(ligand_files)
        
        # Analyze results
        results_df = runner.analyze_results(docking_results)
        
        # Save results
        runner.save_results(results_df)
        
        logger.info("🎉 EGFR docking analysis completed successfully!")
        logger.info(f"📁 Check results in: {runner.analysis_dir / '05_Results'}")
        logger.info("📊 Key files:")
        logger.info("   - final_docking_results.csv (complete data)")
        logger.info("   - top_performers.csv (best molecules)")
        logger.info("   - EGFR_Docking_Summary_Report.txt (analysis summary)")
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()