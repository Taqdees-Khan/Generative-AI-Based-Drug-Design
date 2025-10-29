# Combined EGFR Docking Analysis
Generated: 2025-07-30 11:39:50

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
