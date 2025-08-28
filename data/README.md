# Data Directory

This directory contains input data and configuration files for BindingForge.

## Files:
- `grid_params.txt` - AutoDock Vina grid parameters for EGFR binding site
- `egfr_receptor.pdb` - Prepared EGFR receptor structure (1M17)
- `training_dataset.csv` - IC50 training data from ChEMBL
- `validation_molecules.sdf` - Test molecules for validation

## Usage:
Place your molecular structure files (SDF, MOL2, PDBQT) in this directory for docking analysis.

## Data Sources:
- EGFR structure: PDB ID 1M17 (erlotinib complex)
- IC50 data: ChEMBL database
- Binding site: Defined from crystal structure analysis
