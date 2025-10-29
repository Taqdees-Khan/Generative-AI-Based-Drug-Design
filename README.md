# Supporting Information for "Generative AI-Based Drug Design: Target-Aware Molecular Generation for EGFR Type I Inhibitors with Multi-Platform Docking Validation"

This document maps the supporting information files to their corresponding figures, tables, and analyses in the paper.

## Directory Structure
```
supporting information/
├── dataset/               # Raw and processed data
├── figures/              # Supplementary figures and analyses
├── final_results/        # Analysis results and validations
├── paper_figures/        # Main manuscript figures in sequence
└── graphical-abstract.pdf # Graphical abstract for the paper
```

## Figure Mapping

### Main Manuscript Figures (paper_figures/)
These files are organized in sequence as they appear in the manuscript:

- **graphical-abstract.pdf**: Overview of the research methodology
- **figure-1.pdf**: LSTM Architecture and Training Dynamics
- **figure2.pdf**: System Architecture Overview
- **figure3.pdf**: Dataset Property Correlation Matrix
- **figure4.pdf**: PyMOL Visualization Summary
- **figure4a,b,c,d,e.pdf**: Detailed PyMOL Visualizations

### Supplementary Figures
- **Figure 1**: LSTM Architecture
  - Location: `figures/figure_1_vina_correlation.png`
  
- **Figure 2**: System Architecture
  - Location: `figures/figure_2_cbdock_correlation.png`
  
- **Figure 3**: Dataset Property Correlation Matrix
  - Location: `figures/Property Correlation Matrix.png`
  
- **Figure 4**: Generated Molecules
  - Location: `final_results/molecular_structures/`
  - Files: MOL1.png through MOL5.png
  - Advanced visualizations: ADVMOL2.png through ADVMOL5.png

### Supplementary Figures
- **Figure S1**: PyMOL Molecular Visualizations
  - Location: `final_results/molecular_structures/PYMOL MOL1-5.png`
  
- **Figure S2**: Comprehensive Correlation Analysis
  - Location: `figures/figure_9_comprehensive_comparison.png`
  
- **Figure S3**: Structure-based Cavity Analysis
  - Location: `final_results/cavity_analysis/STRUCTURE BASED CAVITYDIAGRAM MOL*.png`
  
- **Figure S4**: SAR Heatmaps
  - Location: `figures/sar_heatmap.png`

## Table Mapping

- **Table 1**: Lead Candidate Properties
  - Location: `final_results/tables/COMPLETE MOLCUES DATA TABLE.png`
  
- **Table 2**: CB-Dock Cavity Detection Results
  - Location: `final_results/tables/CB DOCK TABLE.png`

### Supplementary Tables
- **Table S1**: Complete Molecular Properties and Drug-likeness Analysis
  - Location: `final_results/tables/LIPINSKI DRUG TABLE ASSESMENT TABLE.png`
  
- **Table S2**: AutoDock Vina Docking Results
  - Location: `final_results/tables/AUTODOCK VINA NAALYSIS TABLE.png`
  
- **Table S3**: CBDock Cross-validation Analysis
  - Location: `final_results/tables/CB DOCK TABLE.png`
  
- **Table S4**: PLIP Protein-Ligand Interaction Analysis
  - Location: `final_results/tables/PYMOL TABLE.png`
  
- **Table S5**: Structure-Activity Relationship Analysis
  - Location: `final_results/tables/SAR TABLE.png`

## Model Data and Training
Located in `model_data/`:

### Model Files
- `best_egfr_model.pt`: Best performing trained model
- `bindingforge_lstm_model.py`: LSTM model architecture implementation
- `tokenizer.json`: Molecule tokenizer configuration

### Training Components
- `training_pipeline.py`: Main training script
- `simple_training.py`: Basic training implementation
- `training_history.pkl`: Saved training metrics
- `training_curves.png`, `training_history.png`: Training visualizations
- Checkpoints: Sequential model checkpoints (epochs 10-100)

### Generation Tools
- `egfr_molecule_generator.py`: Main molecule generation script
- `generate_molecules.py`: Generation pipeline
- `seed_based_generator.py`: Controlled molecule generation
- `molecule_validator.py`: Structure validation tools
- `generated_molecules.csv`: Output of generation runs

### Training Logs and Results
- `training_logs/`: Detailed training logs
- `results/`: Generation and validation results

## Key Validation Files

### Docking Analysis
- **Erlotinib Validation**
  - Location: `final_results/docking_analysis/05_Results/erlotinib_validation.txt`
  - Contains: RMSD validation (4.41 Å) and docking protocol details
  
- **Docking Results**
  - Location: `final_results/docking_analysis/05_Results/final_docking_results.csv`
  - Contains: Binding affinity scores (-7.7 to -8.4 kcal/mol)

### Machine Learning Metrics
- **Model Validation**
  - Location: `final_results/docking_analysis/05_Results/model_validation_metrics.csv`
  - Contains:
    - Training loss: 0.142
    - Validation loss: 0.163
    - Test loss: 0.171
    - Generation metrics:
      - Validity: 82.6%
      - Uniqueness: 28.3%
      - Novelty: 79.5%

### Structural Analysis
- **PLIP Analysis**
  - Location: `final_results/plip_analysis/`
  - Files: PLIPMOL1.png through PLIPMOL5.png
  - Contains: Protein-ligand interaction profiles
  
- **Cavity Analysis**
  - Location: `final_results/cavity_analysis/`
  - Contains: Binding site characterization for all lead molecules

## Validation Scripts
Located in `scripts/`:
- `calculate_rmsd.py`: Calculate RMSD between predicted and crystal poses
- `compile_validation_report.py`: Generate comprehensive validation reports
- `cross_dock_validation.py`: Cross-docking validation pipeline
- `extract_cocrystal_ligand.py`: Extract co-crystal ligands for validation
- `prepare_cross_dock_receptors.py`: Prepare receptor structures for cross-docking
- `run_validation_pipeline.py`: Main validation pipeline script
- `run_validation_simplified.py`: Simplified validation for quick tests

## Dataset Files
- **Raw Data**
  - Location: `dataset/raw_data/egfr_raw_chembl.csv`
  - Contains: Original ChEMBL extractions
  
- **Processed Data**
  - Location: `dataset/processed_data/egfr_type1_filtered.csv`
  - Contains: Filtered and processed EGFR inhibitors

## Code Availability
The implementation code and trained model weights are available upon reasonable request to the corresponding author.