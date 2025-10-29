#!/usr/bin/env python3
"""
BindingForge Dataset Setup Script
Creates the complete directory structure for EGFR Type 1 inhibitor dataset generation.

Based on final optimized approach:
- Type 1 EGFR inhibitors only (professor's recommendation)
- LogP ≤ 3.6 filtering for albumin binding optimization
- Comprehensive binding site analysis
- Publication-quality visualizations
"""

import os
import sys
from pathlib import Path

def create_directory_structure(base_path=None):
    """
    Create the complete BindingForge directory structure.
    
    Args:
        base_path (str): Root directory for the project. If None, uses current directory.
    """
    
    if base_path is None:
        base_path = os.getcwd()  # Use current working directory
    """
    Create the complete BindingForge directory structure.
    
    Args:
        base_path (str): Root directory for the project
    """
    
    print("🚀 Setting up BindingForge Dataset Directory Structure...")
    print(f"📂 Base directory: {base_path}")
    
    # Define all directories needed for the project
    directories = [
        "",  # Base directory
        "raw_data",
        "processed_data", 
        "binding_site_data",
        "albumin_binding_analysis",
        "visualizations",
        "documentation"
    ]
    
    # Create directories
    created_count = 0
    for directory in directories:
        dir_path = Path(base_path) / directory
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            if directory == "":
                print(f"✅ Created base directory: {dir_path}")
            else:
                print(f"✅ Created subdirectory: {directory}/")
            created_count += 1
        except Exception as e:
            print(f"❌ Error creating {directory}: {e}")
            return False
    
    # Create placeholder files with descriptions
    placeholder_files = {
        "raw_data/README.md": """# Raw Data Directory
This directory contains:
- egfr_raw_chembl.csv: Raw EGFR inhibitor data from ChEMBL
- 1M17.pdb: EGFR kinase structure from Protein Data Bank
""",
        
        "processed_data/README.md": """# Processed Data Directory  
This directory contains:
- egfr_type1_filtered.csv: Type 1 inhibitors with LogP ≤ 3.6
- egfr_type1_final.csv: Final optimized dataset (~1,400 compounds)
- binding_site_features.csv: EGFR binding site characterization
- dataset_statistics.json: Comprehensive dataset statistics
""",
        
        "binding_site_data/README.md": """# Binding Site Analysis Directory
This directory contains:
- 1M17_protein_clean.pdb: Clean EGFR protein structure
- 1M17_binding_site.pdb: Extracted binding site residues
- 1M17_binding_site_residues.txt: List of 20 key binding residues
- binding_site_aggregate_features.json: Binding site properties
""",
        
        "albumin_binding_analysis/README.md": """# Albumin Binding Analysis Directory
This directory contains:
- albumin_binding_predictions.csv: Predicted albumin binding for all compounds
- logp_albumin_correlation.csv: LogP vs albumin binding analysis
- pharmacokinetic_optimization.csv: PK-optimized compound selection
""",
        
        "visualizations/README.md": """# Visualizations Directory
Contains 9 publication-quality figures:
1. activity_distribution.png - Activity range analysis
2. logp_vs_activity.png - LogP-activity relationship  
3. albumin_binding_optimization.png - Albumin binding analysis
4. binding_site_analysis.png - EGFR binding site profile
5. property_correlation_matrix.png - Molecular property correlations
6. scaffold_analysis.png - Common molecular scaffolds
7. drug_likeness_analysis.png - QED and Lipinski analysis
8. property_scatter_matrix.png - Property distribution matrix
9. dataset_summary_report.png - Overall dataset summary
""",
        
        "documentation/README.md": """# Documentation Directory
This directory contains:
- README.md: Main project documentation
- dataset_methodology.md: Dataset creation methodology
- binding_site_analysis.md: EGFR binding site characterization
- albumin_binding_report.md: Albumin binding optimization analysis
- dataset_metadata.json: Complete dataset metadata and statistics
"""
    }
    
    # Create placeholder files
    for file_path, content in placeholder_files.items():
        full_path = Path(base_path) / file_path
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"📝 Created documentation: {file_path}")
        except Exception as e:
            print(f"❌ Error creating {file_path}: {e}")
    
    # Create requirements.txt
    requirements = """# BindingForge Dataset Generation Requirements
# Core data processing
pandas>=1.5.0
numpy>=1.21.0
scipy>=1.9.0

# Chemical informatics
rdkit>=2022.9.1
chembl-webresource-client>=0.10.8

# Structural biology
biopython>=1.79

# Machine learning
scikit-learn>=1.1.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.10.0

# Utilities
tqdm>=4.64.0
requests>=2.28.0
lxml>=4.9.0

# Jupyter notebook support (optional)
jupyter>=1.0.0
ipykernel>=6.15.0
"""
    
    try:
        with open(Path(base_path) / "requirements.txt", 'w') as f:
            f.write(requirements)
        print("📦 Created requirements.txt")
    except Exception as e:
        print(f"❌ Error creating requirements.txt: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print(f"🎉 BindingForge Directory Structure Created Successfully!")
    print(f"📊 Total directories created: {created_count}")
    print(f"📝 Documentation files created: {len(placeholder_files)}")
    print(f"📂 Project root: {base_path}")
    print("="*60)
    
    print("\n🚀 Next Steps:")
    print("1. Navigate to the project directory:")
    print(f"   cd {base_path}")
    print("2. Install required packages:")
    print("   pip install -r requirements.txt")
    print("3. Run the dataset generation pipeline:")
    print("   python master_script.py")
    print("\n✨ Ready to generate your optimized EGFR dataset!")
    
    return True

def validate_setup(base_path=None):
    """
    Validate that the directory structure was created correctly.
    
    Args:
        base_path (str): Root directory to validate. If None, uses current directory.
        
    Returns:
        bool: True if validation passes
    """
    
    if base_path is None:
        base_path = os.getcwd()  # Use current working directory
    """
    Validate that the directory structure was created correctly.
    
    Args:
        base_path (str): Root directory to validate
        
    Returns:
        bool: True if validation passes
    """
    
    required_dirs = [
        "raw_data", "processed_data", "binding_site_data",
        "albumin_binding_analysis", "visualizations", "documentation"
    ]
    
    print("\n🔍 Validating directory structure...")
    
    for directory in required_dirs:
        dir_path = Path(base_path) / directory
        if dir_path.exists() and dir_path.is_dir():
            print(f"✅ {directory}/ - OK")
        else:
            print(f"❌ {directory}/ - MISSING")
            return False
    
    # Check requirements.txt
    req_path = Path(base_path) / "requirements.txt"
    if req_path.exists():
        print(f"✅ requirements.txt - OK")
    else:
        print(f"❌ requirements.txt - MISSING")
        return False
    
    print("✅ Directory structure validation PASSED!")
    return True

if __name__ == "__main__":
    # Allow custom base path via command line argument
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = None  # Will use current directory
    
    # Create directory structure
    success = create_directory_structure(base_path)
    
    if success:
        # Validate the setup
        validate_setup(base_path)
    else:
        print("❌ Failed to create directory structure!")
        sys.exit(1)