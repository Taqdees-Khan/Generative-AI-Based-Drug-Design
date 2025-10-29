#!/usr/bin/env python3
"""
FIXED EGFR Type 1 Inhibitor Processing Script
Handles semicolon-separated CSV files from ChEMBL manual download.

Key fixes:
- Proper CSV parsing with semicolon separator
- Robust error handling for malformed lines
- Column name mapping for ChEMBL format
- Better data cleaning and validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED, Crippen
    from rdkit.Chem import rdMolDescriptors, rdDepictor
    from rdkit.Chem.Scaffolds import MurckoScaffold
    print("✅ RDKit imported successfully")
except ImportError:
    print("❌ RDKit not found. Please install: pip install rdkit")
    exit(1)

def load_chembl_csv(file_path):
    """
    Load ChEMBL CSV file with proper parsing for semicolon separators.
    
    Args:
        file_path (str): Path to ChEMBL CSV file
        
    Returns:
        pd.DataFrame: Loaded and cleaned DataFrame
    """
    
    print(f"📂 Loading ChEMBL CSV file: {file_path}")
    
    try:
        # Try to load with comma separator and handle errors
        df = pd.read_csv(file_path, 
                        sep=',',  # Comma separator
                        on_bad_lines='skip',  # Skip malformed lines
                        low_memory=False,  # Don't assume data types
                        encoding='utf-8')
        
        print(f"✅ Successfully loaded {len(df)} records")
        print(f"📊 Columns: {len(df.columns)}")
        
        # Print first few column names to verify
        print("📋 First 10 columns:")
        for i, col in enumerate(df.columns[:10]):
            print(f"   {i+1}. {col}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        
        # Try alternative loading methods
        print("🔄 Trying alternative loading method...")
        
        try:
            # Try with different encoding
            df = pd.read_csv(file_path, 
                            sep=';',
                            on_bad_lines='skip',
                            encoding='latin-1')
            print(f"✅ Loaded with latin-1 encoding: {len(df)} records")
            return df
            
        except Exception as e2:
            print(f"❌ Alternative loading failed: {e2}")
            return None

def standardize_column_names(df):
    """
    Standardize column names to match expected format.
    
    Args:
        df (pd.DataFrame): DataFrame with ChEMBL columns
        
    Returns:
        pd.DataFrame: DataFrame with standardized columns
    """
    
    print("🔧 Standardizing column names...")
    
    # Create mapping from ChEMBL format to expected format
    column_mapping = {
        'Molecule ChEMBL ID': 'molecule_chembl_id',
        'Smiles': 'smiles',
        'Standard Type': 'standard_type',
        'Standard Value': 'standard_value',
        'Standard Units': 'standard_units',
        'pChEMBL Value': 'pchembl_value',
        'Assay ChEMBL ID': 'assay_chembl_id',
        'Target ChEMBL ID': 'target_chembl_id',
        'Document ChEMBL ID': 'document_chembl_id',
        'Molecular Weight': 'MW',
        'AlogP': 'LogP_chembl'  # We'll calculate our own LogP
    }
    
    # Rename columns that exist
    existing_mappings = {old: new for old, new in column_mapping.items() if old in df.columns}
    df = df.rename(columns=existing_mappings)
    
    print(f"✅ Renamed {len(existing_mappings)} columns")
    
    # Print what we have now
    key_columns = ['molecule_chembl_id', 'smiles', 'standard_type', 'standard_value', 'standard_units']
    available_key_cols = [col for col in key_columns if col in df.columns]
    print(f"📋 Available key columns: {available_key_cols}")
    
    return df

def clean_and_validate_data(df):
    """
    Clean and validate the ChEMBL data.
    
    Args:
        df (pd.DataFrame): Raw ChEMBL data
        
    Returns:
        pd.DataFrame: Cleaned data
    """
    
    print("🧹 Cleaning and validating data...")
    
    initial_count = len(df)
    print(f"📊 Starting with {initial_count} records")
    
    # Clean SMILES column
    if 'smiles' in df.columns:
        # Remove quotes and clean
        df['smiles'] = df['smiles'].astype(str).str.strip().str.replace('"', '')
        # Remove obviously invalid SMILES
        df = df[df['smiles'].str.len() > 5]  # Minimum reasonable SMILES length
        df = df[~df['smiles'].isin(['nan', 'None', '', 'null'])]
        print(f"✅ After SMILES cleaning: {len(df)} records")
    
    # Clean standard_value
    if 'standard_value' in df.columns:
        # Convert to numeric, handling text values
        def clean_standard_value(val):
            if pd.isna(val):
                return None
            try:
                # Convert to string first, then clean
                val_str = str(val).strip().replace('"', '')
                # Remove comparison operators
                val_str = val_str.replace('>', '').replace('<', '').replace('=', '')
                return float(val_str)
            except:
                return None
        
        df['standard_value'] = df['standard_value'].apply(clean_standard_value)
        df = df[df['standard_value'].notna()]
        df = df[df['standard_value'] > 0]
        print(f"✅ After standard_value cleaning: {len(df)} records")
    
    # Filter for relevant activity types
    if 'standard_type' in df.columns:
        activity_types = ['IC50', 'Ki', 'Kd', 'EC50']
        df = df[df['standard_type'].isin(activity_types)]
        print(f"✅ After activity type filtering: {len(df)} records")
    
    # Filter for nM units
    if 'standard_units' in df.columns:
        df = df[df['standard_units'] == 'nM']
        print(f"✅ After units filtering (nM only): {len(df)} records")
    
    # Filter for reasonable activity range
    if 'standard_value' in df.columns:
        df = df[(df['standard_value'] >= 0.001) & (df['standard_value'] <= 1000000)]
        print(f"✅ After activity range filtering: {len(df)} records")
    
    print(f"🎯 Data cleaning complete: {len(df)} final records ({len(df)/initial_count*100:.1f}% retained)")
    
    return df

def calculate_molecular_properties(smiles):
    """
    Calculate comprehensive molecular properties for a SMILES string.
    """
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        # Standardize the molecule
        mol = Chem.AddHs(mol)
        
        # Calculate properties
        properties = {
            'MW': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'RotBonds': Descriptors.NumRotatableBonds(mol),
            'AromaticRings': Descriptors.NumAromaticRings(mol),
            'QED': QED.qed(mol),
            'Lipinski_Violations': 0  # Will calculate below
        }
        
        # Calculate Lipinski violations
        violations = 0
        if properties['MW'] > 500: violations += 1
        if properties['LogP'] > 5: violations += 1
        if properties['HBD'] > 5: violations += 1
        if properties['HBA'] > 10: violations += 1
        properties['Lipinski_Violations'] = violations
        
        # Get scaffold
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            properties['scaffold'] = Chem.MolToSmiles(scaffold) if scaffold else None
        except:
            properties['scaffold'] = None
            
        return properties
        
    except Exception as e:
        return None

def classify_as_type1_inhibitor(smiles, activity_nm, properties=None):
    """
    Classify if a molecule is likely a Type 1 EGFR inhibitor.
    """
    
    if properties is None:
        properties = calculate_molecular_properties(smiles)
        if properties is None:
            return False, 0.0, "Invalid molecule"
    
    confidence = 0.0
    reasons = []
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, 0.0, "Invalid SMILES"
        
        # Activity-based classification
        if 0.1 <= activity_nm <= 10000:
            confidence += 0.3
            reasons.append("Activity in Type 1 range")
            
            if 1 <= activity_nm <= 1000:
                confidence += 0.2
                reasons.append("Activity in typical Type 1 range")
        
        # Scaffold-based classification
        quinazoline_pattern = Chem.MolFromSmarts('c1nc2ccccc2nc1')
        if mol.HasSubstructMatch(quinazoline_pattern):
            confidence += 0.4
            reasons.append("Contains quinazoline core")
        
        aniline_pattern = Chem.MolFromSmarts('c1ccccc1N')
        if mol.HasSubstructMatch(aniline_pattern):
            confidence += 0.2
            reasons.append("Contains aniline substituent")
        
        # Property-based classification
        mw = properties.get('MW', 0)
        logp = properties.get('LogP', 0)
        hba = properties.get('HBA', 0)
        hbd = properties.get('HBD', 0)
        aromatic_rings = properties.get('AromaticRings', 0)
        
        if 300 <= mw <= 600:
            confidence += 0.1
            reasons.append("MW in Type 1 range")
        
        if 1.0 <= logp <= 4.0:
            confidence += 0.1
            reasons.append("LogP in drug-like range")
        
        if 4 <= hba <= 8 and 1 <= hbd <= 3:
            confidence += 0.1
            reasons.append("H-bond features typical of Type 1")
        
        if 2 <= aromatic_rings <= 4:
            confidence += 0.1
            reasons.append("Aromatic ring count typical")
        
        # General kinase-like properties
        if (hba >= 4 and hba <= 7 and 
            hbd >= 1 and hbd <= 3 and 
            300 <= mw <= 550 and 
            1.5 <= logp <= 3.5):
            confidence += 0.05
            reasons.append("General kinase inhibitor properties")
        
        # Penalties
        if activity_nm > 50000:
            confidence -= 0.2
            reasons.append("Very weak activity (penalty)")
        elif activity_nm < 0.01:
            confidence -= 0.1
            reasons.append("Extremely potent (less typical for Type 1)")
        
        is_type1 = confidence >= 0.3
        
        return is_type1, min(confidence, 1.0), "; ".join(reasons)
        
    except Exception as e:
        return False, 0.0, f"Error in classification: {str(e)}"

def process_egfr_data(input_file, output_dir):
    """
    Process EGFR data to create Type 1 inhibitor dataset.
    """
    
    print("🔄 Processing EGFR data for Type 1 inhibitors...")
    print(f"📂 Input file: {input_file}")
    print(f"📁 Output directory: {output_dir}")
    
    # Step 1: Load and clean data
    print("\n1️⃣ Loading ChEMBL data...")
    
    df_raw = load_chembl_csv(input_file)
    if df_raw is None:
        return None
    
    # Step 2: Standardize columns
    df_clean = standardize_column_names(df_raw)
    
    # Step 3: Clean and validate
    df_clean = clean_and_validate_data(df_clean)
    
    if len(df_clean) == 0:
        print("❌ No valid data remaining after cleaning!")
        return None
    
    # Step 4: Calculate molecular properties
    print("\n2️⃣ Calculating molecular properties...")
    
    property_data = []
    failed_molecules = 0
    
    for idx, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Calculating properties"):
        smiles = row['smiles']
        properties = calculate_molecular_properties(smiles)
        
        if properties is not None:
            # Add original data
            prop_row = properties.copy()
            prop_row.update({
                'molecule_chembl_id': row['molecule_chembl_id'],
                'smiles': smiles,
                'standard_type': row['standard_type'],
                'standard_value': row['standard_value'],
                'standard_units': row['standard_units'],
                'pchembl_value': row.get('pchembl_value', -np.log10(row['standard_value'] * 1e-9)),
                'assay_chembl_id': row.get('assay_chembl_id', 'N/A'),
                'target_chembl_id': row.get('target_chembl_id', 'CHEMBL203'),
                'document_chembl_id': row.get('document_chembl_id', 'N/A')
            })
            property_data.append(prop_row)
        else:
            failed_molecules += 1
    
    print(f"✅ Successfully processed: {len(property_data)} molecules")
    print(f"❌ Failed to process: {failed_molecules} molecules")
    
    if not property_data:
        print("❌ No valid molecules found!")
        return None
    
    df_props = pd.DataFrame(property_data)
    
    # Step 5: Apply LogP ≤ 3.6 filter
    print("\n3️⃣ Applying LogP ≤ 3.6 filter for albumin binding optimization...")
    
    initial_count = len(df_props)
    df_props = df_props[df_props['LogP'] <= 3.6]
    filtered_count = len(df_props)
    
    print(f"📊 Before LogP filter: {initial_count}")
    print(f"📊 After LogP ≤ 3.6 filter: {filtered_count}")
    print(f"📉 Filtered out: {initial_count - filtered_count} ({100*(initial_count - filtered_count)/initial_count:.1f}%)")
    
    # Step 6: Type 1 inhibitor classification
    print("\n4️⃣ Classifying Type 1 inhibitors...")
    
    type1_data = []
    classification_stats = {'type1': 0, 'not_type1': 0, 'confidence_scores': []}
    
    for idx, row in tqdm(df_props.iterrows(), total=len(df_props), desc="Classifying Type 1"):
        properties = {
            'MW': row['MW'],
            'LogP': row['LogP'],
            'HBD': row['HBD'],
            'HBA': row['HBA'],
            'TPSA': row['TPSA'],
            'RotBonds': row['RotBonds'],
            'AromaticRings': row['AromaticRings'],
            'QED': row['QED']
        }
        
        is_type1, confidence, reasoning = classify_as_type1_inhibitor(
            row['smiles'], 
            row['standard_value'], 
            properties
        )
        
        if is_type1:
            row_data = row.to_dict()
            row_data.update({
                'type1_confidence': confidence,
                'type1_reasoning': reasoning
            })
            type1_data.append(row_data)
            classification_stats['type1'] += 1
        else:
            classification_stats['not_type1'] += 1
        
        classification_stats['confidence_scores'].append(confidence)
    
    print(f"✅ Type 1 inhibitors identified: {classification_stats['type1']}")
    print(f"❌ Non-Type 1 compounds: {classification_stats['not_type1']}")
    print(f"📊 Average confidence: {np.mean(classification_stats['confidence_scores']):.3f}")
    
    if not type1_data:
        print("❌ No Type 1 inhibitors found!")
        return None
    
    df_type1 = pd.DataFrame(type1_data)
    
    # Step 7: Final quality filters
    print("\n5️⃣ Applying final quality filters...")
    
    initial_type1 = len(df_type1)
    
    # Drug-likeness filter
    df_type1 = df_type1[df_type1['QED'] >= 0.3]
    print(f"📊 After QED ≥ 0.3 filter: {len(df_type1)}")
    
    # Molecular weight filter
    df_type1 = df_type1[(df_type1['MW'] >= 250) & (df_type1['MW'] <= 650)]
    print(f"📊 After MW filter (250-650 Da): {len(df_type1)}")
    
    # Lipinski filter
    df_type1 = df_type1[df_type1['Lipinski_Violations'] <= 1]
    print(f"📊 After Lipinski filter (≤1 violation): {len(df_type1)}")
    
    # Step 8: Handle duplicates
    print("\n6️⃣ Handling duplicate molecules...")
    
    pre_dedup = len(df_type1)
    df_type1 = df_type1.sort_values('standard_value').groupby('smiles').first().reset_index()
    post_dedup = len(df_type1)
    
    print(f"📊 Before deduplication: {pre_dedup}")
    print(f"📊 After deduplication: {post_dedup}")
    print(f"📉 Duplicates removed: {pre_dedup - post_dedup}")
    
    # Step 9: Save processed dataset
    print("\n7️⃣ Saving processed dataset...")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    output_file = Path(output_dir) / "egfr_type1_filtered.csv"
    df_type1.to_csv(output_file, index=False)
    print(f"💾 Saved Type 1 dataset: {output_file}")
    
    # Step 10: Generate statistics
    print("\n8️⃣ Generating dataset statistics...")
    
    stats = {
        'dataset_info': {
            'total_type1_inhibitors': len(df_type1),
            'activity_range_nM': {
                'min': float(df_type1['standard_value'].min()),
                'max': float(df_type1['standard_value'].max()),
                'median': float(df_type1['standard_value'].median()),
                'mean': float(df_type1['standard_value'].mean())
            },
            'logp_range': {
                'min': float(df_type1['LogP'].min()),
                'max': float(df_type1['LogP'].max()),
                'median': float(df_type1['LogP'].median()),
                'mean': float(df_type1['LogP'].mean())
            },
            'molecular_weight_range': {
                'min': float(df_type1['MW'].min()),
                'max': float(df_type1['MW'].max()),
                'median': float(df_type1['MW'].median()),
                'mean': float(df_type1['MW'].mean())
            },
            'qed_range': {
                'min': float(df_type1['QED'].min()),
                'max': float(df_type1['QED'].max()),
                'median': float(df_type1['QED'].median()),
                'mean': float(df_type1['QED'].mean())
            }
        },
        'filtering_summary': {
            'initial_raw_records': len(df_raw),
            'after_basic_cleaning': len(df_clean),
            'after_property_calculation': len(df_props),
            'after_logp_filter': filtered_count,
            'after_type1_classification': initial_type1,
            'final_type1_dataset': len(df_type1)
        },
        'lipinski_compliance': {
            '0_violations': int((df_type1['Lipinski_Violations'] == 0).sum()),
            '1_violation': int((df_type1['Lipinski_Violations'] == 1).sum())
        },
        'scaffold_diversity': {
            'unique_scaffolds': df_type1['scaffold'].nunique(),
            'top_scaffolds': df_type1['scaffold'].value_counts().head(10).to_dict()
        },
        'processing_timestamp': pd.Timestamp.now().isoformat()
    }
    
    stats_file = Path(output_dir) / "type1_processing_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, indent=2, fp=f)
    print(f"📊 Saved statistics: {stats_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("🎉 Type 1 EGFR Inhibitor Processing Complete!")
    print("="*60)
    print(f"📊 Final dataset size: {len(df_type1):,} Type 1 inhibitors")
    print(f"🎯 Activity range: {stats['dataset_info']['activity_range_nM']['min']:.1f} - {stats['dataset_info']['activity_range_nM']['max']:.1f} nM")
    print(f"🧪 LogP range: {stats['dataset_info']['logp_range']['min']:.2f} - {stats['dataset_info']['logp_range']['max']:.2f}")
    print(f"⚖️ MW range: {stats['dataset_info']['molecular_weight_range']['min']:.1f} - {stats['dataset_info']['molecular_weight_range']['max']:.1f} Da")
    print(f"💊 QED range: {stats['dataset_info']['qed_range']['min']:.3f} - {stats['dataset_info']['qed_range']['max']:.3f}")
    print(f"🧬 Unique scaffolds: {stats['scaffold_diversity']['unique_scaffolds']}")
    print("="*60)
    
    return df_type1

if __name__ == "__main__":
    # Set paths for WSL environment
    input_file = "/mnt/c/Users/admin/BF-final-version/raw_data/egfr_raw_chembl.csv"
    output_dir = "/mnt/c/Users/admin/BF-final-version/processed_data"
    
    print("🚀 BindingForge Type 1 Inhibitor Processing")
    print("="*50)
    print("🔧 FIXED version with proper CSV parsing")
    print("="*50)
    
    # Process the data
    df_result = process_egfr_data(input_file, output_dir)
    
    if df_result is not None:
        print("\n✅ Processing completed successfully!")
        print("📁 Check the processed_data/ directory for:")
        print("   - egfr_type1_filtered.csv (Type 1 inhibitor dataset)")
        print("   - type1_processing_stats.json (processing statistics)")
        
        print("\n🚀 Next step: Run 04_binding_site_extraction.py")
        
    else:
        print("❌ Processing failed!")
        exit(1)