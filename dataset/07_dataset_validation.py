#!/usr/bin/env python3
"""
Dataset Validation Script
Comprehensive validation of the BindingForge EGFR Type 1 inhibitor dataset.

This script validates:
- Data quality and completeness
- Chemical structure validity
- Property distributions and outliers
- LogP ≤ 3.6 optimization compliance
- Albumin binding optimization results
- Binding site characterization quality
- Overall dataset integrity

Ensures the final dataset meets all quality standards for model training.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# RDKit imports for validation
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED
    print("✅ RDKit imported successfully")
except ImportError:
    print("❌ RDKit not found. Please install: pip install rdkit")
    exit(1)

def validate_dataset_structure(base_dir):
    """
    Validate that all required files and directories exist.
    
    Args:
        base_dir (str): Base project directory
        
    Returns:
        dict: Validation results
    """
    
    print("🔍 Validating dataset structure...")
    
    required_files = {
        'main_dataset': 'processed_data/egfr_type1_filtered.csv',
        'raw_chembl': 'raw_data/egfr_raw_chembl.csv',
        'pdb_structure': 'raw_data/1M17.pdb',
        'binding_site_features': 'binding_site_data/binding_site_features.csv',
        'binding_site_summary': 'binding_site_data/binding_site_analysis_summary.json',
        'albumin_predictions': 'albumin_binding_analysis/egfr_type1_albumin_binding.csv',
        'albumin_stats': 'albumin_binding_analysis/albumin_binding_statistics.json'
    }
    
    results = {
        'structure_valid': True,
        'missing_files': [],
        'existing_files': [],
        'file_sizes': {}
    }
    
    for file_type, file_path in required_files.items():
        full_path = Path(base_dir) / file_path
        
        if full_path.exists():
            try:
                size = full_path.stat().st_size
                results['existing_files'].append(file_type)
                results['file_sizes'][file_type] = size
                print(f"✅ {file_type}: {file_path} ({size//1024}KB)")
            except Exception as e:
                print(f"⚠️ {file_type}: {file_path} - Error reading: {e}")
                results['missing_files'].append(file_type)
                results['structure_valid'] = False
        else:
            print(f"❌ {file_type}: {file_path} - MISSING")
            results['missing_files'].append(file_type)
            results['structure_valid'] = False
    
    return results

def validate_chemical_structures(df):
    """
    Validate SMILES strings and calculate molecular properties.
    
    Args:
        df (pd.DataFrame): Dataset with SMILES column
        
    Returns:
        dict: Chemical validation results
    """
    
    print("🧪 Validating chemical structures...")
    
    results = {
        'total_molecules': len(df),
        'valid_smiles': 0,
        'invalid_smiles': 0,
        'property_validation': {},
        'outliers': {},
        'chemical_diversity': {}
    }
    
    valid_molecules = []
    invalid_molecules = []
    
    # Validate each SMILES
    for idx, row in df.iterrows():
        smiles = row['smiles']
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                results['valid_smiles'] += 1
                valid_molecules.append(idx)
            else:
                results['invalid_smiles'] += 1
                invalid_molecules.append((idx, smiles))
        except Exception as e:
            results['invalid_smiles'] += 1
            invalid_molecules.append((idx, f"Error: {e}"))
    
    print(f"✅ Valid SMILES: {results['valid_smiles']}/{results['total_molecules']} ({results['valid_smiles']/results['total_molecules']*100:.1f}%)")
    
    if results['invalid_smiles'] > 0:
        print(f"⚠️ Invalid SMILES: {results['invalid_smiles']}")
        if len(invalid_molecules) <= 10:  # Show first 10 invalid
            for idx, issue in invalid_molecules[:10]:
                print(f"   Row {idx}: {issue}")
    
    # Validate molecular properties
    if 'MW' in df.columns:
        mw_valid = df['MW'].between(100, 1000).sum()
        mw_outliers = len(df) - mw_valid
        results['property_validation']['MW'] = {
            'valid_range': mw_valid,
            'outliers': mw_outliers,
            'min': float(df['MW'].min()),
            'max': float(df['MW'].max()),
            'mean': float(df['MW'].mean())
        }
        print(f"⚖️ Molecular Weight: {mw_valid}/{len(df)} in range 100-1000 Da")
    
    if 'LogP' in df.columns:
        logp_valid = df['LogP'].between(-2, 6).sum()
        logp_le_36 = (df['LogP'] <= 3.6).sum()
        results['property_validation']['LogP'] = {
            'valid_range': logp_valid,
            'logp_le_36': logp_le_36,
            'logp_le_36_percent': logp_le_36 / len(df) * 100,
            'min': float(df['LogP'].min()),
            'max': float(df['LogP'].max()),
            'mean': float(df['LogP'].mean())
        }
        print(f"🧪 LogP: {logp_valid}/{len(df)} in reasonable range")
        print(f"🎯 LogP ≤ 3.6: {logp_le_36}/{len(df)} ({logp_le_36/len(df)*100:.1f}%)")
    
    if 'QED' in df.columns:
        qed_good = (df['QED'] >= 0.3).sum()
        results['property_validation']['QED'] = {
            'drug_like': qed_good,
            'drug_like_percent': qed_good / len(df) * 100,
            'min': float(df['QED'].min()),
            'max': float(df['QED'].max()),
            'mean': float(df['QED'].mean())
        }
        print(f"💊 QED ≥ 0.3: {qed_good}/{len(df)} ({qed_good/len(df)*100:.1f}%)")
    
    # Chemical diversity analysis
    if 'scaffold' in df.columns:
        unique_scaffolds = df['scaffold'].nunique()
        scaffold_counts = df['scaffold'].value_counts()
        results['chemical_diversity'] = {
            'unique_scaffolds': unique_scaffolds,
            'diversity_ratio': unique_scaffolds / len(df),
            'top_scaffold_frequency': scaffold_counts.iloc[0] / len(df) if len(scaffold_counts) > 0 else 0
        }
        print(f"🧬 Scaffold diversity: {unique_scaffolds} unique scaffolds ({unique_scaffolds/len(df)*100:.1f}% diversity)")
    
    return results

def validate_activity_data(df):
    """
    Validate biological activity data.
    
    Args:
        df (pd.DataFrame): Dataset with activity columns
        
    Returns:
        dict: Activity validation results
    """
    
    print("🎯 Validating activity data...")
    
    results = {
        'activity_validation': {},
        'activity_distribution': {},
        'potency_analysis': {}
    }
    
    if 'standard_value' in df.columns:
        # Basic activity validation
        valid_activities = df['standard_value'].notna().sum()
        positive_activities = (df['standard_value'] > 0).sum()
        
        # Activity range analysis
        activity_stats = {
            'total_values': valid_activities,
            'positive_values': positive_activities,
            'min_activity': float(df['standard_value'].min()),
            'max_activity': float(df['standard_value'].max()),
            'median_activity': float(df['standard_value'].median()),
            'mean_activity': float(df['standard_value'].mean())
        }
        
        # Potency categories
        highly_active = (df['standard_value'] <= 100).sum()  # ≤ 100 nM
        moderately_active = ((df['standard_value'] > 100) & (df['standard_value'] <= 1000)).sum()
        weakly_active = (df['standard_value'] > 1000).sum()
        
        potency_analysis = {
            'highly_active_le_100nM': highly_active,
            'moderately_active_100_1000nM': moderately_active,
            'weakly_active_gt_1000nM': weakly_active,
            'highly_active_percent': highly_active / len(df) * 100,
            'moderately_active_percent': moderately_active / len(df) * 100,
            'weakly_active_percent': weakly_active / len(df) * 100
        }
        
        results['activity_validation'] = activity_stats
        results['potency_analysis'] = potency_analysis
        
        print(f"📊 Valid activities: {valid_activities}/{len(df)}")
        print(f"🎯 Activity range: {activity_stats['min_activity']:.1f} - {activity_stats['max_activity']:.1f} nM")
        print(f"⚡ Highly active (≤100 nM): {highly_active} ({potency_analysis['highly_active_percent']:.1f}%)")
        print(f"🟡 Moderately active (100-1000 nM): {moderately_active} ({potency_analysis['moderately_active_percent']:.1f}%)")
        print(f"🔵 Weakly active (>1000 nM): {weakly_active} ({potency_analysis['weakly_active_percent']:.1f}%)")
    
    return results

def validate_albumin_optimization(df, albumin_stats_file):
    """
    Validate albumin binding optimization results.
    
    Args:
        df (pd.DataFrame): Dataset with albumin binding data
        albumin_stats_file (str): Path to albumin statistics file
        
    Returns:
        dict: Albumin validation results
    """
    
    print("💊 Validating albumin binding optimization...")
    
    results = {
        'albumin_validation': {},
        'optimization_success': False,
        'category_distribution': {}
    }
    
    try:
        # Load albumin statistics if available
        if Path(albumin_stats_file).exists():
            with open(albumin_stats_file) as f:
                albumin_stats = json.load(f)
            
            # Check optimization results
            if 'logp_optimization_results' in albumin_stats:
                opt_results = albumin_stats['logp_optimization_results']
                reduction_achieved = opt_results.get('reduction_percentage', 0)
                target_achieved = opt_results.get('target_achieved', False)
                
                results['optimization_success'] = target_achieved
                results['albumin_validation'] = {
                    'reduction_percentage': reduction_achieved,
                    'target_achieved': target_achieved,
                    'high_binding_before': opt_results.get('high_binding_before_filter', 0),
                    'high_binding_after': opt_results.get('high_binding_after_filter', 0)
                }
                
                print(f"🎯 Albumin binding reduction: {reduction_achieved:.1f}%")
                print(f"✅ Target achieved: {'YES' if target_achieved else 'NO'}")
    
    except Exception as e:
        print(f"⚠️ Could not load albumin statistics: {e}")
    
    # Validate albumin binding categories if present
    if 'albumin_binding_category' in df.columns:
        category_dist = df['albumin_binding_category'].value_counts()
        category_percentages = (category_dist / len(df) * 100).round(1)
        
        results['category_distribution'] = category_percentages.to_dict()
        
        print(f"📊 Albumin binding distribution:")
        for category, percentage in category_percentages.items():
            count = category_dist[category]
            print(f"   {category}: {count} ({percentage:.1f}%)")
        
        # Check for successful optimization (should have more low binders)
        low_medium_binders = category_dist.get('Low', 0) + category_dist.get('Low-Medium', 0)
        low_medium_percent = (low_medium_binders / len(df)) * 100
        
        if low_medium_percent >= 60:  # Target: majority in low/low-medium
            print(f"✅ Albumin optimization successful: {low_medium_percent:.1f}% in Low/Low-Medium categories")
        else:
            print(f"⚠️ Albumin optimization needs improvement: {low_medium_percent:.1f}% in Low/Low-Medium categories")
    
    return results

def validate_binding_site_data(binding_site_file, binding_summary_file):
    """
    Validate EGFR binding site characterization.
    
    Args:
        binding_site_file (str): Path to binding site features CSV
        binding_summary_file (str): Path to binding site summary JSON
        
    Returns:
        dict: Binding site validation results
    """
    
    print("🧬 Validating EGFR binding site data...")
    
    results = {
        'binding_site_valid': False,
        'residue_count': 0,
        'physicochemical_properties': {},
        'composition_analysis': {}
    }
    
    try:
        # Load binding site features
        if Path(binding_site_file).exists():
            df_binding = pd.read_csv(binding_site_file)
            results['residue_count'] = len(df_binding)
            
            # Validate expected columns
            required_columns = ['chain', 'residue_id', 'residue_name', 'hydrophobicity', 'charge']
            missing_columns = [col for col in required_columns if col not in df_binding.columns]
            
            if not missing_columns:
                results['binding_site_valid'] = True
                
                # Calculate properties
                results['physicochemical_properties'] = {
                    'avg_hydrophobicity': float(df_binding['hydrophobicity'].mean()),
                    'net_charge': int(df_binding['charge'].sum()),
                    'hydrophobicity_range': [float(df_binding['hydrophobicity'].min()), 
                                           float(df_binding['hydrophobicity'].max())],
                    'total_h_donors': int(df_binding['h_donor'].sum()) if 'h_donor' in df_binding.columns else 0,
                    'total_h_acceptors': int(df_binding['h_acceptor'].sum()) if 'h_acceptor' in df_binding.columns else 0
                }
                
                # Composition analysis
                if 'category' in df_binding.columns:
                    composition = df_binding['category'].value_counts()
                    results['composition_analysis'] = composition.to_dict()
                
                print(f"✅ Binding site data valid: {len(df_binding)} residues")
                print(f"🧪 Average hydrophobicity: {results['physicochemical_properties']['avg_hydrophobicity']:.2f}")
                print(f"⚡ Net charge: {results['physicochemical_properties']['net_charge']}")
                
                if results['composition_analysis']:
                    print(f"📊 Residue composition:")
                    for category, count in results['composition_analysis'].items():
                        percentage = (count / len(df_binding)) * 100
                        print(f"   {category}: {count} ({percentage:.1f}%)")
            else:
                print(f"❌ Missing required columns: {missing_columns}")
        else:
            print(f"❌ Binding site file not found: {binding_site_file}")
    
    except Exception as e:
        print(f"❌ Error validating binding site data: {e}")
    
    # Validate binding site summary
    try:
        if Path(binding_summary_file).exists():
            with open(binding_summary_file) as f:
                summary = json.load(f)
            
            if 'binding_site_analysis' in summary:
                analysis = summary['binding_site_analysis']
                expected_residues = analysis.get('total_binding_site_residues', 0)
                
                if expected_residues == results['residue_count']:
                    print(f"✅ Binding site summary consistent: {expected_residues} residues")
                else:
                    print(f"⚠️ Binding site summary inconsistent: {expected_residues} vs {results['residue_count']}")
    
    except Exception as e:
        print(f"⚠️ Could not validate binding site summary: {e}")
    
    return results

def generate_validation_report(base_dir, validation_results):
    """
    Generate comprehensive validation report.
    
    Args:
        base_dir (str): Base project directory
        validation_results (dict): All validation results
    """
    
    print("📋 Generating validation report...")
    
    # Calculate overall score
    scores = {
        'structure': 100 if validation_results['structure']['structure_valid'] else 0,
        'chemistry': 0,
        'activity': 0,
        'albumin': 0,
        'binding_site': 0
    }
    
    # Chemistry score
    if 'chemistry' in validation_results:
        chem = validation_results['chemistry']
        validity_score = (chem['valid_smiles'] / chem['total_molecules']) * 100
        scores['chemistry'] = validity_score
    
    # Activity score
    if 'activity' in validation_results:
        act = validation_results['activity']
        if 'potency_analysis' in act:
            # Score based on having good activity distribution
            high_active = act['potency_analysis'].get('highly_active_percent', 0)
            moderate_active = act['potency_analysis'].get('moderately_active_percent', 0)
            scores['activity'] = min(100, high_active + moderate_active)
    
    # Albumin score
    if 'albumin' in validation_results:
        alb = validation_results['albumin']
        if alb['optimization_success']:
            scores['albumin'] = 100
        elif 'albumin_validation' in alb:
            reduction = alb['albumin_validation'].get('reduction_percentage', 0)
            scores['albumin'] = min(100, reduction)
    
    # Binding site score
    if 'binding_site' in validation_results:
        bs = validation_results['binding_site']
        if bs['binding_site_valid'] and bs['residue_count'] >= 15:
            scores['binding_site'] = 100
        elif bs['binding_site_valid']:
            scores['binding_site'] = 80
    
    overall_score = sum(scores.values()) / len(scores)
    
    # Generate report
    report = f"""# BindingForge Dataset Validation Report

**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Overall Quality Score:** {overall_score:.1f}/100

## Validation Summary

### Overall Status: {'✅ PASSED' if overall_score >= 80 else '⚠️ NEEDS ATTENTION' if overall_score >= 60 else '❌ FAILED'}

| Component | Score | Status |
|-----------|-------|--------|
| Structure | {scores['structure']:.1f}/100 | {'✅' if scores['structure'] >= 80 else '⚠️' if scores['structure'] >= 60 else '❌'} |
| Chemistry | {scores['chemistry']:.1f}/100 | {'✅' if scores['chemistry'] >= 80 else '⚠️' if scores['chemistry'] >= 60 else '❌'} |
| Activity | {scores['activity']:.1f}/100 | {'✅' if scores['activity'] >= 80 else '⚠️' if scores['activity'] >= 60 else '❌'} |
| Albumin Opt | {scores['albumin']:.1f}/100 | {'✅' if scores['albumin'] >= 80 else '⚠️' if scores['albumin'] >= 60 else '❌'} |
| Binding Site | {scores['binding_site']:.1f}/100 | {'✅' if scores['binding_site'] >= 80 else '⚠️' if scores['binding_site'] >= 60 else '❌'} |

## Detailed Results

### 1. Dataset Structure
"""
    
    if 'structure' in validation_results:
        struct = validation_results['structure']
        report += f"- **Files Found:** {len(struct['existing_files'])}\n"
        report += f"- **Missing Files:** {len(struct['missing_files'])}\n"
        if struct['missing_files']:
            report += f"- **Missing:** {', '.join(struct['missing_files'])}\n"
    
    if 'chemistry' in validation_results:
        chem = validation_results['chemistry']
        report += f"""
### 2. Chemical Structure Validation
- **Total Molecules:** {chem['total_molecules']:,}
- **Valid SMILES:** {chem['valid_smiles']:,} ({chem['valid_smiles']/chem['total_molecules']*100:.1f}%)
- **Invalid SMILES:** {chem['invalid_smiles']}
"""
        
        if 'property_validation' in chem:
            prop = chem['property_validation']
            if 'LogP' in prop:
                report += f"- **LogP ≤ 3.6:** {prop['LogP']['logp_le_36']} ({prop['LogP']['logp_le_36_percent']:.1f}%)\n"
            if 'QED' in prop:
                report += f"- **Drug-like (QED ≥ 0.3):** {prop['QED']['drug_like']} ({prop['QED']['drug_like_percent']:.1f}%)\n"
        
        if 'chemical_diversity' in chem:
            div = chem['chemical_diversity']
            report += f"- **Scaffold Diversity:** {div['unique_scaffolds']} unique scaffolds ({div['diversity_ratio']*100:.1f}%)\n"
    
    if 'activity' in validation_results:
        act = validation_results['activity']
        if 'potency_analysis' in act:
            pot = act['potency_analysis']
            report += f"""
### 3. Activity Data Validation
- **Highly Active (≤100 nM):** {pot['highly_active_le_100nM']} ({pot['highly_active_percent']:.1f}%)
- **Moderately Active (100-1000 nM):** {pot['moderately_active_100_1000nM']} ({pot['moderately_active_percent']:.1f}%)
- **Weakly Active (>1000 nM):** {pot['weakly_active_gt_1000nM']} ({pot['weakly_active_percent']:.1f}%)
"""
    
    if 'albumin' in validation_results:
        alb = validation_results['albumin']
        if 'albumin_validation' in alb:
            alb_val = alb['albumin_validation']
            report += f"""
### 4. Albumin Binding Optimization
- **Reduction Achieved:** {alb_val['reduction_percentage']:.1f}%
- **Target Achieved:** {'✅ YES' if alb_val['target_achieved'] else '❌ NO'}
- **High Binders Before:** {alb_val['high_binding_before']:.1f}%
- **High Binders After:** {alb_val['high_binding_after']:.1f}%
"""
        
        if 'category_distribution' in alb:
            report += "**Category Distribution:**\n"
            for category, percentage in alb['category_distribution'].items():
                report += f"- {category}: {percentage:.1f}%\n"
    
    if 'binding_site' in validation_results:
        bs = validation_results['binding_site']
        report += f"""
### 5. EGFR Binding Site Validation
- **Residues Identified:** {bs['residue_count']}
- **Data Valid:** {'✅ YES' if bs['binding_site_valid'] else '❌ NO'}
"""
        
        if 'physicochemical_properties' in bs:
            props = bs['physicochemical_properties']
            report += f"- **Average Hydrophobicity:** {props['avg_hydrophobicity']:.2f}\n"
            report += f"- **Net Charge:** {props['net_charge']}\n"
            report += f"- **H-bond Donors:** {props['total_h_donors']}\n"
            report += f"- **H-bond Acceptors:** {props['total_h_acceptors']}\n"
    
    report += f"""
## Quality Assessment

### Strengths
"""
    
    strengths = []
    if scores['structure'] >= 80:
        strengths.append("✅ Complete file structure")
    if scores['chemistry'] >= 90:
        strengths.append("✅ High-quality chemical structures")
    if scores['activity'] >= 80:
        strengths.append("✅ Good activity distribution")
    if scores['albumin'] >= 90:
        strengths.append("✅ Successful albumin binding optimization")
    if scores['binding_site'] >= 80:
        strengths.append("✅ Comprehensive binding site characterization")
    
    for strength in strengths:
        report += f"- {strength}\n"
    
    report += "\n### Areas for Improvement\n"
    
    improvements = []
    if scores['structure'] < 80:
        improvements.append("❌ Missing required files")
    if scores['chemistry'] < 80:
        improvements.append("❌ Chemical structure quality issues")
    if scores['activity'] < 80:
        improvements.append("❌ Activity data needs review")
    if scores['albumin'] < 80:
        improvements.append("❌ Albumin optimization incomplete")
    if scores['binding_site'] < 80:
        improvements.append("❌ Binding site data insufficient")
    
    if not improvements:
        report += "- None identified - dataset meets quality standards\n"
    else:
        for improvement in improvements:
            report += f"- {improvement}\n"
    
    report += f"""
## Recommendations

### For Model Training
- **Dataset Size:** {'✅ Suitable' if validation_results.get('chemistry', {}).get('total_molecules', 0) >= 1000 else '⚠️ Consider expanding'}
- **Chemical Diversity:** {'✅ Good' if validation_results.get('chemistry', {}).get('chemical_diversity', {}).get('diversity_ratio', 0) >= 0.1 else '⚠️ Limited'}
- **Activity Range:** {'✅ Appropriate' if scores['activity'] >= 80 else '⚠️ Review distribution'}

### For Publication
- **Data Quality:** {'✅ Publication ready' if overall_score >= 80 else '⚠️ Needs improvement'}
- **Optimization Claims:** {'✅ Validated' if scores['albumin'] >= 90 else '⚠️ Verify results'}
- **Structural Analysis:** {'✅ Complete' if scores['binding_site'] >= 80 else '⚠️ Expand analysis'}

---
*BindingForge Dataset Validation v1.0*
"""
    
    # Save report
    report_file = Path(base_dir) / "documentation" / "dataset_validation_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📋 Validation report saved: {report_file}")
    
    return overall_score

def main():
    """
    Main validation function.
    """
    
    print("🔍 BindingForge Dataset Validation")
    print("="*50)
    
    base_dir = os.getcwd()
    
    # Run all validation checks
    validation_results = {}
    
    # 1. Structure validation
    validation_results['structure'] = validate_dataset_structure(base_dir)
    
    # 2. Chemical validation
    try:
        main_dataset_file = Path(base_dir) / "processed_data" / "egfr_type1_filtered.csv"
        if main_dataset_file.exists():
            df = pd.read_csv(main_dataset_file)
            validation_results['chemistry'] = validate_chemical_structures(df)
            validation_results['activity'] = validate_activity_data(df)
        else:
            print(f"⚠️ Main dataset not found: {main_dataset_file}")
    except Exception as e:
        print(f"❌ Error loading main dataset: {e}")
    
    # 3. Albumin validation
    try:
        albumin_file = Path(base_dir) / "albumin_binding_analysis" / "egfr_type1_albumin_binding.csv"
        albumin_stats_file = Path(base_dir) / "albumin_binding_analysis" / "albumin_binding_statistics.json"
        
        if albumin_file.exists():
            df_albumin = pd.read_csv(albumin_file)
            validation_results['albumin'] = validate_albumin_optimization(df_albumin, albumin_stats_file)
        else:
            print(f"⚠️ Albumin binding file not found: {albumin_file}")
    except Exception as e:
        print(f"❌ Error validating albumin data: {e}")
    
    # 4. Binding site validation
    try:
        binding_site_file = Path(base_dir) / "binding_site_data" / "binding_site_features.csv"
        binding_summary_file = Path(base_dir) / "binding_site_data" / "binding_site_analysis_summary.json"
        
        validation_results['binding_site'] = validate_binding_site_data(binding_site_file, binding_summary_file)
    except Exception as e:
        print(f"❌ Error validating binding site data: {e}")
    
    # 5. Generate comprehensive report
    overall_score = generate_validation_report(base_dir, validation_results)
    
    # Final summary
    print(f"\n{'='*60}")
    print("🎯 DATASET VALIDATION COMPLETE")
    print(f"{'='*60}")
    
    print(f"📊 Overall Quality Score: {overall_score:.1f}/100")
    
    if overall_score >= 80:
        print("✅ DATASET VALIDATION PASSED!")
        print("🎉 Dataset meets quality standards for model training")
        print("📄 Ready for publication and research use")
    elif overall_score >= 60:
        print("⚠️ DATASET VALIDATION PARTIAL")
        print("🔧 Some issues detected - review validation report")
        print("📋 Address recommendations before model training")
    else:
        print("❌ DATASET VALIDATION FAILED")
        print("🚨 Significant issues detected")
        print("🔧 Major improvements needed before use")
    
    # Specific recommendations
    print(f"\n📋 Key Recommendations:")
    
    if validation_results.get('structure', {}).get('structure_valid', False):
        print("✅ File structure complete")
    else:
        print("❌ Fix missing files and directory structure")
    
    if validation_results.get('chemistry', {}).get('valid_smiles', 0) >= validation_results.get('chemistry', {}).get('total_molecules', 1) * 0.95:
        print("✅ Chemical structures high quality")
    else:
        print("⚠️ Review and fix invalid SMILES structures")
    
    if validation_results.get('albumin', {}).get('optimization_success', False):
        print("✅ Albumin binding optimization successful")
    else:
        print("⚠️ Verify albumin binding optimization results")
    
    if validation_results.get('binding_site', {}).get('binding_site_valid', False):
        print("✅ Binding site characterization complete")
    else:
        print("❌ Complete binding site extraction and analysis")
    
    print(f"\n📁 Validation report saved to:")
    print(f"   documentation/dataset_validation_report.md")
    
    return overall_score >= 60  # Return success if score >= 60

if __name__ == "__main__":
    # Set paths
    print("🚀 BindingForge Dataset Validation")
    print("="*50)
    
    try:
        success = main()
        if success:
            print("\n✅ Dataset validation completed successfully!")
            print("🚀 Ready for model implementation phase!")
        else:
            print("\n⚠️ Dataset validation completed with issues.")
            print("📋 Please review the validation report and address recommendations.")
        
        print("\n🎯 Next step: Review documentation and proceed with model training")
        
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        print("🔧 Please check file paths and dependencies")
        exit(1)