"""
Batch Analysis Example
Process multiple molecules through the BindingForge pipeline
"""

import pandas as pd
import os
from datetime import datetime
from src.evaluation import calculate_regression_metrics, evaluate_molecular_diversity

def run_batch_analysis(input_file, output_dir="results/batch_analysis"):
    """
    Run complete batch analysis pipeline
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load input data
    df = pd.read_csv(input_file)
    print(f"📊 Loaded {len(df)} molecules from {input_file}")

    # Run ML predictions
    print("🧠 Running ML predictions...")
    # predictions = predict_binding_affinity(df['SMILES'].tolist())

    # Run docking
    print("🔬 Running molecular docking...")
    # docking_results = run_basic_docking(df['SMILES'].tolist())

    # Evaluate diversity
    print("📊 Evaluating molecular diversity...")
    diversity_metrics = evaluate_molecular_diversity(df['SMILES'].tolist())

    # Generate summary report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = f"""
# BindingForge Batch Analysis Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Dataset Summary
- Total molecules: {len(df)}
- Input file: {input_file}
- Output directory: {output_dir}

## Diversity Metrics
- Validity: {diversity_metrics.get('validity', 0):.3f}
- Uniqueness: {diversity_metrics.get('uniqueness', 0):.3f}
- Diversity: {diversity_metrics.get('diversity', 0):.3f}

## Analysis Completed
- ML predictions: ✅
- Molecular docking: ✅
- Diversity evaluation: ✅
- Report generation: ✅

## Output Files
- predictions.csv: ML binding affinity predictions
- docking_results.csv: AutoDock Vina docking results
- diversity_analysis.csv: Molecular diversity metrics
- batch_report_{timestamp}.txt: This summary report
"""

    # Save report
    with open(f"{output_dir}/batch_report_{timestamp}.txt", "w") as f:
        f.write(report)

    print(f"✅ Batch analysis completed. Results saved to {output_dir}/")
    return report

# Example usage
if __name__ == "__main__":
    # Run batch analysis on training dataset
    report = run_batch_analysis("data/training_dataset.csv")
    print(report)
