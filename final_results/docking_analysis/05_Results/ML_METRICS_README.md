# Machine Learning Validation Metrics

This document provides detailed information about the machine learning model's performance metrics and validation results.

## Model Architecture
- Type: Three-layer bidirectional LSTM with attention
- Embedding dimension: 256
- Hidden units per layer: 512
- Attention heads: 8
- Dropout rate: 0.2

## Training Parameters
- Optimizer: Adam
- Learning rate: 6×10⁻⁴
- β₁: 0.9
- β₂: 0.999
- Batch size: 128
- Maximum epochs: 100
- Early stopping patience: 10
- L2 regularization: 1×10⁻⁵
- Gradient clipping: max-norm 1.0

## Model Performance
### Loss Metrics
- Training loss: 0.142
- Validation loss: 0.163
- Test loss: 0.171
- Validation-Training gap: 0.021

### Generation Metrics
- Chemical validity: 82.6%
- Uniqueness among valid molecules: 28.3%
- Novelty vs training set: 79.5%
- Scaffold preservation: 89.7%
- Similar scaffold retention (Tanimoto > 0.8): 97.3%

### Attention Analysis
Key residue attention weights:
- Met793 (hinge): 0.23 ± 0.04
- Thr790 (gatekeeper): 0.18 ± 0.03
- Lys745 (catalytic): 0.15 ± 0.02

### Target Conditioning Impact
- Unconditioned baseline filtered candidates: 0.8%
- Target-conditioned filtered candidates: 1.8%
- Improvement factor: 2.25×

### Property Control
- 2-property objectives: Well-separated clusters
- 3-property constraints: Moderate overlap with maintained separation
- Success rate within ±20% of target properties: 73%

## Validation Protocol
- Dataset split: 80/10/10 (train/validation/test)
- Split method: Scaffold-based
- Cross-validation: 5-fold
- Random seeds: 5 different initializations

## Multi-seed Validation Results
Average performance across 5 random seeds:
- Validity: 82.6 ± 3.1%
- Uniqueness: 28.3 ± 4.2%
- Novelty: 79.5 ± 2.8%

## Generated Molecule Properties
Lead candidates (n=5):
- MW range: 247.30 - 431.54 Da
- LogP range: 3.89 - 4.86
- TPSA range: 37.81 - 100.64 Å²
- QED scores: 0.58 - 0.64
- Lipinski violations: 0
- Binding affinities: -7.7 to -8.4 kcal/mol