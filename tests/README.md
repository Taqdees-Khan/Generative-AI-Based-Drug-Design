# Tests Directory

This directory contains unit tests and validation scripts.

## Test Files:
- `test_docking.py` - Docking pipeline tests
- `test_ml_model.py` - Machine learning model tests
- `test_utils.py` - Utility function tests
- `test_validation.py` - End-to-end validation tests

## Running Tests:
```bash
python -m pytest tests/
```

## Validation:
- Erlotinib re-docking validation (RMSD < 2.0 Å)
- Cross-validation on ChEMBL dataset
- Performance benchmarks
