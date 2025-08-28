"""
Basic tests for BindingForge
"""

import unittest
import torch
import pandas as pd
from src.model_architecture import BindingForge
from src.utils import calculate_molecular_properties, validate_drug_likeness
from src.evaluation import calculate_regression_metrics

class TestBindingForge(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.model = BindingForge(
            vocab_size=64,
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2
        )

    def test_model_initialization(self):
        """Test model initializes correctly"""
        self.assertIsInstance(self.model, BindingForge)
        self.assertEqual(self.model.vocab_size, 64)
        self.assertEqual(self.model.hidden_dim, 256)

    def test_model_forward(self):
        """Test model forward pass"""
        batch_size = 2
        seq_len = 50
        target_features = 100

        molecules = torch.randint(0, 64, (batch_size, seq_len))
        target_feat = torch.randn(batch_size, target_features)

        output_logits, affinity_pred, hidden = self.model(molecules, target_feat)

        self.assertEqual(output_logits.shape, (batch_size, seq_len, 64))
        self.assertEqual(affinity_pred.shape, (batch_size, 1))

    def test_molecular_properties(self):
        """Test molecular property calculation"""
        smiles = "CCOc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OCCCN1CCOCC1"  # Erlotinib
        props = calculate_molecular_properties(smiles)

        self.assertIn('MW', props)
        self.assertIn('LogP', props)
        self.assertGreater(props['MW'], 0)

    def test_drug_likeness(self):
        """Test drug-likeness validation"""
        erlotinib = "CCOc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OCCCN1CCOCC1"
        self.assertTrue(validate_drug_likeness(erlotinib))

        # Test large molecule (should fail)
        large_mol = "C" * 100
        self.assertFalse(validate_drug_likeness(large_mol))

    def test_regression_metrics(self):
        """Test regression metric calculations"""
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 2.1, 2.9, 4.1, 4.9]

        metrics = calculate_regression_metrics(y_true, y_pred)

        self.assertIn('RMSE', metrics)
        self.assertIn('R2', metrics)
        self.assertGreater(metrics['R2'], 0.9)  # Should be high correlation

if __name__ == '__main__':
    unittest.main()
