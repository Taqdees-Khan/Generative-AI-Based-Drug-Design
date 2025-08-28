"""
Machine Learning Prediction Example
"""

import torch
import pandas as pd
from src.model_architecture import BindingForge
from src.utils import smiles_to_tokens, prepare_target_features

def load_trained_model(model_path="models/bindingforge_egfr.pth"):
    """Load pre-trained BindingForge model"""

    model = BindingForge(
        vocab_size=64,
        embedding_dim=256,
        hidden_dim=512,
        num_layers=3,
        target_features=100
    )

    # Load trained weights (if available)
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"✅ Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"⚠️ Model file not found: {model_path}")
        print("Using randomly initialized model for demonstration")

    return model

def predict_binding_affinity(smiles_list, target_pdb="data/egfr_receptor.pdb"):
    """Predict binding affinities for a list of SMILES"""

    # Load model
    model = load_trained_model()

    # Prepare target features
    target_features = prepare_target_features(target_pdb)
    target_tensor = torch.FloatTensor(target_features).unsqueeze(0)

    # Load vocabulary (simplified for demo)
    vocab = {char: i for i, char in enumerate('CNOPSFClBr()[]=#-+123456789@')}
    vocab['<PAD>'] = len(vocab)
    vocab['<UNK>'] = len(vocab)

    predictions = []

    with torch.no_grad():
        for smiles in smiles_list:
            # Convert SMILES to tokens
            tokens = smiles_to_tokens(smiles, vocab)
            tokens_tensor = torch.LongTensor(tokens).unsqueeze(0)

            # Predict
            _, affinity_pred, _ = model(tokens_tensor, target_tensor)
            predicted_affinity = affinity_pred.item()

            predictions.append({
                'SMILES': smiles,
                'Predicted_pIC50': predicted_affinity,
                'Predicted_IC50_nM': 10**(-predicted_affinity) * 1e9
            })

    return pd.DataFrame(predictions)

# Example usage
if __name__ == "__main__":
    test_smiles = [
        'CCOc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OCCCN1CCOCC1',
        'COc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OC',
        'C=CC(=O)Nc1cc(Nc2nccc(n2)c2cn(C)c3ccccc23)c(OC)cc1N(C)CCN(C)C'
    ]

    predictions = predict_binding_affinity(test_smiles)
    print(predictions)
