# Simple Generation Script - CPU Only
# Save as: model_data/generate_molecules.py

import torch
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Force CPU only
torch.cuda.is_available = lambda: False
device = torch.device('cpu')

# Import model
sys.path.append(os.path.dirname(__file__))
from model_data.csv_inspector import (
    ImprovedTargetAwareLSTM, EnhancedSMILESTokenizer
)

def load_model(checkpoint_path="checkpoints/cpu_best_model.pt"):
    """Load trained model"""
    print(f"📁 Loading model from {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Model file not found: {checkpoint_path}")
        return None, None
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct tokenizer
    tokenizer = EnhancedSMILESTokenizer()
    tokenizer_data = checkpoint['tokenizer_data']
    tokenizer.char_to_idx = tokenizer_data['char_to_idx']
    tokenizer.idx_to_char = tokenizer_data['idx_to_char']
    tokenizer.vocab_size = tokenizer_data['vocab_size']
    tokenizer.max_length = tokenizer_data['max_length']
    
    # Reconstruct model
    config = checkpoint['config']
    model = ImprovedTargetAwareLSTM(vocab_size=tokenizer.vocab_size, **config['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    validity = checkpoint.get('validity', 0.0)
    print(f"✅ Model loaded (training validity: {validity:.1%})")
    
    return model, tokenizer

def prepare_binding_site():
    """Load and prepare binding site features"""
    print("🧬 Loading binding site...")
    
    binding_site_df = pd.read_csv('../binding_site_data/binding_site_features.csv')
    
    # Get numeric features
    numeric_cols = binding_site_df.select_dtypes(include=[np.number]).columns
    features = binding_site_df[numeric_cols].values.astype(np.float32)
    features = np.nan_to_num(features)
    
    # Normalize
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
    
    # Ensure 13 features
    if features.shape[1] < 13:
        padding = np.zeros((features.shape[0], 13 - features.shape[1]))
        features = np.concatenate([features, padding], axis=1)
    else:
        features = features[:, :13]
    
    binding_features = torch.FloatTensor(features).to(device)
    print(f"✅ Binding site shape: {binding_features.shape}")
    
    return binding_features

def validate_molecule(smiles):
    """Simple molecule validation"""
    if not isinstance(smiles, str) or len(smiles) < 5:
        return False, "Too short"
    
    # Check for chemical atoms
    if not any(atom in smiles for atom in ['C', 'N', 'O', 'S', 'P', 'c', 'n', 'o', 's']):
        return False, "No chemical atoms"
    
    # Check for reasonable character diversity
    if len(set(smiles)) < 3:
        return False, "Not diverse"
    
    # Try RDKit validation if available
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "Invalid SMILES"
        if mol.GetNumAtoms() < 5:
            return False, "Too few atoms"
        return True, "Valid"
    except ImportError:
        # Simple fallback validation
        invalid_patterns = ['rr', ')))', '((', '===']
        if any(pattern in smiles for pattern in invalid_patterns):
            return False, "Invalid pattern"
        return True, "Valid (no RDKit)"

def generate_molecules(num_molecules=20):
    """Generate novel molecules"""
    print("🧬 Generating Novel EGFR Inhibitors")
    print("=" * 50)
    
    # Load model
    model, tokenizer = load_model()
    if model is None:
        print("❌ Cannot generate without trained model")
        return []
    
    # Load binding site
    binding_features = prepare_binding_site()
    
    # Generate molecules
    print(f"\n🎯 Generating {num_molecules} molecules...")
    
    all_molecules = []
    valid_molecules = []
    
    # Generate in batches of 5
    for i in range(0, num_molecules, 5):
        batch_size = min(5, num_molecules - i)
        
        try:
            generated = model.generate_with_constraints(
                binding_features[:batch_size], 
                tokenizer,
                max_length=60,
                temperature=1.3,  # Higher for diversity
                top_k=20,
                device=device,
                min_length=8
            )
            
            all_molecules.extend(generated)
            
        except Exception as e:
            print(f"⚠️ Generation error in batch {i//5 + 1}: {e}")
            all_molecules.extend([f"ERROR_BATCH_{i//5 + 1}"] * batch_size)
    
    # Validate molecules
    print(f"\n📊 Validating {len(all_molecules)} molecules...")
    
    for i, smiles in enumerate(all_molecules[:num_molecules]):
        is_valid, reason = validate_molecule(smiles)
        
        status = "✅" if is_valid else "❌"
        print(f"{i+1:2d}. {status} {smiles[:50]:<50} ({reason})")
        
        if is_valid:
            valid_molecules.append(smiles)
    
    # Summary
    validity_rate = len(valid_molecules) / len(all_molecules) if all_molecules else 0
    unique_valid = list(set(valid_molecules))
    uniqueness_rate = len(unique_valid) / len(valid_molecules) if valid_molecules else 0
    
    print(f"\n📈 Generation Results:")
    print(f"   🧬 Total generated: {len(all_molecules)}")
    print(f"   ✅ Valid molecules: {len(valid_molecules)} ({validity_rate:.1%})")
    print(f"   🎯 Unique valid: {len(unique_valid)} ({uniqueness_rate:.1%})")
    print(f"   📏 Avg length: {np.mean([len(m) for m in valid_molecules if isinstance(m, str)]):.1f}")
    
    if validity_rate > 0.3:
        print(f"🎉 SUCCESS! Much better than previous 0% validity!")
    elif validity_rate > 0.1:
        print(f"✅ Progress! Getting some valid molecules")
    else:
        print(f"⚠️ Still needs improvement")
    
    # Save results
    if valid_molecules:
        results_df = pd.DataFrame({
            'smiles': unique_valid,
            'length': [len(s) for s in unique_valid],
            'generated_by': 'BindingForge_Enhanced'
        })
        
        results_path = "results/generated_molecules.csv"
        Path("results").mkdir(exist_ok=True)
        results_df.to_csv(results_path, index=False)
        print(f"💾 Valid molecules saved: {results_path}")
    
    return valid_molecules

def quick_test():
    """Quick test of generation"""
    print("🧪 Quick Generation Test")
    print("=" * 30)
    
    molecules = generate_molecules(10)
    
    if len(molecules) >= 3:
        print(f"\n💎 Best 3 molecules:")
        for i, mol in enumerate(molecules[:3]):
            print(f"   {i+1}. {mol}")
        print(f"\n🎯 Success! Generated {len(molecules)} valid molecules")
    else:
        print(f"\n⚠️ Only {len(molecules)} valid molecules generated")
    
    return molecules

if __name__ == "__main__":
    molecules = quick_test()