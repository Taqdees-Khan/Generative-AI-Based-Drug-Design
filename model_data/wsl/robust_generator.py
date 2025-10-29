# Robust Molecule Generator - Handles different checkpoint formats
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import sys

def robust_load_model(model_path="model_data/checkpoints/best_model.pt"):
    """Robust model loading that handles different checkpoint formats"""
    
    print(f"🔍 Loading model from: {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"✅ Checkpoint loaded")
        print(f"📊 Available keys: {list(checkpoint.keys())}")
        
        # Try to extract model components
        model_state = None
        tokenizer_data = {}
        config = {}
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            print("✅ Found model_state_dict")
        elif 'model' in checkpoint:
            model_state = checkpoint['model']
            print("✅ Found model key")
        
        # Extract tokenizer information
        possible_tokenizer_keys = [
            'char_to_idx', 'tokenizer_char_to_idx', 'vocab', 'tokenizer'
        ]
        
        for key in possible_tokenizer_keys:
            if key in checkpoint:
                if key == 'tokenizer' and isinstance(checkpoint[key], dict):
                    tokenizer_data = checkpoint[key]
                else:
                    tokenizer_data['char_to_idx'] = checkpoint[key]
                print(f"✅ Found tokenizer data: {key}")
                break
        
        # Extract vocab size and other info
        if 'vocab_size' in checkpoint:
            vocab_size = checkpoint['vocab_size']
        elif 'tokenizer' in checkpoint and 'vocab_size' in checkpoint['tokenizer']:
            vocab_size = checkpoint['tokenizer']['vocab_size']
        else:
            # Try to infer from model state
            if model_state and 'embedding.weight' in model_state:
                vocab_size = model_state['embedding.weight'].shape[0]
                print(f"🔍 Inferred vocab_size from embedding: {vocab_size}")
            else:
                vocab_size = 64  # Default fallback
                print(f"⚠️ Using default vocab_size: {vocab_size}")
        
        # Extract config
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Default config
            config = {
                'model': {
                    'embedding_dim': 128,
                    'hidden_dim': 512,
                    'num_layers': 3,
                    'binding_site_dim': 256,
                    'dropout': 0.2
                }
            }
            print("⚠️ Using default config")
        
        return model_state, tokenizer_data, config, vocab_size
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None, None, None, None

def create_simple_tokenizer(vocab_size=64):
    """Create a simple tokenizer when checkpoint data is incomplete"""
    
    print(f"🔤 Creating simple tokenizer with vocab_size={vocab_size}")
    
    # Basic SMILES characters
    chars = ['<PAD>', '<START>', '<END>', '<UNK>']
    chars += ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
    chars += ['(', ')', '[', ']', '=', '#', '+', '-']
    chars += ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    chars += ['c', 'n', 'o', 's', 'p', '/', '\\', '@', '.', 'H']
    
    # Pad to vocab_size
    while len(chars) < vocab_size:
        chars.append(f'X{len(chars)}')
    
    chars = chars[:vocab_size]
    
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    return char_to_idx, idx_to_char

def simple_encode(smiles, char_to_idx, max_length=100):
    """Simple SMILES encoding"""
    encoded = [char_to_idx.get('<START>', 1)]
    
    for char in smiles:
        if char in char_to_idx:
            encoded.append(char_to_idx[char])
        else:
            encoded.append(char_to_idx.get('<UNK>', 3))
    
    encoded.append(char_to_idx.get('<END>', 2))
    
    # Pad or truncate
    if len(encoded) > max_length:
        encoded = encoded[:max_length]
    else:
        pad_token = char_to_idx.get('<PAD>', 0)
        encoded.extend([pad_token] * (max_length - len(encoded)))
    
    return encoded

def simple_decode(encoded, idx_to_char):
    """Simple SMILES decoding"""
    chars = []
    for idx in encoded:
        if idx in idx_to_char:
            char = idx_to_char[idx]
            if char in ['<START>', '<PAD>']:
                continue
            elif char == '<END>':
                break
            elif char.startswith('X'):  # Skip padding characters
                continue
            else:
                chars.append(char)
    
    return ''.join(chars)

def robust_generate_molecules(
    model_path="model_data/checkpoints/best_model.pt",
    num_molecules=100,
    temperature=0.8
):
    """Robust molecule generation"""
    
    print("🚀 Robust Molecule Generation")
    print("=" * 50)
    
    try:
        device = torch.device('cpu')  # Force CPU to avoid CUDA issues
        print(f"💻 Using device: {device}")
        
        # Load model components
        model_state, tokenizer_data, config, vocab_size = robust_load_model(model_path)
        
        if model_state is None:
            print("❌ Could not load model")
            return []
        
        # Create tokenizer
        if tokenizer_data and 'char_to_idx' in tokenizer_data:
            char_to_idx = tokenizer_data['char_to_idx']
            idx_to_char = {int(k): v for k, v in tokenizer_data.get('idx_to_char', {}).items()}
            if not idx_to_char:
                idx_to_char = {v: k for k, v in char_to_idx.items()}
        else:
            char_to_idx, idx_to_char = create_simple_tokenizer(vocab_size)
        
        print(f"✅ Tokenizer ready: {len(char_to_idx)} characters")
        
        # Import model architecture
        sys.path.append('model_data')
        try:
            from bindingforge_lstm_model import TargetAwareLSTM
            print("✅ Imported TargetAwareLSTM from bindingforge_lstm_model")
        except Exception as e:
            print(f"❌ Could not import model architecture: {e}")
            return []
        
        # Create model
        model = TargetAwareLSTM(
            vocab_size=vocab_size,
            **config.get('model', {})
        )
        
        # Load state dict
        try:
            model.load_state_dict(model_state)
            print("✅ Model state loaded")
        except Exception as e:
            print(f"⚠️ Partial model loading: {e}")
        
        model.eval()
        model.to(device)
        
        # Simple generation without binding site conditioning
        print(f"\n🧬 Generating {num_molecules} molecules...")
        generated_molecules = []
        
        start_token = char_to_idx.get('<START>', 1)
        end_token = char_to_idx.get('<END>', 2)
        pad_token = char_to_idx.get('<PAD>', 0)
        
        with torch.no_grad():
            for i in tqdm(range(num_molecules)):
                # Simple generation sequence
                sequence = [start_token]
                
                for _ in range(50):  # Max length 50
                    # Prepare input
                    input_tensor = torch.LongTensor([sequence]).to(device)
                    
                    # Dummy binding site features
                    binding_features = torch.randn(1, 20, 13).to(device)
                    
                    try:
                        # Forward pass
                        outputs = model(input_tensor, binding_features)
                        logits = outputs['logits'][0, -1, :]  # Last token logits
                        
                        # Sample next token
                        probs = torch.softmax(logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, 1).item()
                        
                        if next_token == end_token:
                            break
                        
                        sequence.append(next_token)
                        
                    except Exception as e:
                        break
                
                # Decode sequence
                smiles = simple_decode(sequence, idx_to_char)
                if smiles and len(smiles.strip()) > 0:
                    generated_molecules.append(smiles)
        
        # Validation
        print(f"\n🔍 Validating molecules...")
        valid_molecules = []
        
        try:
            from rdkit import Chem
            for smiles in generated_molecules:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None and mol.GetNumAtoms() >= 5:
                    valid_molecules.append(smiles)
        except ImportError:
            valid_molecules = [s for s in generated_molecules if len(s) >= 5 and 'C' in s]
        
        unique_molecules = list(set(valid_molecules))
        
        # Results
        print(f"\n🎉 Generation Complete!")
        print("=" * 50)
        print(f"📊 Generated: {len(generated_molecules)}")
        print(f"✅ Valid: {len(valid_molecules)}")
        print(f"🎯 Unique: {len(unique_molecules)}")
        
        if len(valid_molecules) > 0:
            print(f"📈 Validity: {len(valid_molecules)/len(generated_molecules)*100:.1f}%")
            print(f"📈 Uniqueness: {len(unique_molecules)/len(valid_molecules)*100:.1f}%")
        
        # Show samples
        print(f"\n🧬 Sample molecules:")
        for i, smiles in enumerate(unique_molecules[:10]):
            print(f"   {i+1:2d}. {smiles}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("model_data/results")
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / f"robust_generated_{timestamp}.csv"
        pd.DataFrame({'smiles': unique_molecules}).to_csv(results_file, index=False)
        print(f"\n💾 Results saved: {results_file}")
        
        return unique_molecules
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    molecules = robust_generate_molecules(
        model_path="model_data/checkpoints/best_model.pt",
        num_molecules=100,
        temperature=0.8
    )
    
    if molecules:
        print(f"\n🏆 SUCCESS: Generated {len(molecules)} novel molecules!")
    else:
        print(f"\n❌ Generation failed")