#!/usr/bin/env python3
"""
Constrained EGFR Generator - High Success Rate
Author: TAQDEES
Description: Generate EGFR molecules with strict constraints for validity
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from datetime import datetime
from tqdm import tqdm
import random
import re

# Try to import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED
    RDKIT_AVAILABLE = True
    print("✅ RDKit available for validation")
except ImportError:
    print("⚠️  RDKit not available. Using basic validation.")
    RDKIT_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProteinConditionedLSTM(nn.Module):
    """LSTM model - same as working version"""
    
    def __init__(self, vocab_size, embedding_dim=256, lstm_dim=512, 
                 protein_feature_dim=60, num_layers=3, dropout=0.3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.protein_processor = nn.Sequential(
            nn.Linear(protein_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embedding_dim)
        )
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim * 2,
            hidden_size=lstm_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_dim * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.output_projection = nn.Sequential(
            nn.Linear(lstm_dim * 2, lstm_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_dim, vocab_size)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, smiles, protein_features, lengths=None):
        batch_size, seq_len = smiles.shape
        
        smiles_embedded = self.embedding(smiles)
        protein_embedded = self.protein_processor(protein_features)
        protein_embedded = protein_embedded.unsqueeze(1).expand(-1, seq_len, -1)
        
        combined_input = torch.cat([smiles_embedded, protein_embedded], dim=-1)
        lstm_out, _ = self.lstm(combined_input)
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        output = self.output_projection(attended_out)
        
        return output

def load_model_corrected():
    """Load model with corrected dimensions - same as working version"""
    
    model_path = "model_data/best_egfr_model.pt"
    checkpoint = torch.load(model_path, map_location='cpu')
    
    char_to_idx = checkpoint['char_to_idx']
    idx_to_char = checkpoint['idx_to_char']
    
    # Fix protein features
    protein_features_raw = checkpoint['protein_features']
    protein_features = protein_features_raw[0].numpy()[:60]
    
    # Force correct config
    config = {
        'vocab_size': len(char_to_idx),
        'embedding_dim': 256,
        'lstm_dim': 512,
        'protein_feature_dim': 60,
        'num_layers': 3,
        'dropout': 0.3
    }
    
    model = ProteinConditionedLSTM(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, char_to_idx, idx_to_char, protein_features

class SMILESConstraintChecker:
    """Class to check SMILES validity during generation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset for new molecule"""
        self.paren_count = 0
        self.bracket_count = 0
        self.ring_numbers = set()
        self.last_char = ''
        self.atom_count = 0
        self.bond_count = 0
    
    def is_valid_next_char(self, current_smiles, next_char):
        """Check if next character would create valid SMILES"""
        
        # Skip special tokens
        if next_char in ['<PAD>', '<START>', '<UNK>']:
            return False
        
        # End token - allow if we have reasonable length
        if next_char == '<END>':
            return len(current_smiles) >= 6 and self.paren_count == 0 and self.bracket_count == 0
        
        # Parentheses rules
        if next_char == '(':
            if self.paren_count >= 3:  # Max 3 nested levels
                return False
            self.paren_count += 1
            return True
        
        if next_char == ')':
            if self.paren_count <= 0:  # Can't close unopened
                return False
            self.paren_count -= 1
            return True
        
        # Bracket rules
        if next_char == '[':
            if self.bracket_count >= 1:  # Max 1 bracket at a time
                return False
            self.bracket_count += 1
            return True
        
        if next_char == ']':
            if self.bracket_count <= 0:
                return False
            self.bracket_count -= 1
            return True
        
        # Ring numbers
        if next_char.isdigit():
            digit = int(next_char)
            if digit > 9:  # Only single digits
                return False
            if digit in self.ring_numbers:
                self.ring_numbers.remove(digit)  # Close ring
            else:
                if len(self.ring_numbers) >= 3:  # Max 3 open rings
                    return False
                self.ring_numbers.add(digit)  # Open ring
            return True
        
        # Atoms (C, N, O, etc.)
        if next_char in ['C', 'c', 'N', 'n', 'O', 'o', 'S', 's', 'F', 'H', 'B', 'I']:
            self.atom_count += 1
            self.last_char = next_char
            return True
        
        # Bonds (=, #, -, etc.)
        if next_char in ['=', '#', '-', '\\', '/']:
            # Can't have bond after bond
            if self.last_char in ['=', '#', '-', '\\', '/']:
                return False
            self.last_char = next_char
            return True
        
        # Other characters
        if next_char in ['+', '@', '.']:
            self.last_char = next_char
            return True
        
        # Default allow
        self.last_char = next_char
        return True

def generate_constrained_molecule(model, char_to_idx, idx_to_char, protein_features,
                                max_length=60, temperature=0.7, device='cpu'):
    """Generate molecule with strict constraints"""
    
    protein_tensor = torch.tensor(protein_features, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Start with drug-like patterns
    good_starts = [
        'COc1cc',      # Methoxy benzene start
        'Cc1ccc',      # Methyl benzene start  
        'Nc1ccc',      # Amino benzene start
        'c1ccc2',      # Fused ring start
        'CNc1',        # Methylamino start
    ]
    
    start_pattern = random.choice(good_starts)
    
    # Convert to tokens
    generated = [char_to_idx.get('<START>', 1)]
    constraint_checker = SMILESConstraintChecker()
    
    for char in start_pattern:
        if char in char_to_idx:
            generated.append(char_to_idx[char])
            constraint_checker.is_valid_next_char('', char)  # Update state
    
    with torch.no_grad():
        for step in range(max_length - len(generated)):
            current_seq = torch.tensor([generated], dtype=torch.long).to(device)
            
            output = model(current_seq, protein_tensor)
            logits = output[0, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            
            # Get current SMILES for validation
            current_smiles = ''.join([idx_to_char.get(idx, '') for idx in generated[1:]])
            
            # Filter valid characters
            valid_chars = []
            valid_probs = []
            
            for idx, prob in enumerate(probs):
                char = idx_to_char.get(idx, '')
                
                # Create temporary checker to test validity
                temp_checker = SMILESConstraintChecker()
                temp_checker.paren_count = constraint_checker.paren_count
                temp_checker.bracket_count = constraint_checker.bracket_count
                temp_checker.ring_numbers = constraint_checker.ring_numbers.copy()
                temp_checker.last_char = constraint_checker.last_char
                temp_checker.atom_count = constraint_checker.atom_count
                
                if temp_checker.is_valid_next_char(current_smiles, char):
                    valid_chars.append(idx)
                    valid_probs.append(prob.item())
            
            # Sample from valid characters
            if valid_chars:
                valid_probs = np.array(valid_probs)
                
                # Boost probability of ending if molecule is long enough
                if len(generated) > 15:
                    end_idx = char_to_idx.get('<END>')
                    if end_idx in valid_chars:
                        end_pos = valid_chars.index(end_idx)
                        valid_probs[end_pos] *= 3.0  # Boost END probability
                
                # Normalize and sample
                valid_probs = valid_probs / valid_probs.sum()
                next_idx = np.random.choice(valid_chars, p=valid_probs)
                next_char = idx_to_char[next_idx]
                
                # Stop if END
                if next_char == '<END>':
                    break
                
                # Update constraint checker
                constraint_checker.is_valid_next_char(current_smiles, next_char)
                generated.append(next_idx)
                
            else:
                # No valid characters - end generation
                break
    
    # Convert to SMILES
    smiles = ''.join([idx_to_char.get(idx, '') for idx in generated[1:]])
    
    return smiles

def is_valid_smiles(smiles):
    """Enhanced SMILES validation"""
    if not smiles or len(smiles) < 5:
        return False
    
    # Basic pattern checks
    if smiles.count('(') != smiles.count(')'):
        return False
    if smiles.count('[') != smiles.count(']'):
        return False
    
    # RDKit validation
    if RDKIT_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None and mol.GetNumAtoms() >= 5
        except:
            return False
    
    return True

def generate_high_validity_batch(model, char_to_idx, idx_to_char, protein_features,
                               target_count=50, device='cpu'):
    """Generate batch with high validity rate"""
    
    valid_molecules = []
    attempts = 0
    max_attempts = target_count * 2  # Much lower attempt limit
    
    logger.info(f"Generating {target_count} molecules with constrained decoding...")
    
    with tqdm(total=target_count, desc="Constrained generation") as pbar:
        while len(valid_molecules) < target_count and attempts < max_attempts:
            try:
                # Use lower temperature for better validity
                temp = random.choice([0.5, 0.6, 0.7])
                
                smiles = generate_constrained_molecule(
                    model, char_to_idx, idx_to_char, protein_features,
                    max_length=50, temperature=temp, device=device
                )
                
                # Validate
                if (is_valid_smiles(smiles) and 
                    8 <= len(smiles) <= 80 and
                    smiles not in valid_molecules):
                    
                    valid_molecules.append(smiles)
                    pbar.update(1)
                    pbar.set_postfix({
                        'Valid': len(valid_molecules),
                        'Rate': f"{len(valid_molecules)/max(attempts,1)*100:.1f}%"
                    })
                
            except Exception as e:
                pass
            
            attempts += 1
    
    success_rate = len(valid_molecules) / attempts * 100 if attempts > 0 else 0
    logger.info(f"Generated {len(valid_molecules)} molecules in {attempts} attempts")
    logger.info(f"Success rate: {success_rate:.1f}%")
    
    return valid_molecules

def calculate_properties(molecules):
    """Calculate molecular properties"""
    if not RDKIT_AVAILABLE:
        return []
    
    properties = []
    for smiles in molecules:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                props = {
                    'smiles': smiles,
                    'molecular_weight': Descriptors.MolWt(mol),
                    'logp': Descriptors.MolLogP(mol),
                    'hbd': Descriptors.NumHDonors(mol),
                    'hba': Descriptors.NumHAcceptors(mol),
                    'qed': QED.qed(mol),
                    'num_atoms': mol.GetNumAtoms(),
                    'tpsa': Descriptors.TPSA(mol),
                    'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                    'aromatic_rings': Descriptors.NumAromaticRings(mol)
                }
                
                # Lipinski violations
                props['lipinski_violations'] = sum([
                    props['molecular_weight'] > 500,
                    props['logp'] > 5,
                    props['hbd'] > 5,
                    props['hba'] > 10
                ])
                
                properties.append(props)
        except:
            continue
    
    return properties

def main():
    """Main constrained generation"""
    
    logger.info("🎯 Constrained EGFR Generator")
    logger.info("=" * 50)
    
    device = torch.device('cpu')
    
    try:
        # Load model
        model, char_to_idx, idx_to_char, protein_features = load_model_corrected()
        model = model.to(device)
        
        # Generate with constraints
        target_molecules = 50
        logger.info(f"🎯 Target: {target_molecules} valid molecules with constraints")
        
        valid_molecules = generate_high_validity_batch(
            model, char_to_idx, idx_to_char, protein_features,
            target_count=target_molecules, device=device
        )
        
        if valid_molecules:
            # Calculate properties
            logger.info("📊 Calculating molecular properties...")
            properties = calculate_properties(valid_molecules)
            
            # Save results
            output_dir = "constrained_results"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save molecules
            molecules_df = pd.DataFrame({'smiles': valid_molecules})
            molecules_df.to_csv(f"{output_dir}/constrained_egfr_molecules.csv", index=False)
            
            if properties:
                props_df = pd.DataFrame(properties)
                props_df.to_csv(f"{output_dir}/molecular_properties.csv", index=False)
                
                # Statistics
                avg_qed = np.mean([p['qed'] for p in properties])
                drug_like = sum(1 for p in properties if p['qed'] >= 0.5)
                lipinski_ok = sum(1 for p in properties if p['lipinski_violations'] == 0)
                
                logger.info(f"\n🎉 CONSTRAINED GENERATION RESULTS:")
                logger.info(f"✅ Valid molecules: {len(valid_molecules)}")
                logger.info(f"⭐ Average QED: {avg_qed:.3f}")
                logger.info(f"💊 Drug-like (QED ≥ 0.5): {drug_like}/{len(properties)} ({drug_like/len(properties)*100:.1f}%)")
                logger.info(f"⚖️  Lipinski compliant: {lipinski_ok}/{len(properties)} ({lipinski_ok/len(properties)*100:.1f}%)")
                
                # Show best molecules
                sorted_props = sorted(properties, key=lambda x: x['qed'], reverse=True)
                logger.info(f"\n🏆 Top 15 EGFR candidates by QED:")
                for i, prop in enumerate(sorted_props[:15], 1):
                    logger.info(f"{i:2d}. {prop['smiles']}")
                    logger.info(f"    QED: {prop['qed']:.3f} | MW: {prop['molecular_weight']:.0f} | LogP: {prop['logp']:.2f} | Atoms: {prop['num_atoms']}")
            
            logger.info(f"\n💾 Results saved to: {output_dir}/")
            
        else:
            logger.warning("❌ No valid molecules generated")
            
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()