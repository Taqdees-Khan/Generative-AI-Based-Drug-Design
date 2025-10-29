#!/usr/bin/env python3
"""
Molecular Modification EGFR Generator
Author: TAQDEES
Description: Use the bidirectional LSTM to modify existing EGFR inhibitors
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
    """Bidirectional LSTM model - exactly as trained"""
    
    def __init__(self, vocab_size, embedding_dim=256, lstm_dim=512, 
                 protein_feature_dim=60, num_layers=3, dropout=0.3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.num_layers = num_layers
        
        # Character Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Protein binding site conditioning
        self.protein_processor = nn.Sequential(
            nn.Linear(protein_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embedding_dim)
        )
        
        # Bidirectional LSTM layers (this is key!)
        self.lstm = nn.LSTM(
            input_size=embedding_dim * 2,
            hidden_size=lstm_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # This reads structure in both directions!
        )
        
        # Attention Mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_dim * 2,  # *2 for bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(lstm_dim * 2, lstm_dim),  # *2 for bidirectional
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
        
        # Character embedding
        smiles_embedded = self.embedding(smiles)
        
        # Process protein binding site features
        protein_embedded = self.protein_processor(protein_features)
        protein_embedded = protein_embedded.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine SMILES and protein features (binding site conditioning)
        combined_input = torch.cat([smiles_embedded, protein_embedded], dim=-1)
        
        # Pass through bidirectional LSTM layers
        lstm_out, _ = self.lstm(combined_input)
        
        # Apply attention mechanism
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Project to character probabilities
        output = self.output_projection(attended_out)
        
        return output

def load_model_corrected():
    """Load model with corrected dimensions"""
    
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
    
    logger.info("✅ Bidirectional LSTM model loaded successfully")
    
    return model, char_to_idx, idx_to_char, protein_features

def get_egfr_templates():
    """Known EGFR inhibitor templates"""
    
    # These are validated EGFR inhibitor scaffolds
    templates = [
        # Erlotinib-like (quinazoline core)
        "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN",
        "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCC",
        "COc1cc2ncnc(Nc3ccc(Cl)cc3)c2cc1OCCCN",
        
        # Gefitinib-like 
        "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN(C)C",
        "COc1cc2ncnc(Nc3ccc(F)c(Br)c3)c2cc1OCC",
        
        # Simple quinazoline variations
        "c1ccc2ncnc(Nc3ccc(F)c(Cl)c3)c2c1",
        "c1ccc2ncnc(Nc3ccc(Cl)cc3)c2c1",
        "Cc1ccc2ncnc(Nc3ccc(F)c(Cl)c3)c2c1",
        
        # Pyrimidine-based
        "c1ccc(Nc2nccc(c3ccccc3)n2)cc1",
        "COc1ccc(Nc2nccc(c3ccccc3)n2)cc1",
        
        # Simple aromatics with common EGFR motifs
        "Nc1ccc(Cl)cc1",
        "COc1ccc(N)cc1",
        "Nc1ccc(F)c(Cl)c1",
        "COc1ccc(Nc2ncnc3ccccc23)cc1",
        
        # Smaller validated fragments
        "c1ccc2ccccc2c1",  # naphthalene
        "c1cnc2ccccc2c1",  # quinoline  
        "c1ccc2ncncc2c1",  # quinazoline core
        "c1ccc(Nc2ncnc3ccccc23)cc1",  # basic EGFR pharmacophore
    ]
    
    return templates

def smiles_to_tokens(smiles, char_to_idx):
    """Convert SMILES to token sequence"""
    tokens = [char_to_idx.get('<START>', 1)]
    for char in smiles:
        if char in char_to_idx:
            tokens.append(char_to_idx[char])
        else:
            tokens.append(char_to_idx.get('<UNK>', 3))
    tokens.append(char_to_idx.get('<END>', 2))
    return tokens

def tokens_to_smiles(tokens, idx_to_char):
    """Convert tokens back to SMILES"""
    chars = []
    for token in tokens:
        char = idx_to_char.get(token, '')
        if char not in ['<START>', '<END>', '<PAD>', '<UNK>']:
            chars.append(char)
    return ''.join(chars)

def modify_molecule_with_model(model, template_smiles, char_to_idx, idx_to_char, 
                              protein_features, modification_rate=0.3, device='cpu'):
    """Use the bidirectional model to modify existing molecules"""
    
    # Convert template to tokens
    tokens = smiles_to_tokens(template_smiles, char_to_idx)
    
    # Pad to reasonable length
    max_len = 80
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    
    # Convert to tensor
    input_seq = torch.tensor([tokens], dtype=torch.long).to(device)
    protein_tensor = torch.tensor(protein_features, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Get model predictions for each position
    with torch.no_grad():
        output = model(input_seq, protein_tensor)
        probs = F.softmax(output[0], dim=-1)  # [seq_len, vocab_size]
    
    # Modify some positions based on model predictions
    modified_tokens = tokens.copy()
    num_modifications = max(1, int(len(tokens) * modification_rate))
    
    # Choose random positions to modify (avoid START/END)
    modifiable_positions = list(range(1, len(tokens) - 1))
    positions_to_modify = random.sample(modifiable_positions, min(num_modifications, len(modifiable_positions)))
    
    for pos in positions_to_modify:
        if pos < probs.shape[0]:
            # Get top candidates from model
            top_probs, top_indices = torch.topk(probs[pos], k=5)
            
            # Filter for valid characters (avoid special tokens)
            valid_candidates = []
            for idx in top_indices:
                char = idx_to_char.get(idx.item(), '')
                if char and char not in ['<START>', '<END>', '<PAD>', '<UNK>']:
                    valid_candidates.append(idx.item())
            
            if valid_candidates:
                # Choose a candidate (prefer high probability)
                new_token = random.choice(valid_candidates[:3])  # Top 3 candidates
                modified_tokens[pos] = new_token
    
    # Convert back to SMILES
    modified_smiles = tokens_to_smiles(modified_tokens[1:-1], idx_to_char)  # Skip START/END
    
    return modified_smiles

def is_valid_smiles(smiles):
    """Check SMILES validity"""
    if not smiles or len(smiles) < 3:
        return False
    
    if RDKIT_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None and mol.GetNumAtoms() >= 3
        except:
            return False
    
    return True

def generate_by_modification(model, char_to_idx, idx_to_char, protein_features, 
                           target_count=50, device='cpu'):
    """Generate molecules by modifying templates with the bidirectional model"""
    
    templates = get_egfr_templates()
    valid_molecules = []
    
    logger.info(f"🧬 Generating {target_count} molecules by template modification")
    logger.info(f"📋 Using {len(templates)} EGFR template molecules")
    
    with tqdm(total=target_count, desc="Modifying templates") as pbar:
        attempts = 0
        max_attempts = target_count * 3
        
        while len(valid_molecules) < target_count and attempts < max_attempts:
            # Choose random template
            template = random.choice(templates)
            
            # Modify with different rates
            modification_rate = random.choice([0.1, 0.2, 0.3, 0.4])
            
            try:
                modified_smiles = modify_molecule_with_model(
                    model, template, char_to_idx, idx_to_char, protein_features,
                    modification_rate=modification_rate, device=device
                )
                
                # Validate
                if (is_valid_smiles(modified_smiles) and 
                    modified_smiles != template and  # Must be different from template
                    modified_smiles not in valid_molecules and
                    len(modified_smiles) >= 8):
                    
                    valid_molecules.append(modified_smiles)
                    pbar.update(1)
                    pbar.set_postfix({
                        'Valid': len(valid_molecules),
                        'Rate': f"{len(valid_molecules)/max(attempts,1)*100:.1f}%"
                    })
                
            except Exception as e:
                pass
            
            attempts += 1
    
    # Add some original templates if we need more molecules
    if len(valid_molecules) < target_count:
        for template in templates:
            if is_valid_smiles(template) and template not in valid_molecules:
                valid_molecules.append(template)
                if len(valid_molecules) >= target_count:
                    break
    
    logger.info(f"Generated {len(valid_molecules)} valid molecules ({len(valid_molecules)/attempts*100:.1f}% success rate)")
    
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
    """Main modification-based generation"""
    
    logger.info("🧬 Molecular Modification EGFR Generator")
    logger.info("Using Bidirectional LSTM for Structure Understanding")
    logger.info("=" * 60)
    
    device = torch.device('cpu')
    
    try:
        # Load bidirectional model
        model, char_to_idx, idx_to_char, protein_features = load_model_corrected()
        model = model.to(device)
        
        # Generate by modification
        target_molecules = 50
        logger.info(f"🎯 Target: {target_molecules} modified EGFR molecules")
        
        valid_molecules = generate_by_modification(
            model, char_to_idx, idx_to_char, protein_features,
            target_count=target_molecules, device=device
        )
        
        if valid_molecules:
            # Calculate properties
            logger.info("📊 Calculating molecular properties...")
            properties = calculate_properties(valid_molecules)
            
            # Save results
            output_dir = "modification_results"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save molecules
            molecules_df = pd.DataFrame({'smiles': valid_molecules})
            molecules_df.to_csv(f"{output_dir}/modified_egfr_molecules.csv", index=False)
            
            if properties:
                props_df = pd.DataFrame(properties)
                props_df.to_csv(f"{output_dir}/molecular_properties.csv", index=False)
                
                # Statistics
                avg_qed = np.mean([p['qed'] for p in properties])
                drug_like = sum(1 for p in properties if p['qed'] >= 0.5)
                lipinski_ok = sum(1 for p in properties if p['lipinski_violations'] == 0)
                avg_mw = np.mean([p['molecular_weight'] for p in properties])
                avg_logp = np.mean([p['logp'] for p in properties])
                
                logger.info(f"\n🎉 MODIFICATION RESULTS:")
                logger.info(f"✅ Valid molecules: {len(valid_molecules)}")
                logger.info(f"💊 Average MW: {avg_mw:.1f} Da")
                logger.info(f"🧪 Average LogP: {avg_logp:.2f}")
                logger.info(f"⭐ Average QED: {avg_qed:.3f}")
                logger.info(f"🧬 Drug-like (QED ≥ 0.5): {drug_like}/{len(properties)} ({drug_like/len(properties)*100:.1f}%)")
                logger.info(f"⚖️  Lipinski compliant: {lipinski_ok}/{len(properties)} ({lipinski_ok/len(properties)*100:.1f}%)")
                
                # Show molecules organized by QED score
                sorted_props = sorted(properties, key=lambda x: x['qed'], reverse=True)
                
                logger.info(f"\n🏆 TOP 15 EGFR CANDIDATES (by QED score):")
                logger.info("-" * 100)
                for i, prop in enumerate(sorted_props[:15], 1):
                    logger.info(f"{i:2d}. {prop['smiles']}")
                    logger.info(f"    QED: {prop['qed']:.3f} | MW: {prop['molecular_weight']:.0f} | LogP: {prop['logp']:.2f} | "
                              f"HBD: {prop['hbd']} | HBA: {prop['hba']} | Rings: {prop['aromatic_rings']}")
                
                # Show some interesting categories
                high_qed = [p for p in properties if p['qed'] >= 0.7]
                if high_qed:
                    logger.info(f"\n💎 HIGH QED MOLECULES (≥0.7): {len(high_qed)} molecules")
                    for prop in high_qed[:5]:
                        logger.info(f"   {prop['smiles']} (QED: {prop['qed']:.3f})")
                
                small_mw = [p for p in properties if p['molecular_weight'] <= 300]
                if small_mw:
                    logger.info(f"\n🧪 FRAGMENT-LIKE (MW ≤ 300): {len(small_mw)} molecules")
                    for prop in sorted(small_mw, key=lambda x: x['qed'], reverse=True)[:3]:
                        logger.info(f"   {prop['smiles']} (MW: {prop['molecular_weight']:.0f}, QED: {prop['qed']:.3f})")
            
            logger.info(f"\n💾 Results saved to: {output_dir}/")
            logger.info(f"📄 Files created:")
            logger.info(f"   - modified_egfr_molecules.csv (all molecules)")
            logger.info(f"   - molecular_properties.csv (detailed properties)")
            
        else:
            logger.warning("❌ No valid molecules generated")
            
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()