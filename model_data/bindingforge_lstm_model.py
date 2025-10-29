# BindingForge Bidirectional LSTM Generator
# Safe, robust implementation based on your proven approach

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
import random
from collections import Counter
import re

# RDKit for validation
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    print("⚠️ RDKit not available - molecule validation will be limited")
    RDKIT_AVAILABLE = False

class SMILESTokenizer:
    """
    Safe SMILES tokenizer with comprehensive error handling
    Based on your successful character-level approach
    """
    
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        self.max_length = 0
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.START_TOKEN = '<START>'
        self.END_TOKEN = '<END>'
        self.UNK_TOKEN = '<UNK>'
        
        self.special_tokens = [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN]
        
        # Common SMILES characters (from your successful implementation)
        self.smiles_chars = [
            'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',  # Atoms
            '(', ')', '[', ']',                               # Branches/rings
            '=', '#', '+', '-',                              # Bonds/charges
            '1', '2', '3', '4', '5', '6', '7', '8', '9',     # Ring numbers
            'c', 'n', 'o', 's', 'p',                        # Aromatic
            '/', '\\', '@', '.',                             # Stereo/misc
            'H'                                              # Hydrogen
        ]
        
        self.logger = logging.getLogger(__name__)
    
    def build_vocabulary(self, smiles_list: List[str]) -> None:
        """
        Build character vocabulary from SMILES list
        
        Args:
            smiles_list: List of SMILES strings
        """
        self.logger.info(f"🔤 Building vocabulary from {len(smiles_list):,} SMILES...")
        
        # Count all characters
        char_counter = Counter()
        max_len = 0
        
        for smiles in smiles_list:
            if pd.isna(smiles) or not isinstance(smiles, str):
                continue
                
            # Clean SMILES
            clean_smiles = self._clean_smiles(smiles)
            char_counter.update(clean_smiles)
            max_len = max(max_len, len(clean_smiles))
        
        # Build vocabulary: special tokens + common chars + found chars
        vocabulary = self.special_tokens.copy()
        
        # Add common SMILES characters first
        for char in self.smiles_chars:
            if char not in vocabulary:
                vocabulary.append(char)
        
        # Add any additional characters found in data
        for char, count in char_counter.most_common():
            if char not in vocabulary and count >= 2:  # Only include chars seen at least twice
                vocabulary.append(char)
        
        # Create mappings
        self.char_to_idx = {char: idx for idx, char in enumerate(vocabulary)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(vocabulary)
        self.max_length = min(max_len + 2, 150)  # +2 for START/END tokens, cap at 150
        
        print(f"📊 VOCABULARY BUILT")
        print("=" * 30)
        print(f"🔤 Vocabulary size: {self.vocab_size}")
        print(f"📏 Max sequence length: {self.max_length}")
        print(f"📋 Character frequency (top 20):")
        
        for char, count in char_counter.most_common(20):
            print(f"  '{char}': {count:,}")
        
        # Validate critical characters
        critical_chars = ['C', 'N', 'O', '(', ')', '=']
        missing_critical = [char for char in critical_chars if char not in self.char_to_idx]
        
        if missing_critical:
            self.logger.warning(f"⚠️ Missing critical characters: {missing_critical}")
        else:
            print(f"✅ All critical SMILES characters present")
    
    def save_tokenizer(self, filepath: str = "model_data/tokenizer.json") -> None:
        """Save tokenizer to model_data folder"""
        # Ensure directory exists
        Path(filepath).parent.mkdir(exist_ok=True)
        
        tokenizer_data = {
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'special_tokens': self.special_tokens,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(tokenizer_data, f, indent=2)
        
        print(f"💾 Tokenizer saved: {filepath}")
    
    def load_tokenizer(self, filepath: str = "model_data/tokenizer.json") -> None:
        """Load tokenizer from model_data folder"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.char_to_idx = data['char_to_idx']
        self.idx_to_char = {int(k): v for k, v in data['idx_to_char'].items()}
        self.vocab_size = data['vocab_size']
        self.max_length = data['max_length']
        
        print(f"📁 Tokenizer loaded: {filepath}")
    
    def _clean_smiles(self, smiles: str) -> str:
        """Clean SMILES string"""
        if not isinstance(smiles, str):
            return ""
        
        # Basic cleaning
        smiles = smiles.strip()
        
        # Remove spaces and common artifacts
        smiles = re.sub(r'\s+', '', smiles)
        
        return smiles
    
    def encode(self, smiles: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode SMILES string to token indices
        
        Args:
            smiles: SMILES string
            add_special_tokens: Whether to add START/END tokens
            
        Returns:
            List of token indices
        """
        if pd.isna(smiles) or not isinstance(smiles, str):
            return [self.char_to_idx[self.UNK_TOKEN]]
        
        clean_smiles = self._clean_smiles(smiles)
        
        # Convert to indices
        indices = []
        
        if add_special_tokens:
            indices.append(self.char_to_idx[self.START_TOKEN])
        
        for char in clean_smiles:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                indices.append(self.char_to_idx[self.UNK_TOKEN])
        
        if add_special_tokens:
            indices.append(self.char_to_idx[self.END_TOKEN])
        
        return indices
    
    def decode(self, indices: List[int], remove_special_tokens: bool = True) -> str:
        """
        Decode token indices to SMILES string
        
        Args:
            indices: List of token indices
            remove_special_tokens: Whether to remove special tokens
            
        Returns:
            SMILES string
        """
        chars = []
        
        for idx in indices:
            if idx < len(self.idx_to_char):
                char = self.idx_to_char[idx]
                
                if remove_special_tokens and char in self.special_tokens:
                    if char == self.END_TOKEN:
                        break
                    continue
                
                chars.append(char)
        
        return ''.join(chars)
    
    def pad_sequence(self, indices: List[int], max_length: Optional[int] = None) -> List[int]:
        """Pad sequence to max length"""
        if max_length is None:
            max_length = self.max_length
        
        if len(indices) > max_length:
            return indices[:max_length]
        
        # Pad with PAD tokens
        pad_length = max_length - len(indices)
        return indices + [self.char_to_idx[self.PAD_TOKEN]] * pad_length
    
    def save(self, filepath: str) -> None:
        """Save tokenizer"""
        tokenizer_data = {
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'special_tokens': self.special_tokens,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(tokenizer_data, f, indent=2)
        
        print(f"💾 Tokenizer saved: {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load tokenizer"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.char_to_idx = data['char_to_idx']
        self.idx_to_char = {int(k): v for k, v in data['idx_to_char'].items()}
        self.vocab_size = data['vocab_size']
        self.max_length = data['max_length']
        
        print(f"📁 Tokenizer loaded: {filepath}")

class BindingSiteEncoder(nn.Module):
    """
    Encoder for binding site features (your 20 residues)
    """
    
    def __init__(self, input_dim: int = 20, hidden_dim: int = 128, output_dim: int = 256):
        """
        Args:
            input_dim: Number of binding site features per residue
            hidden_dim: Hidden dimension for processing
            output_dim: Output feature dimension
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Process binding site features
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
        
        # Attention for residue importance
        self.attention = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, binding_site_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            binding_site_features: [batch_size, num_residues, features_per_residue]
            
        Returns:
            Encoded binding site representation: [batch_size, output_dim]
        """
        batch_size, num_residues, feature_dim = binding_site_features.shape
        
        # Encode each residue
        # Reshape for batch processing: [batch_size * num_residues, feature_dim]
        reshaped_features = binding_site_features.view(-1, feature_dim)
        encoded_residues = self.feature_encoder(reshaped_features)
        
        # Reshape back: [batch_size, num_residues, output_dim]
        encoded_residues = encoded_residues.view(batch_size, num_residues, self.output_dim)
        
        # Compute attention weights
        attention_scores = self.attention(encoded_residues)  # [batch_size, num_residues, 1]
        attention_weights = F.softmax(attention_scores.squeeze(-1), dim=1)  # [batch_size, num_residues]
        
        # Weighted sum of residue representations
        binding_site_repr = torch.sum(
            encoded_residues * attention_weights.unsqueeze(-1), 
            dim=1
        )  # [batch_size, output_dim]
        
        return binding_site_repr

class TargetAwareLSTM(nn.Module):
    """
    Target-aware bidirectional LSTM for molecular generation
    Based on your proven architecture with enhancements
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 3,
        binding_site_dim: int = 256,
        dropout: float = 0.2
    ):
        """
        Args:
            vocab_size: Size of SMILES vocabulary
            embedding_dim: Embedding dimension for characters
            hidden_dim: Hidden dimension for LSTM layers
            num_layers: Number of LSTM layers
            binding_site_dim: Dimension of binding site features
            dropout: Dropout rate
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.binding_site_dim = binding_site_dim
        
        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bidirectional LSTM layers (your proven approach)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Binding site encoder
        self.binding_site_encoder = BindingSiteEncoder(
            input_dim=13,  # Your binding site has 13 features per residue
            output_dim=binding_site_dim
        )
        
        # Attention mechanism for target-aware generation
        self.target_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # Bidirectional doubles the size
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Context fusion
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + binding_site_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with successful parameters"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM weights
                    nn.init.orthogonal_(param)
                else:
                    # Linear weights
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        binding_site_features: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: [batch_size, seq_len] - Token indices
            binding_site_features: [batch_size, num_residues, features_per_residue]
            hidden_state: Optional hidden state for generation
            
        Returns:
            Dictionary with logits and attention weights
        """
        batch_size, seq_len = input_ids.shape
        
        # Embed characters
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # LSTM forward pass
        lstm_output, hidden_state = self.lstm(embedded, hidden_state)
        # lstm_output: [batch_size, seq_len, hidden_dim * 2]
        
        # Encode binding site
        binding_site_repr = self.binding_site_encoder(binding_site_features)
        # binding_site_repr: [batch_size, binding_site_dim]
        
        # Expand binding site representation for each position
        binding_site_expanded = binding_site_repr.unsqueeze(1).expand(-1, seq_len, -1)
        # binding_site_expanded: [batch_size, seq_len, binding_site_dim]
        
        # Apply target-aware attention
        attended_output, attention_weights = self.target_attention(
            lstm_output, lstm_output, lstm_output
        )
        # attended_output: [batch_size, seq_len, hidden_dim * 2]
        
        # Fuse LSTM output with binding site information
        combined_features = torch.cat([attended_output, binding_site_expanded], dim=-1)
        # combined_features: [batch_size, seq_len, hidden_dim * 2 + binding_site_dim]
        
        # Context fusion
        fused_output = self.context_fusion(combined_features)
        # fused_output: [batch_size, seq_len, hidden_dim * 2]
        
        # Generate logits
        logits = self.output_projection(fused_output)
        # logits: [batch_size, seq_len, vocab_size]
        
        return {
            'logits': logits,
            'hidden_state': hidden_state,
            'attention_weights': attention_weights,
            'binding_site_repr': binding_site_repr
        }
    
    def generate(
        self,
        binding_site_features: torch.Tensor,
        tokenizer: SMILESTokenizer,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 10,
        device: str = 'cpu'
    ) -> List[str]:
        """
        Generate SMILES strings conditioned on binding site
        
        Args:
            binding_site_features: Binding site features
            tokenizer: SMILES tokenizer
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            device: Computing device
            
        Returns:
            List of generated SMILES strings
        """
        self.eval()
        
        batch_size = binding_site_features.shape[0]
        generated_sequences = []
        
        with torch.no_grad():
            for i in range(batch_size):
                # Initialize with START token
                current_sequence = [tokenizer.char_to_idx[tokenizer.START_TOKEN]]
                hidden_state = None
                
                # Single binding site features
                single_binding_site = binding_site_features[i:i+1]
                
                for _ in range(max_length):
                    # Current input
                    input_ids = torch.tensor([current_sequence], device=device)
                    
                    # Forward pass
                    outputs = self.forward(input_ids, single_binding_site, hidden_state)
                    
                    # Get next token logits
                    next_token_logits = outputs['logits'][0, -1, :] / temperature
                    
                    # Top-k sampling
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                        probs = F.softmax(top_k_logits, dim=-1)
                        next_token_idx = top_k_indices[torch.multinomial(probs, 1)]
                    else:
                        probs = F.softmax(next_token_logits, dim=-1)
                        next_token_idx = torch.multinomial(probs, 1)
                    
                    next_token = next_token_idx.item()
                    
                    # Check for END token
                    if next_token == tokenizer.char_to_idx[tokenizer.END_TOKEN]:
                        break
                    
                    current_sequence.append(next_token)
                    hidden_state = outputs['hidden_state']
                
                # Decode sequence
                generated_smiles = tokenizer.decode(current_sequence)
                generated_sequences.append(generated_smiles)
        
        return generated_sequences

def create_model_config() -> Dict:
    """Create model configuration based on your successful setup"""
    return {
        'model': {
            'embedding_dim': 128,
            'hidden_dim': 512,        # Your proven setting
            'num_layers': 3,          # Your proven setting
            'binding_site_dim': 256,
            'dropout': 0.2
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'grad_clip': 1.0,
            'patience': 10,
            'temperature': 0.8        # Your proven setting
        },
        'data': {
            'train_split': 0.8,       # Your proven 80/20 split
            'val_split': 0.2,
            'max_length': 150
        }
    }

def save_model_checkpoint(
    model: TargetAwareLSTM,
    tokenizer: SMILESTokenizer,
    config: Dict,
    epoch: int,
    loss: float,
    filepath: str = None
) -> None:
    """Save comprehensive model checkpoint to model_data folder"""
    
    if filepath is None:
        # Auto-generate filepath in model_data
        checkpoint_dir = Path("model_data/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = checkpoint_dir / f"model_checkpoint_epoch_{epoch}_{timestamp}.pt"
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'config': config,
        'loss': loss,
        'vocab_size': tokenizer.vocab_size,
        'char_to_idx': tokenizer.char_to_idx,
        'idx_to_char': tokenizer.idx_to_char,
        'max_length': tokenizer.max_length,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, filepath)
    print(f"💾 Model checkpoint saved: {filepath}")

def load_model_checkpoint(filepath: str, device: str = 'cpu') -> Tuple[TargetAwareLSTM, SMILESTokenizer, Dict]:
    """Load model checkpoint from model_data folder"""
    
    checkpoint = torch.load(filepath, map_location=device)
    
    # Reconstruct tokenizer
    tokenizer = SMILESTokenizer()
    tokenizer.char_to_idx = checkpoint['char_to_idx']
    tokenizer.idx_to_char = checkpoint['idx_to_char']
    tokenizer.vocab_size = checkpoint['vocab_size']
    tokenizer.max_length = checkpoint['max_length']
    
    # Reconstruct model
    config = checkpoint['config']
    model = TargetAwareLSTM(
        vocab_size=tokenizer.vocab_size,
        **config['model']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"📁 Model checkpoint loaded: {filepath}")
    print(f"📊 Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
    
    return model, tokenizer, config

def test_model_components():
    """Test all model components with your data structure"""
    print("🧪 Testing BindingForge Model Components")
    print("=" * 50)
    
    try:
        # Test with sample data that matches your structure
        sample_smiles = [
            "Brc1cc2c(NCc3ccccn3)ncnc2s1",
            "Brc1cc2c(NCc3cccnc3)ncnc2s1", 
            "Cc1cc(C)c(C=C2C(=O)Nc3ncnc(Nc4ccc(F)c(Cl)c4)c32)[nH]1"
        ]
        
        print("🔤 Testing SMILES Tokenizer...")
        tokenizer = SMILESTokenizer()
        tokenizer.build_vocabulary(sample_smiles)
        
        # Test encoding/decoding
        encoded = tokenizer.encode(sample_smiles[0])
        decoded = tokenizer.decode(encoded)
        print(f"✅ Original: {sample_smiles[0]}")
        print(f"✅ Decoded:  {decoded}")
        
        print("\n🧬 Testing Binding Site Encoder...")
        binding_site_encoder = BindingSiteEncoder(input_dim=13, output_dim=256)
        
        # Test with sample binding site data (20 residues, 13 features)
        sample_binding_features = torch.randn(1, 20, 13)
        binding_output = binding_site_encoder(sample_binding_features)
        print(f"✅ Binding site input shape: {sample_binding_features.shape}")
        print(f"✅ Binding site output shape: {binding_output.shape}")
        
        print("\n🧠 Testing Target-Aware LSTM...")
        model = TargetAwareLSTM(vocab_size=tokenizer.vocab_size)
        
        # Test forward pass
        sample_input = torch.LongTensor([[1, 2, 3, 4, 5]])  # Sample token sequence
        outputs = model(sample_input, sample_binding_features)
        
        print(f"✅ Model input shape: {sample_input.shape}")
        print(f"✅ Model output logits shape: {outputs['logits'].shape}")
        print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        print("\n🎯 Testing Molecule Generation...")
        generated = model.generate(
            sample_binding_features, tokenizer, 
            max_length=50, temperature=0.8
        )
        print(f"✅ Generated molecule: {generated[0]}")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Model components ready for training")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 BindingForge LSTM Model Components")
    print("=" * 50)
    print("✅ SMILESTokenizer - Character-level encoding")
    print("✅ BindingSiteEncoder - Target conditioning")  
    print("✅ TargetAwareLSTM - Bidirectional generation")
    print("✅ Model checkpointing - Safe training")
    print("✅ Organized in model_data/ folder")
    print("\n🧪 Running component tests...")
    
    if test_model_components():
        print("\n🎯 Ready for training pipeline implementation!")
        print("👉 Next step: Create training_pipeline.py")
    else:
        print("\n❌ Please fix issues before proceeding")