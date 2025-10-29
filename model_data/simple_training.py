#!/usr/bin/env python3
"""
Enhanced EGFR LSTM Training Script with Binding Site Conditioning
Author: TAQDEES
Description: Complete implementation with proper data handling and 100 epochs
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SMILESDataset(Dataset):
    """Dataset for SMILES sequences with protein conditioning"""
    
    def __init__(self, smiles_list, protein_features, char_to_idx, max_length=200):
        self.smiles_list = smiles_list
        self.protein_features = protein_features
        self.char_to_idx = char_to_idx
        self.max_length = max_length
        
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        
        # Convert SMILES to indices
        indices = [self.char_to_idx.get(char, self.char_to_idx['<UNK>']) for char in smiles]
        
        # Add special tokens
        indices = [self.char_to_idx['<START>']] + indices + [self.char_to_idx['<END>']]
        
        # Truncate if too long
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        
        return {
            'smiles_indices': torch.tensor(indices, dtype=torch.long),
            'protein_features': torch.tensor(self.protein_features, dtype=torch.float32),
            'length': len(indices)
        }

def collate_fn(batch):
    """Custom collate function for variable length sequences"""
    smiles_seqs = [item['smiles_indices'] for item in batch]
    protein_features = torch.stack([item['protein_features'] for item in batch])
    lengths = [item['length'] for item in batch]
    
    # Pad sequences
    padded_seqs = pad_sequence(smiles_seqs, batch_first=True, padding_value=0)
    
    return {
        'smiles': padded_seqs,
        'protein_features': protein_features,
        'lengths': torch.tensor(lengths, dtype=torch.long)
    }

class ProteinConditionedLSTM(nn.Module):
    """LSTM model conditioned on protein binding site features with attention"""
    
    def __init__(self, vocab_size, embedding_dim=256, lstm_dim=512, 
                 protein_feature_dim=60, num_layers=3, dropout=0.3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.num_layers = num_layers
        
        # Character Embedding layer (vocab_size=64 as shown in diagram)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Protein binding site conditioning
        self.protein_processor = nn.Sequential(
            nn.Linear(protein_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embedding_dim)
        )
        
        # Bidirectional LSTM layers (3 layers, 512 units each as per diagram)
        self.lstm = nn.LSTM(
            input_size=embedding_dim * 2,  # embedding + protein features
            hidden_size=lstm_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention Mechanism (as shown in red box in diagram)
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_dim * 2,  # *2 for bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection to character probabilities
        self.output_projection = nn.Sequential(
            nn.Linear(lstm_dim * 2, lstm_dim),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_dim, vocab_size)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, smiles, protein_features, lengths=None):
        batch_size, seq_len = smiles.shape
        
        # Character embedding
        smiles_embedded = self.embedding(smiles)  # [batch, seq, embed_dim]
        
        # Process protein binding site features
        protein_embedded = self.protein_processor(protein_features)  # [batch, embed_dim]
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

def build_vocabulary(smiles_list):
    """Build character vocabulary from SMILES"""
    char_counts = {}
    for smiles in smiles_list:
        for char in smiles:
            char_counts[char] = char_counts.get(char, 0) + 1
    
    # Create vocabulary
    chars = sorted(char_counts.keys())
    
    # Add special tokens
    special_tokens = ['<PAD>', '<START>', '<END>', '<UNK>']
    vocab = special_tokens + chars
    
    char_to_idx = {char: idx for idx, char in enumerate(vocab)}
    idx_to_char = {idx: char for idx, char in enumerate(vocab)}
    
    return char_to_idx, idx_to_char

def prepare_protein_features(binding_site_df):
    """Prepare protein binding site features as a single vector"""
    # Extract relevant features for each residue
    features = []
    
    for _, row in binding_site_df.iterrows():
        residue_features = [
            row['x'], row['y'], row['z'],  # 3D coordinates
            row['hydrophobicity'],         # Hydrophobicity
            row['charge'],                 # Charge
            row['h_donor'],               # H-bond donor
            row['h_acceptor'],            # H-bond acceptor
            row['size']                   # Residue size
        ]
        features.extend(residue_features)
    
    # Should be 20 residues * 8 features = 160 features
    # But we'll flatten and pad/truncate to a fixed size
    protein_vector = np.array(features[:60])  # Take first 60 features
    
    # Pad if necessary
    if len(protein_vector) < 60:
        protein_vector = np.pad(protein_vector, (0, 60 - len(protein_vector)))
    
    return protein_vector

def calculate_statistics(results):
    """Calculate comprehensive statistics"""
    total = results['total_generated']
    valid_count = len(results['valid_molecules'])
    
    stats = {
        'validity_rate': valid_count / total if total > 0 else 0,
        'uniqueness_rate': len(set(results['valid_molecules'])) / valid_count if valid_count > 0 else 0
    }
    
    if results['properties']:
        props_df = pd.DataFrame(results['properties'])
        
        # Drug-likeness statistics
        stats['avg_qed'] = props_df['qed'].mean()
        stats['drug_like_rate'] = (props_df['qed'] >= 0.5).mean()
        stats['lipinski_compliant_rate'] = (props_df['lipinski_violations'] == 0).mean()
        stats['avg_molecular_weight'] = props_df['molecular_weight'].mean()
        stats['avg_logp'] = props_df['logp'].mean()
        stats['logp_optimal_rate'] = ((props_df['logp'] >= 0) & (props_df['logp'] <= 5)).mean()
    
    return stats

def train_model():
    """Main training function"""
    logger.info("Starting EGFR LSTM Training...")
    
    # Create output directory
    output_dir = "model_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    
    # Load molecular data with robust CSV parsing
    try:
        # Special handling for concatenated CSV files
        logger.info("Reading CSV file with potential multiple headers...")
        
        # Read all lines first
        with open('processed_data/egfr_type1_filtered.csv', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Find all header lines (lines that start with 'smiles,')
        header_lines = []
        for i, line in enumerate(lines):
            if line.strip().startswith('smiles,'):
                header_lines.append(i)
        
        logger.info(f"Found {len(header_lines)} header lines at positions: {header_lines}")
        
        # If multiple headers found, process each section separately
        if len(header_lines) > 1:
            logger.info("Multiple headers detected. Processing sections separately...")
            
            all_dataframes = []
            for i, header_pos in enumerate(header_lines):
                # Determine end position
                if i + 1 < len(header_lines):
                    end_pos = header_lines[i + 1]
                else:
                    end_pos = len(lines)
                
                # Extract section
                section_lines = lines[header_pos:end_pos]
                
                # Write temporary file
                temp_file = f'temp_section_{i}.csv'
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.writelines(section_lines)
                
                # Read section
                try:
                    section_df = pd.read_csv(temp_file, on_bad_lines='skip')
                    all_dataframes.append(section_df)
                    logger.info(f"Section {i+1}: {len(section_df)} rows, {len(section_df.columns)} columns")
                    
                    # Clean up temp file
                    os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to read section {i+1}: {e}")
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            
            # Combine all sections
            if all_dataframes:
                # Find common columns
                common_columns = set(all_dataframes[0].columns)
                for df in all_dataframes[1:]:
                    common_columns = common_columns.intersection(set(df.columns))
                
                common_columns = list(common_columns)
                logger.info(f"Common columns across sections: {len(common_columns)} columns")
                logger.info(f"Common columns: {common_columns}")
                
                # Keep only common columns and combine
                combined_dfs = []
                for df in all_dataframes:
                    combined_dfs.append(df[common_columns])
                
                compounds_df = pd.concat(combined_dfs, ignore_index=True)
                logger.info(f"Combined dataset: {len(compounds_df)} rows")
                
            else:
                raise Exception("No valid sections found")
        
        else:
            # Single header - use skip bad lines
            compounds_df = pd.read_csv('processed_data/egfr_type1_filtered.csv', 
                                     on_bad_lines='skip',
                                     sep=',',
                                     quotechar='"',
                                     encoding='utf-8')
        
        logger.info(f"Successfully loaded {len(compounds_df)} compounds")
        logger.info(f"Final columns: {list(compounds_df.columns)}")
        
    except Exception as e:
        logger.error(f"Error reading processed_data/egfr_type1_filtered.csv: {e}")
        return
    
    # Load binding site features with robust parsing
    try:
        binding_site_df = pd.read_csv('binding_site_data/binding_site_features.csv',
                                    on_bad_lines='skip',
                                    sep=',',
                                    encoding='utf-8')
        logger.info(f"Loaded {len(binding_site_df)} binding site residues")
        logger.info(f"Binding site columns: {list(binding_site_df.columns)}")
        
    except Exception as e:
        logger.error(f"Error reading binding_site_data/binding_site_features.csv: {e}")
        return
    
    # Validate essential columns exist
    required_compound_columns = ['smiles', 'pchembl_value']
    missing_columns = [col for col in required_compound_columns if col not in compounds_df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns in compounds data: {missing_columns}")
        logger.info(f"Available columns: {list(compounds_df.columns)}")
        
        # Try common alternative column names
        column_mapping = {
            'smiles': ['SMILES', 'canonical_smiles', 'smiles_string'],
            'pchembl_value': ['pchembl', 'pChEMBL', 'activity_value', 'ic50_value']
        }
        
        for required_col, alternatives in column_mapping.items():
            if required_col not in compounds_df.columns:
                for alt in alternatives:
                    if alt in compounds_df.columns:
                        compounds_df[required_col] = compounds_df[alt]
                        logger.info(f"Mapped {alt} to {required_col}")
                        break
        
        # Check again
        missing_columns = [col for col in required_compound_columns if col not in compounds_df.columns]
        if missing_columns:
            logger.error(f"Still missing required columns: {missing_columns}")
            return
    # Clean and validate data
    # Remove rows with missing SMILES or activity values
    initial_count = len(compounds_df)
    compounds_df = compounds_df.dropna(subset=['smiles', 'pchembl_value'])
    logger.info(f"Removed {initial_count - len(compounds_df)} rows with missing data")
    
    # Filter for high-activity compounds (pChEMBL > 7.0)
    high_activity = compounds_df[compounds_df['pchembl_value'] >= 7.0]
    logger.info(f"Filtered to {len(high_activity)} high-activity compounds")
    
    # If we have very few compounds, lower the threshold
    if len(high_activity) < 10:
        logger.warning("Very few high-activity compounds found. Lowering threshold to pChEMBL > 6.0")
        high_activity = compounds_df[compounds_df['pchembl_value'] >= 6.0]
        logger.info(f"Using {len(high_activity)} compounds with pChEMBL > 6.0")
    
    # If still too few, use all available compounds
    if len(high_activity) < 5:
        logger.warning("Still very few compounds. Using all available compounds.")
        high_activity = compounds_df
        logger.info(f"Using all {len(high_activity)} compounds")
    
    # Extract SMILES
    smiles_list = high_activity['smiles'].tolist()
    
    # Build vocabulary
    logger.info("Building vocabulary...")
    char_to_idx, idx_to_char = build_vocabulary(smiles_list)
    vocab_size = len(char_to_idx)
    logger.info(f"Vocabulary size: {vocab_size}")
    
    # Prepare protein features
    logger.info("Preparing protein features...")
    protein_features = prepare_protein_features(binding_site_df)
    logger.info(f"Protein feature vector size: {len(protein_features)}")
    
    # Split data - handle small datasets
    if len(smiles_list) < 5:
        logger.warning("Very small dataset. Using the same data for training and validation.")
        train_smiles = smiles_list
        val_smiles = smiles_list  # Use same data for validation in small datasets
    else:
        train_smiles, val_smiles = train_test_split(smiles_list, test_size=0.2, random_state=42)
    
    logger.info(f"Training set: {len(train_smiles)}, Validation set: {len(val_smiles)}")
    
    # Create datasets
    train_dataset = SMILESDataset(train_smiles, protein_features, char_to_idx)
    val_dataset = SMILESDataset(val_smiles, protein_features, char_to_idx)
    
    # Create data loaders - smaller batch size for small datasets
    batch_size = min(32, len(train_smiles))  # Adjust batch size to dataset size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           collate_fn=collate_fn, num_workers=0)
    
    # Initialize model
    logger.info("Initializing model...")
    
    # Force CPU usage to avoid CUDA issues
    device = torch.device('cpu')
    logger.info(f"Using device: {device} (CPU only)")
    
    model = ProteinConditionedLSTM(
        vocab_size=vocab_size,
        embedding_dim=256,
        lstm_dim=512,
        protein_feature_dim=len(protein_features),
        num_layers=3,
        dropout=0.3
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training loop
    num_epochs = 100
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in train_pbar:
            optimizer.zero_grad()
            
            smiles = batch['smiles'].to(device)
            protein_features = batch['protein_features'].to(device)
            lengths = batch['lengths']
            
            # Create targets (shift by one position)
            targets = smiles[:, 1:].contiguous()
            inputs = smiles[:, :-1].contiguous()
            
            # Forward pass
            outputs = model(inputs, protein_features)
            
            # Calculate loss
            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
            
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / train_steps
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for batch in val_pbar:
                smiles = batch['smiles'].to(device)
                protein_features = batch['protein_features'].to(device)
                
                targets = smiles[:, 1:].contiguous()
                inputs = smiles[:, :-1].contiguous()
                
                outputs = model(inputs, protein_features)
                loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
                
                val_loss += loss.item()
                val_steps += 1
                
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / val_steps
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Logging
        logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                   f"Train Loss: {avg_train_loss:.4f}, "
                   f"Val Loss: {avg_val_loss:.4f}, "
                   f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'char_to_idx': char_to_idx,
                'idx_to_char': idx_to_char,
                'protein_features': protein_features,
                'model_config': {
                    'vocab_size': vocab_size,
                    'embedding_dim': 256,
                    'lstm_dim': 512,
                    'protein_feature_dim': len(protein_features),
                    'num_layers': 3,
                    'dropout': 0.3
                }
            }
            torch.save(checkpoint, os.path.join(output_dir, 'best_egfr_model.pt'))
            logger.info(f"New best model saved with validation loss: {avg_val_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Over Time')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'vocab_size': vocab_size,
        'training_samples': len(train_smiles),
        'validation_samples': len(val_smiles)
    }
    
    with open(os.path.join(output_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Model and results saved in: {output_dir}")
    
    return model, char_to_idx, idx_to_char, protein_features

if __name__ == "__main__":
    try:
        model, char_to_idx, idx_to_char, protein_features = train_model()
        print("\n✅ Training completed successfully!")
        print("📁 Check the 'experiments' folder for:")
        print("   - best_model.pt (trained model)")
        print("   - training_curves.png (loss plots)")
        print("   - training_history.pkl (training data)")
        print("   - checkpoint files")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise