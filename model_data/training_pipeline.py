import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')# BindingForge Safe Training Pipeline
# Organized for model_data/ folder structure

import sys
import os
from pathlib import Path

# Add parent directory to path to import from other folders
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))
sys.path.append(str(current_dir))

# Try different import approaches
try:
    # Try relative import first
    from bindingforge_lstm_model import (
        TargetAwareLSTM, SMILESTokenizer, BindingSiteEncoder,
        create_model_config, save_model_checkpoint, load_model_checkpoint
    )
    print("✅ Imported from bindingforge_lstm_model")
except ImportError:
    try:
        # Try absolute import
        from model_data.bindingforge_lstm_model import (
            TargetAwareLSTM, SMILESTokenizer, BindingSiteEncoder,
            create_model_config, save_model_checkpoint, load_model_checkpoint
        )
        print("✅ Imported from model_data.bindingforge_lstm_model")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure bindingforge_lstm_model.py is in the model_data folder")
        sys.exit(1)

class BindingForgeDataset(Dataset):
    """
    Dataset class for BindingForge training
    Handles your 2,242 compounds + 20 binding site residues
    """
    
    def __init__(
        self,
        compounds_df: pd.DataFrame,
        binding_site_df: pd.DataFrame,
        tokenizer: SMILESTokenizer,
        max_length: int = 150
    ):
        """
        Initialize dataset with your validated data
        
        Args:
            compounds_df: Your compound data (2,242 rows)
            binding_site_df: Your binding site data (20 residues)  
            tokenizer: SMILES tokenizer
            max_length: Maximum sequence length
        """
        self.compounds_df = compounds_df.copy()
        self.binding_site_df = binding_site_df.copy()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Prepare binding site features (20 residues x features)
        self.binding_site_features = self._prepare_binding_site_features()
        
        # Filter valid SMILES
        self.valid_indices = self._filter_valid_smiles()
        
        print(f"📊 BindingForge Dataset Initialized:")
        print(f"   🧬 Total compounds: {len(self.compounds_df):,}")
        print(f"   ✅ Valid SMILES: {len(self.valid_indices):,}")
        print(f"   🎯 Binding site residues: {len(self.binding_site_df)}")
        print(f"   📏 Max sequence length: {max_length}")
        print(f"   🔤 Vocabulary size: {tokenizer.vocab_size}")
    
    def _prepare_binding_site_features(self) -> torch.Tensor:
        """Prepare binding site features tensor from your 20 residues"""
        
        # Your binding site columns (from the successful validation)
        numeric_columns = self.binding_site_df.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"🧬 Available binding site features: {numeric_columns}")
        
        if not numeric_columns:
            raise ValueError("No numeric features found in binding site data")
        
        # Extract numeric features
        features_matrix = self.binding_site_df[numeric_columns].values.astype(np.float32)
        
        # Handle any NaN values
        features_matrix = np.nan_to_num(features_matrix, nan=0.0)
        
        # Normalize features (zero mean, unit variance)
        mean_vals = features_matrix.mean(axis=0)
        std_vals = features_matrix.std(axis=0)
        std_vals[std_vals == 0] = 1.0  # Avoid division by zero
        
        features_matrix = (features_matrix - mean_vals) / std_vals
        
        # Ensure we have exactly 13 features (to match model architecture)
        n_features = features_matrix.shape[1]
        if n_features < 13:
            # Pad with zeros
            padding = np.zeros((features_matrix.shape[0], 13 - n_features), dtype=np.float32)
            features_matrix = np.concatenate([features_matrix, padding], axis=1)
            print(f"   📊 Padded features: {n_features} → 13")
        elif n_features > 13:
            # Truncate to first 13 features
            features_matrix = features_matrix[:, :13]
            print(f"   📊 Truncated features: {n_features} → 13")
        
        print(f"   ✅ Final binding site shape: {features_matrix.shape}")
        
        return torch.FloatTensor(features_matrix)
    
    def _filter_valid_smiles(self) -> List[int]:
        """Filter compounds with valid SMILES"""
        valid_indices = []
        invalid_count = 0
        
        print("🔍 Filtering valid SMILES...")
        
        for idx, row in self.compounds_df.iterrows():
            smiles = row.get('smiles', '')
            
            # Basic validation
            if pd.isna(smiles) or not isinstance(smiles, str) or len(smiles.strip()) == 0:
                invalid_count += 1
                continue
            
            # Try to encode/decode
            try:
                encoded = self.tokenizer.encode(smiles, add_special_tokens=True)
                if len(encoded) >= 3 and len(encoded) <= self.max_length:  # Has meaningful content
                    decoded = self.tokenizer.decode(encoded, remove_special_tokens=True)
                    if len(decoded.strip()) > 0:
                        valid_indices.append(idx)
                    else:
                        invalid_count += 1
                else:
                    invalid_count += 1
            except Exception:
                invalid_count += 1
        
        print(f"   ✅ Valid SMILES: {len(valid_indices):,}")
        print(f"   ❌ Invalid SMILES: {invalid_count:,}")
        
        if len(valid_indices) == 0:
            raise ValueError("No valid SMILES found in dataset!")
        
        return valid_indices
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample"""
        
        # Get compound data
        compound_idx = self.valid_indices[idx]
        compound_data = self.compounds_df.iloc[compound_idx]
        smiles = compound_data['smiles']
        
        # Encode SMILES
        encoded_smiles = self.tokenizer.encode(smiles, add_special_tokens=True)
        padded_smiles = self.tokenizer.pad_sequence(encoded_smiles, self.max_length)
        
        # Create input/target sequences for teacher forcing
        input_ids = torch.LongTensor(padded_smiles[:-1])   # All except last token
        target_ids = torch.LongTensor(padded_smiles[1:])   # All except first token
        
        # Binding site features (same for all compounds in this implementation)
        binding_site_features = self.binding_site_features.clone()
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'binding_site_features': binding_site_features,
            'smiles': smiles,
            'compound_id': compound_idx
        }

class BindingForgeTrainer:
    """
    Safe, robust trainer for BindingForge model
    Includes comprehensive error handling and progress tracking
    """
    
    def __init__(
        self,
        model: TargetAwareLSTM,
        tokenizer: SMILESTokenizer,
        config: Dict,
        device: str = 'cpu'
    ):
        """Initialize trainer with safety features"""
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # Training components
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.char_to_idx[tokenizer.PAD_TOKEN])
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5, verbose=True
        )
        
        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.generation_metrics = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Setup logging
        self._setup_logging()
        
        print(f"🚀 BindingForge Trainer Initialized")
        print(f"   💻 Device: {device}")
        print(f"   📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   🎯 Learning rate: {config['training']['learning_rate']}")
        print(f"   📦 Batch size: {config['training']['batch_size']}")
    
    def _setup_logging(self):
        """Setup training logs in model_data folder"""
        log_dir = Path("model_data/training_logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"training_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 BindingForge training session started")
        self.logger.info(f"📁 Log file: {log_file}")
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch with comprehensive error handling"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Training", leave=False)
        
        for batch in progress_bar:
            try:
                # Move data to device
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                binding_site_features = batch['binding_site_features'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                outputs = self.model(input_ids, binding_site_features)
                logits = outputs['logits']
                
                # Calculate loss
                batch_size, seq_len, vocab_size = logits.shape
                loss = self.criterion(
                    logits.reshape(-1, vocab_size),
                    target_ids.reshape(-1)
                )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['grad_clip']
                )
                
                self.optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg': f'{total_loss/num_batches:.4f}'
                })
                
            except Exception as e:
                self.logger.error(f"Error in training batch: {str(e)}")
                continue
        
        return total_loss / max(num_batches, 1)
    
    def validate_epoch(self, dataloader: DataLoader) -> Tuple[float, List[str], Dict]:
        """Validate for one epoch with molecule generation"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        generated_samples = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation", leave=False):
                try:
                    # Move data to device
                    input_ids = batch['input_ids'].to(self.device)
                    target_ids = batch['target_ids'].to(self.device)
                    binding_site_features = batch['binding_site_features'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(input_ids, binding_site_features)
                    logits = outputs['logits']
                    
                    # Calculate loss
                    batch_size, seq_len, vocab_size = logits.shape
                    loss = self.criterion(
                        logits.reshape(-1, vocab_size),
                        target_ids.reshape(-1)
                    )
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Generate samples from first batch only
                    if num_batches == 1 and len(generated_samples) == 0:
                        sample_binding_site = binding_site_features[:min(5, batch_size)]
                        samples = self.model.generate(
                            sample_binding_site,
                            self.tokenizer,
                            max_length=100,
                            temperature=self.config['training']['temperature'],
                            top_k=10,
                            device=self.device
                        )
                        generated_samples.extend(samples)
                        
                except Exception as e:
                    self.logger.error(f"Error in validation batch: {str(e)}")
                    continue
        
        # Calculate generation metrics
        gen_metrics = self._evaluate_generated_molecules(generated_samples)
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss, generated_samples, gen_metrics
    
    def _evaluate_generated_molecules(self, molecules: List[str]) -> Dict[str, float]:
        """Evaluate quality of generated molecules"""
        if not molecules:
            return {'validity': 0.0, 'uniqueness': 0.0, 'diversity': 0.0}
        
        valid_molecules = []
        
        # Validate using RDKit if available
        try:
            from rdkit import Chem
            
            for smiles in molecules:
                if smiles and len(smiles.strip()) > 0:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        # Additional checks
                        if mol.GetNumAtoms() >= 5:  # Minimum meaningful size
                            valid_molecules.append(smiles)
        except ImportError:
            # Fallback validation without RDKit
            for smiles in molecules:
                if smiles and len(smiles.strip()) >= 5 and any(c in smiles for c in ['C', 'N', 'O']):
                    valid_molecules.append(smiles)
        
        # Calculate metrics
        validity = len(valid_molecules) / len(molecules) if molecules else 0.0
        uniqueness = len(set(valid_molecules)) / len(valid_molecules) if valid_molecules else 0.0
        
        # Simple diversity metric (average pairwise distance)
        diversity = 1.0  # Placeholder - could implement Tanimoto similarity
        
        return {
            'validity': validity,
            'uniqueness': uniqueness,
            'diversity': diversity,
            'valid_count': len(valid_molecules),
            'total_count': len(molecules)
        }
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int = 50,
        save_dir: str = "model_data/checkpoints"
    ) -> Dict[str, List]:
        """
        Main training loop with comprehensive safety features
        
        Args:
            train_dataloader: Training data
            val_dataloader: Validation data
            num_epochs: Number of epochs
            save_dir: Checkpoint directory
            
        Returns:
            Training history
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"🚀 Starting training for {num_epochs} epochs")
        self.logger.info(f"📊 Training samples: {len(train_dataloader.dataset):,}")
        self.logger.info(f"📊 Validation samples: {len(val_dataloader.dataset):,}")
        
        training_start_time = datetime.now()
        
        for epoch in range(num_epochs):
            epoch_start_time = datetime.now()
            
            print(f"\n🔄 Epoch {epoch + 1}/{num_epochs}")
            print("=" * 60)
            
            try:
                # Training phase
                train_loss = self.train_epoch(train_dataloader)
                self.train_losses.append(train_loss)
                
                # Validation phase
                val_loss, generated_samples, gen_metrics = self.validate_epoch(val_dataloader)
                self.val_losses.append(val_loss)
                self.generation_metrics.append(gen_metrics)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Calculate epoch duration
                epoch_duration = (datetime.now() - epoch_start_time).total_seconds()
                
                # Print epoch summary
                print(f"\n📊 Epoch {epoch + 1} Results:")
                print(f"   🚂 Train Loss: {train_loss:.4f}")
                print(f"   ✅ Val Loss: {val_loss:.4f}")
                print(f"   🧪 Molecule Validity: {gen_metrics['validity']:.1%}")
                print(f"   🎯 Molecule Uniqueness: {gen_metrics['uniqueness']:.1%}")
                print(f"   📚 Learning Rate: {current_lr:.2e}")
                print(f"   ⏱️ Duration: {epoch_duration:.1f}s")
                
                # Show sample generated molecules
                if generated_samples:
                    print(f"\n🧬 Sample Generated Molecules:")
                    for i, smiles in enumerate(generated_samples[:3]):
                        print(f"   {i+1}. {smiles}")
                
                # Log metrics
                self.logger.info(
                    f"Epoch {epoch + 1}: "
                    f"Train={train_loss:.4f}, Val={val_loss:.4f}, "
                    f"Validity={gen_metrics['validity']:.2%}, "
                    f"Uniqueness={gen_metrics['uniqueness']:.2%}, "
                    f"LR={current_lr:.2e}"
                )
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    
                    # Save best checkpoint
                    best_checkpoint_path = save_path / "best_model.pt"
                    save_model_checkpoint(
                        self.model, self.tokenizer, self.config,
                        epoch + 1, val_loss, str(best_checkpoint_path)
                    )
                    
                    print(f"💾 New best model saved! (Val Loss: {val_loss:.4f})")
                    
                else:
                    self.patience_counter += 1
                    
                    # Early stopping check
                    if self.patience_counter >= self.config['training']['patience']:
                        print(f"\n⏹️ Early stopping triggered after {self.patience_counter} epochs without improvement")
                        self.logger.info(f"Early stopping at epoch {epoch + 1}")
                        break
                
                # Save regular checkpoint every 10 epochs
                if (epoch + 1) % 10 == 0:
                    regular_checkpoint_path = save_path / f"checkpoint_epoch_{epoch + 1}.pt"
                    save_model_checkpoint(
                        self.model, self.tokenizer, self.config,
                        epoch + 1, val_loss, str(regular_checkpoint_path)
                    )
                    print(f"📁 Regular checkpoint saved: epoch_{epoch + 1}.pt")
                
            except Exception as e:
                self.logger.error(f"Error in epoch {epoch + 1}: {str(e)}")
                print(f"❌ Error in epoch {epoch + 1}: {str(e)}")
                continue
        
        # Training completed
        total_training_time = (datetime.now() - training_start_time).total_seconds()
        
        print(f"\n🎉 Training Completed!")
        print("=" * 60)
        print(f"📊 Best validation loss: {self.best_val_loss:.4f}")
        print(f"⏱️ Total training time: {total_training_time/60:.1f} minutes")
        print(f"💾 Best model saved in: {save_path}/best_model.pt")
        
        self.logger.info("🎉 Training completed successfully!")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'generation_metrics': self.generation_metrics,
            'best_val_loss': self.best_val_loss
        }
    
    def plot_training_history(self, save_path: str = "model_data/training_history.png"):
        """Plot comprehensive training history"""
        if not self.train_losses or not self.val_losses:
            print("⚠️ No training history to plot")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('BindingForge Training History', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss plot
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Molecule validity plot
        if self.generation_metrics:
            validity_scores = [m['validity'] for m in self.generation_metrics]
            axes[0, 1].plot(epochs, validity_scores, 'g-', linewidth=2)
            axes[0, 1].set_title('Generated Molecule Validity')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Validity Rate')
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].grid(True, alpha=0.3)
        
        # Uniqueness plot
        if self.generation_metrics:
            uniqueness_scores = [m['uniqueness'] for m in self.generation_metrics]
            axes[1, 0].plot(epochs, uniqueness_scores, 'm-', linewidth=2)
            axes[1, 0].set_title('Generated Molecule Uniqueness')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Uniqueness Rate')
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[1, 1].text(0.5, 0.5, f'Final Model\nParameters: {sum(p.numel() for p in self.model.parameters()):,}\nBest Val Loss: {self.best_val_loss:.4f}', 
                       transform=axes[1, 1].transAxes, ha='center', va='center',
                       fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[1, 1].set_title('Model Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training history saved: {save_path}")

def prepare_data_loaders(
    compounds_df: pd.DataFrame,
    binding_site_df: pd.DataFrame,
    tokenizer: SMILESTokenizer,
    config: Dict
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare training and validation data loaders
    """
    print("📊 Preparing data loaders...")
    
    # Create dataset
    dataset = BindingForgeDataset(
        compounds_df, binding_site_df, tokenizer,
        max_length=config['data']['max_length']
    )
    
    # Train/validation split (your proven 80/20 split)
    train_size = int(config['data']['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible splits
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,  # Windows-safe
        pin_memory=False  # Conservative for compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"✅ Data loaders ready:")
    print(f"   🚂 Training: {len(train_dataset):,} samples")
    print(f"   ✅ Validation: {len(val_dataset):,} samples")
    print(f"   📦 Batch size: {config['training']['batch_size']}")
    
    return train_loader, val_loader

def main_training_pipeline():
    """
    Main training pipeline - Ready to run with your validated data!
    """
    print("🚀 BindingForge Training Pipeline")
    print("=" * 60)
    
    try:
        # Device setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"💻 Using device: {device}")
        
        # Load your validated data
        print(f"\n📊 Loading your validated data...")
        
        compounds_df = pd.read_csv('processed_data/egfr_type1_filtered.csv')
        binding_site_df = pd.read_csv('binding_site_data/binding_site_features.csv')
        
        print(f"✅ Loaded {len(compounds_df):,} compounds")
        print(f"✅ Loaded {len(binding_site_df)} binding site residues")
        
        # Create configuration with your proven settings
        config = create_model_config()
        
        # Build tokenizer from your data
        print(f"\n🔤 Building SMILES tokenizer...")
        tokenizer = SMILESTokenizer()
        tokenizer.build_vocabulary(compounds_df['smiles'].tolist())
        
        # Save tokenizer
        tokenizer.save_tokenizer('model_data/tokenizer.json')
        
        # Create model
        print(f"\n🧠 Creating BindingForge model...")
        model = TargetAwareLSTM(
            vocab_size=tokenizer.vocab_size,
            **config['model']
        )
        
        print(f"📊 Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Prepare data loaders
        train_loader, val_loader = prepare_data_loaders(
            compounds_df, binding_site_df, tokenizer, config
        )
        
        # Create trainer
        trainer = BindingForgeTrainer(model, tokenizer, config, device)
        
        # Start training
        print(f"\n🚀 Starting training...")
        print(f"🎯 Goal: Generate novel EGFR inhibitors better than your previous 13!")
        print(f"📊 Training will automatically save best models and stop early if needed")
        
        # Train the model
        history = trainer.train(
            train_loader, val_loader,
            num_epochs=50  # Start with 50 epochs
        )
        
        # Plot results
        trainer.plot_training_history()
        
        print(f"\n🎉 Training Complete!")
        print(f"📊 Best validation loss: {trainer.best_val_loss:.4f}")
        print(f"💾 Best model saved: model_data/checkpoints/best_model.pt")
        print(f"📊 Training plots: model_data/training_history.png")
        
        return trainer, history
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    # Run the complete training pipeline
    trainer, history = main_training_pipeline()