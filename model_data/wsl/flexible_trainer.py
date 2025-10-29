#!/usr/bin/env python3
# WSL CPU Trainer for BindingForge
# Compatible version that avoids CUDA issues

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime
import json
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def setup_environment():
    """Setup WSL environment - force CPU to avoid CUDA issues"""
    print("🚀 BindingForge WSL Trainer (CPU Mode)")
    print("=" * 60)
    
    # Force CPU to avoid CUDA compatibility issues
    device = torch.device('cpu')
    
    print("🖥️ System Setup:")
    print(f"   💻 Device: CPU (Avoiding CUDA compatibility issues)")
    print(f"   🔄 CPU cores: {os.cpu_count()}")
    print(f"   🧠 PyTorch version: {torch.__version__}")
    
    # Set number of threads for better CPU performance
    torch.set_num_threads(min(8, os.cpu_count()))
    print(f"   ⚡ Using {torch.get_num_threads()} CPU threads")
    
    return device

def find_data_files():
    """Find and validate your actual data files"""
    print("\n📊 Searching for data files...")
    
    # Your actual data file paths
    data_files = {
        'compounds': None,
        'binding_site': None
    }
    
    # Look in albumin_binding_analysis folder for your preferred file
    albumin_file = '/mnt/c/Users/admin/BF-final-version/albumin_binding_analysis/egfr_type1_albumin_binding.csv'
    if Path(albumin_file).exists():
        data_files['compounds'] = albumin_file
        print(f"   ✅ Found compounds: egfr_type1_albumin_binding.csv")
    else:
        # Fallback to filtered file
        filtered_file = '/mnt/c/Users/admin/BF-final-version/processed_data/egfr_type1_filtered.csv'
        if Path(filtered_file).exists():
            data_files['compounds'] = filtered_file
            print(f"   ✅ Found compounds: egfr_type1_filtered.csv")
    
    # Look for binding site file
    binding_file = '/mnt/c/Users/admin/BF-final-version/binding_site_data/binding_site_features.csv'
    if Path(binding_file).exists():
        data_files['binding_site'] = binding_file
        print(f"   ✅ Found binding site: binding_site_features.csv")
    
    if not data_files['compounds'] or not data_files['binding_site']:
        return None
    
    return data_files

def load_and_validate_data(data_files):
    """Load and validate your data"""
    print("\n📊 Loading data...")
    
    try:
        # Load compound data
        compounds_df = pd.read_csv(data_files['compounds'])
        print(f"   🧬 Compounds loaded: {len(compounds_df):,} rows, {len(compounds_df.columns)} columns")
        
        # Check for SMILES column
        smiles_cols = [col for col in compounds_df.columns if 'smiles' in col.lower()]
        if not smiles_cols:
            print("   ❌ No SMILES column found!")
            return None, None
        
        smiles_col = smiles_cols[0]
        valid_smiles = compounds_df[smiles_col].notna().sum()
        print(f"   ✅ Valid SMILES: {valid_smiles:,} ({valid_smiles/len(compounds_df)*100:.1f}%)")
        
        # Load binding site data
        binding_site_df = pd.read_csv(data_files['binding_site'])
        print(f"   🎯 Binding site loaded: {len(binding_site_df)} residues, {len(binding_site_df.columns)} features")
        
        # Show sample data
        print(f"\n📋 Sample compound data:")
        key_cols = [smiles_col] + [col for col in ['MW', 'LogP', 'QED', 'standard_value'] if col in compounds_df.columns][:5]
        print(compounds_df[key_cols].head(3).to_string(index=False))
        
        print(f"\n📋 Binding site overview:")
        numeric_cols = binding_site_df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"   Numeric features ({len(numeric_cols)}): {numeric_cols[:8]}...")
        
        return compounds_df, binding_site_df
        
    except Exception as e:
        print(f"   ❌ Error loading data: {str(e)}")
        return None, None

# Optimized SMILES tokenizer
class OptimizedTokenizer:
    """CPU-optimized SMILES tokenizer"""
    
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