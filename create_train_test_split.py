#!/usr/bin/env python3
"""
Create train/test splits for the MayJun2022 dataset to enable proper training.
"""

import os
import shutil
import random
from pathlib import Path

def create_train_test_split(data_dir, train_ratio=0.8):
    """Create train/test splits from processed data."""
    
    processed_data_dir = Path(data_dir) / "processed_data"
    
    if not processed_data_dir.exists():
        print(f"Error: {processed_data_dir} does not exist")
        return False
    
    # Create train and test directories
    train_dir = processed_data_dir / "train" 
    test_dir = processed_data_dir / "test"
    
    # Remove existing train/test dirs if they exist
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    # Get all .txt files
    txt_files = list(processed_data_dir.glob("*.txt"))
    print(f"Found {len(txt_files)} data files")
    
    if len(txt_files) == 0:
        print("No .txt files found in processed_data directory")
        return False
    
    # Shuffle and split
    random.seed(42)  # For reproducible splits
    random.shuffle(txt_files)
    
    split_idx = int(len(txt_files) * train_ratio)
    train_files = txt_files[:split_idx]
    test_files = txt_files[split_idx:]
    
    print(f"Train files: {len(train_files)}")
    print(f"Test files: {len(test_files)}")
    
    # Copy files to train directory
    for file_path in train_files:
        shutil.copy2(file_path, train_dir / file_path.name)
    
    # Copy files to test directory  
    for file_path in test_files:
        shutil.copy2(file_path, test_dir / file_path.name)
    
    print(f"✅ Successfully created train/test split")
    print(f"   Train: {train_dir}")
    print(f"   Test: {test_dir}")
    
    return True

if __name__ == "__main__":
    data_dir = "/home/ssangeetha3/git/ctaf-intent-inference/dataset/MayJun2022"
    create_train_test_split(data_dir)