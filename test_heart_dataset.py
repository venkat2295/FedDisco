import os
import sys
import torch
from torch.utils.data import DataLoader

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils.datasets import HeartDataset_truncated

def main():
    # Create dataset instances
    train_dataset = HeartDataset_truncated(
        root='./data/heart',
        train=True,
        download=True
    )

    test_dataset = HeartDataset_truncated(
        root='./data/heart',
        train=False,
        download=True
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    # Print dataset information
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")
    print(f"Number of features: {train_dataset.n_features}")

    # Get a batch of data to verify everything works
    features, labels = next(iter(train_loader))
    print(f"\nBatch shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")

if __name__ == "__main__":
    main()
