import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import requests

def download_file(url, filepath):
    """
    Download a file from a URL to a local filepath.
    
    Args:
        url (str): URL to download from
        filepath (str): Local path to save the file
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
def prepare_heart_dataset(root_dir, test_size=0.2, random_state=42):
    """
    Download and prepare the Heart Disease dataset from UCI ML Repository.

    Args:
        root_dir (str): Directory to store the dataset
        test_size (float): Proportion of dataset to include in the test split
        random_state (int): Random state for reproducibility

    Returns:
        tuple: Paths to the training and test CSV files
    """
    # Create directory if it doesn't exist
    os.makedirs(root_dir, exist_ok=True)

    # URL for the Heart Disease dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

    # Download path
    raw_data_path = os.path.join(root_dir, 'heart_disease_raw.csv')

    # Download the dataset if it doesn't exist
    if not os.path.exists(raw_data_path):
        print(f"Downloading heart disease dataset to {raw_data_path}")
        download_file(url, raw_data_path)

    # Column names for the dataset
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]

    # Read CSV and handle missing values
    df = pd.read_csv(raw_data_path, names=column_names, na_values='?')

    # Replace missing values with median for numerical columns
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:
            df[column] = df[column].fillna(df[column].median())

    # Convert target to binary (0 for no disease, 1 for disease)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

    # Split into train and test sets
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['target']
    )

    # Save train and test sets
    train_path = os.path.join(root_dir, 'train.csv')
    test_path = os.path.join(root_dir, 'test.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Dataset prepared and saved to {root_dir}")
    print(f"Training set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")

    return train_path, test_path


if __name__ == '__main__':
    # Example usage
    try:
        train_path, test_path = prepare_heart_dataset('heart_dataset')
        print(f"Training data saved to: {train_path}")
        print(f"Test data saved to: {test_path}")
    except Exception as e:
        print(f"Error preparing dataset: {e}")