import os
import zipfile
import subprocess

from constants import DATA_DIR

def download_dataset():
    data_parent_dir = os.path.dirname(DATA_DIR)
    if not os.path.exists(data_parent_dir):
        os.makedirs(data_parent_dir)
    
    if not os.path.exists(DATA_DIR):
        print("Downloading Plant Village dataset from Kaggle...")
        zip_path = os.path.join(data_parent_dir, 'plant-village.zip')
        
        # Use Kaggle CLI to download
        try:
            subprocess.run(['kaggle', 'datasets', 'download', '-d', 'arjuntejaswi/plant-village', '-p', data_parent_dir], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("Failed to download dataset. Ensure Kaggle API is set up.") from e
        
        # Extract zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_parent_dir)
        
        # Assume extraction creates 'PlantVillage' or similar; rename if needed
        extracted_dir = os.path.join(data_parent_dir, 'PlantVillage')  # Adjust if extraction name differs
        if not os.path.exists(DATA_DIR):
            os.rename(extracted_dir, DATA_DIR)
        
        # Clean up zip
        os.remove(zip_path)
        print("Dataset downloaded and extracted.")
    else:
        print("Dataset already exists.")
