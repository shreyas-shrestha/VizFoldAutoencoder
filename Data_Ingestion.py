import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def load_protein_data(data_directory="Proteins_layer47", normalize=True):
    """
    Load and preprocess protein data for autoencoder training.
    
    Args:
        data_directory (str): Path to folder containing protein .npy files
        normalize (bool): Whether to standardize the data
    
    Returns:
        tuple: (data_matrix, protein_names)
            - data_matrix: numpy array of shape (n_proteins, 49000) 
            - protein_names: list of protein names
    """
    
    # Find all .npy files in the directory
    data_path = Path(data_directory)
    if not data_path.exists():
        raise FileNotFoundError(f"Directory {data_directory} not found")
    
    npy_files = list(data_path.glob("*.npy"))
    if not npy_files:
        raise ValueError(f"No .npy files found in {data_directory}")
    
    print(f"Found {len(npy_files)} protein files")
    
    # Load and flatten each protein's data
    protein_data = []
    protein_names = []
    
    for file_path in npy_files:
        try:
            # Load the protein data (L×L×128)
            data = np.load(file_path)
            
            # Flatten to 1D vector
            flattened = data.flatten()
            
            # Ensure consistent size of 49,000 (pad or truncate if needed)
            if flattened.shape[0] > 49000:
                flattened = flattened[:49000]
            elif flattened.shape[0] < 49000:
                padded = np.zeros(49000)
                padded[:flattened.shape[0]] = flattened
                flattened = padded
            
            protein_data.append(flattened)
            protein_names.append(file_path.stem)
            
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            continue
    
    if not protein_data:
        raise ValueError("No protein data was successfully loaded")
    
    # Convert to numpy array
    data_matrix = np.array(protein_data)
    
    # Normalize the data if requested
    if normalize:
        scaler = StandardScaler()
        data_matrix = scaler.fit_transform(data_matrix)
    
    print(f"Loaded data shape: {data_matrix.shape}")
    print(f"Protein names: {protein_names}")
    
    return data_matrix, protein_names

# Simple usage example
if __name__ == "__main__":
    # Load the data
    data, names = load_protein_data("Proteins_layer47")
    
    print(f"\nData ready for autoencoder:")
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Value range: [{data.min():.3f}, {data.max():.3f}]")