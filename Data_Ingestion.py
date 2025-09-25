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
        tuple: (data_matrix, protein_names, original_L)
            - data_matrix: numpy array of shape (n_proteins * 128, 49000) 
            - protein_names: list of protein names (repeated 128 times)
            - original_L: int, original L dimension before padding to 700x700
    """
    
    data_path = Path(data_directory)
    if not data_path.exists():
        raise FileNotFoundError(f"Directory {data_directory} not found")
    
    npy_files = list(data_path.glob("*.npy"))
    if not npy_files:
        raise ValueError(f"No .npy files found in {data_directory}")
    
    print(f"Found {len(npy_files)} protein files")
    
    # Load protein data and extract vectors for each of the 128 channels
    all_vectors = []
    all_protein_names = []
    original_L = None
    
    for file_path in npy_files:
        try:
            # Load the protein data (L×L×128)
            data = np.load(file_path)
            
            if original_L is None:
                original_L = data.shape[0]  # Store original L dimension
            
            # Extract each of the 128 feature vectors (L×L each)
            # This gives us 128 vectors of size L*L for each protein
            L = data.shape[0]
            for channel in range(data.shape[2]):
                # Get L×L matrix for this channel
                channel_matrix = data[:, :, channel]  # Shape: (L, L)
                
                # Pad or truncate to 700×700 to get exactly 49,000 elements
                if L < 700:
                    # Zero-pad to 700×700
                    padded_matrix = np.zeros((700, 700), dtype=channel_matrix.dtype)
                    padded_matrix[:L, :L] = channel_matrix
                    vector = padded_matrix.flatten()  # 700*700 = 49,000
                elif L > 700:
                    # Truncate to 700×700
                    truncated_matrix = channel_matrix[:700, :700]
                    vector = truncated_matrix.flatten()  # 700*700 = 49,000
                else:
                    # L == 700, perfect fit
                    vector = channel_matrix.flatten()  # 700*700 = 49,000
                
                all_vectors.append(vector)
                all_protein_names.append(f"{file_path.stem}_ch{channel:03d}")
            
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            continue
    
    if not all_vectors:
        raise ValueError("No protein data was successfully loaded")
    
    # Convert to numpy array
    data_matrix = np.array(all_vectors)
    
    # Normalize the data if requested
    if normalize:
        scaler = StandardScaler()
        data_matrix = scaler.fit_transform(data_matrix)
    
    print(f"Loaded data shape: {data_matrix.shape}")
    print(f"Total vectors: {len(all_vectors)} (10 proteins × 128 channels)")
    print(f"Vector size: {data_matrix.shape[1]} (padded to 700×700 = 49,000)")
    print(f"Original L dimension: {original_L}")
    
    return data_matrix, all_protein_names, original_L

# Simple usage example
if __name__ == "__main__":
    # Load the data
    data, names, original_L = load_protein_data("Proteins_layer47")
    
    print(f"\nData ready for autoencoder:")
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Value range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"Original L dimension: {original_L}")
