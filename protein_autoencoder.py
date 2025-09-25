import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from Data_Ingestion import load_protein_data

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim=49000):  # 700*700 = 49,000
        super().__init__()
        # Encoder: 49000 -> 1024 -> 128 -> 8
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),  # proj_dim
            nn.ReLU(),
            nn.Linear(1024, 128),        # hidden_dim
            nn.ReLU(),
            nn.Linear(128, 8)            # latent_dim
        )
        
        # Decoder: 8 -> 128 -> 1024 -> 49000
        self.decoder = nn.Sequential(
            nn.Linear(8, 128),           # latent -> hidden
            nn.ReLU(),
            nn.Linear(128, 1024),        # hidden -> proj
            nn.ReLU(),
            nn.Linear(1024, input_dim)   # proj -> output
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_model(model, train_loader, val_loader, lr, weight_decay, epochs=50):
    """Train model with given hyperparameters"""
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            data = batch[0]
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                data = batch[0]
                output = model(data)
                loss = criterion(output, data)
                val_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
    
    return val_losses[-1]  # Return final validation loss

def k_fold_cross_validation(data, k=5):
    """Perform K-fold cross validation with hyperparameter grid search"""
    
    # Hyperparameter grid
    learning_rates = [0.01, 0.001, 0.0001]
    weight_decays = [1e-4, 1e-5]
    
    best_score = float('inf')
    best_params = None
    results = []
    
    print("Starting hyperparameter grid search with K-fold cross validation...")
    
    for lr in learning_rates:
        for wd in weight_decays:
            print(f"\nTesting lr={lr}, wd={wd}")
            
            # K-fold cross validation for this parameter combination
            kfold = KFold(n_splits=k, shuffle=True, random_state=42)
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(data)):
                # Split data
                train_data = data[train_idx]
                val_data = data[val_idx]
                
                # Create data loaders
                train_dataset = TensorDataset(torch.FloatTensor(train_data))
                val_dataset = TensorDataset(torch.FloatTensor(val_data))
                train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
                
                # Create fresh model for this fold
                model = SimpleAutoencoder(input_dim=train_data.shape[1])
                
                # Train and get validation score
                val_score = train_model(model, train_loader, val_loader, lr, wd)
                fold_scores.append(val_score)
                
                print(f"  Fold {fold+1}: {val_score:.6f}")
            
            # Calculate mean score for this parameter combination
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            
            results.append({
                'lr': lr,
                'wd': wd,
                'mean_score': mean_score,
                'std_score': std_score,
                'fold_scores': fold_scores
            })
            
            print(f"  Mean: {mean_score:.6f} ± {std_score:.6f}")
            
            # Track best parameters
            if mean_score < best_score:
                best_score = mean_score
                best_params = {'lr': lr, 'wd': wd}
    
    return best_params, best_score, results

def train_final_model(data, best_params):
    """Train final model with best parameters on full dataset"""
    print(f"\nTraining final model with best parameters: {best_params}")
    
    # Simple train/val split for final model
    split_idx = int(0.8 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    train_dataset = TensorDataset(torch.FloatTensor(train_data))
    val_dataset = TensorDataset(torch.FloatTensor(val_data))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Train final model
    model = SimpleAutoencoder(input_dim=train_data.shape[1])
    optimizer = optim.SGD(model.parameters(), lr=best_params['lr'], weight_decay=best_params['wd'])
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    print("Training final model...")
    for epoch in range(50):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            data_batch = batch[0]
            
            optimizer.zero_grad()
            output = model(data_batch)
            loss = criterion(output, data_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                data_batch = batch[0]
                output = model(data_batch)
                loss = criterion(output, data_batch)
                val_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/50, Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}')
    
    return model, train_losses, val_losses

def visualize_results(model, data, train_losses, val_losses, protein_names, original_L):
    """Visualize training curves and reconstructions"""
    
    # For visualization, we'll show the original L×L part (before padding)
    L_vis = min(original_L, 700)  # Use original L for visualization
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    # Show reconstruction examples
    plt.subplot(1, 2, 2)
    model.eval()
    with torch.no_grad():
        sample = torch.FloatTensor(data[:1])
        reconstructed = model(sample)
        
        # Reshape to 700x700 first, then take the meaningful part
        original_700 = data[0].reshape(700, 700)
        recon_700 = reconstructed[0].reshape(700, 700).numpy()
        
        # Extract the original L×L part for visualization
        original = original_700[:L_vis, :L_vis]
        recon = recon_700[:L_vis, :L_vis]
        
        plt.imshow(np.abs(original - recon), cmap='Reds')
        plt.title('Reconstruction Error')
        plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    # Show original vs reconstructed examples for first few samples
    model.eval()
    with torch.no_grad():
        for i in range(min(3, len(data))):
            sample = torch.FloatTensor(data[i:i+1])
            reconstructed = model(sample)
            
            # Reshape to 700x700 first, then take the meaningful part
            original_700 = data[i].reshape(700, 700)
            recon_700 = reconstructed[0].reshape(700, 700).numpy()
            
            # Extract the original L×L part for visualization
            original = original_700[:L_vis, :L_vis]
            recon = recon_700[:L_vis, :L_vis]
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            im1 = axes[0].imshow(original, cmap='viridis')
            axes[0].set_title(f'Original - {protein_names[i] if i < len(protein_names) else f"Sample {i}"}')
            plt.colorbar(im1, ax=axes[0])
            
            im2 = axes[1].imshow(recon, cmap='viridis')
            axes[1].set_title('Reconstructed')
            plt.colorbar(im2, ax=axes[1])
            
            im3 = axes[2].imshow(np.abs(original - recon), cmap='Reds')
            axes[2].set_title('Absolute Error')
            plt.colorbar(im3, ax=axes[2])
            
            plt.tight_layout()
            plt.show()

def main():
    # Load data
    print("Loading protein data...")
    data_matrix, protein_names, original_L = load_protein_data("Proteins_layer47", normalize=True)
    print(f"Data shape: {data_matrix.shape}")
    
    # K-fold cross validation with hyperparameter search
    best_params, best_score, all_results = k_fold_cross_validation(data_matrix, k=5)
    
    print(f"\n{'='*50}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*50}")
    
    # Print all results
    for result in all_results:
        print(f"lr={result['lr']:g}, wd={result['wd']:g}: {result['mean_score']:.6f} ± {result['std_score']:.6f}")
    
    print(f"\nBest parameters: lr={best_params['lr']:g}, wd={best_params['wd']:g}")
    print(f"Best score: {best_score:.6f}")
    
    # Train final model with best parameters
    final_model, train_losses, val_losses = train_final_model(data_matrix, best_params)
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(final_model, data_matrix, train_losses, val_losses, protein_names, original_L)
    
    # Save model
    torch.save(final_model.state_dict(), 'best_autoencoder.pth')
    print("Model saved as 'best_autoencoder.pth'")

if __name__ == "__main__":
    main()