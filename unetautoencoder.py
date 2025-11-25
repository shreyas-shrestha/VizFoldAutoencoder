"""
Optimized U-Net Autoencoder for near-zero reconstruction.

Key features:
1. Skip connections - pass fine details directly from encoder to decoder
2. Residual blocks - better gradient flow
3. Interpolation to fixed size - handle variable inputs
4. Strong capacity with proper regularization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import os
import argparse
import time
import json
from typing import List, Optional
from PIL import Image


class InterpolatedDataset(Dataset):
    """Dataset with bilinear interpolation to fixed size"""
    
    def __init__(self, data_paths: List[str], target_size: int = 256,
                 channel_indices: Optional[List[int]] = None):
        self.data_paths = data_paths
        self.target_size = target_size
        self.channel_indices = channel_indices
        self.original_shapes = []
        self.data = []
        self.protein_names = []
        
        for path in data_paths:
            arr = np.load(path)
            self.original_shapes.append(arr.shape)
            self.protein_names.append(os.path.basename(os.path.dirname(path)))
            self.data.append(arr)
        
        if channel_indices is not None:
            self.data = [arr[:, :, channel_indices] for arr in self.data]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        arr = self.data[idx].copy()
        h, w, c = arr.shape
        
        # Normalize per sample to [-1, 1]
        max_val = np.abs(arr).max()
        if max_val > 1e-8:
            arr = arr / max_val
        
        # Interpolate each channel to target size
        interpolated = np.zeros((self.target_size, self.target_size, c), dtype=np.float32)
        for ch in range(c):
            img = Image.fromarray(arr[:, :, ch].astype(np.float32), mode='F')
            img_resized = img.resize((self.target_size, self.target_size), Image.BILINEAR)
            interpolated[:, :, ch] = np.array(img_resized)
        
        # Convert to tensor: (H, W, C) -> (C, H, W)
        tensor = torch.FloatTensor(interpolated).permute(2, 0, 1)
        
        return tensor, idx, max_val


class ResidualBlock(nn.Module):
    """Residual block with two convolutions"""
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out


class EncoderBlock(nn.Module):
    """Encoder block: Conv + ResBlock + Downsample"""
    
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.res_block = ResidualBlock(out_channels)
        self.downsample = nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1)
    
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        skip = self.res_block(x)  # Skip connection output
        x = self.downsample(skip)
        return x, skip


class DecoderBlock(nn.Module):
    """Decoder block: Upsample + Concat Skip + Conv + ResBlock"""
    
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        self.conv = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.res_block = ResidualBlock(out_channels)
    
    def forward(self, x, skip):
        x = self.upsample(x)
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn(self.conv(x)))
        x = self.res_block(x)
        return x


class UNetAutoEncoder(nn.Module):
    """
    U-Net style autoencoder with skip connections for near-perfect reconstruction.
    
    Architecture:
    - Encoder: progressively downsample while increasing channels
    - Skip connections: pass fine details to decoder
    - Decoder: progressively upsample while decreasing channels
    """
    
    def __init__(self, in_channels: int = 1, base_channels: int = 64, 
                 latent_channels: int = 512):
        super(UNetAutoEncoder, self).__init__()
        
        self.in_channels = in_channels
        
        # Initial projection
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.init_bn = nn.BatchNorm2d(base_channels)
        self.init_relu = nn.LeakyReLU(0.2, inplace=True)
        
        # Encoder (256 -> 128 -> 64 -> 32 -> 16 -> 8)
        self.enc1 = EncoderBlock(base_channels, base_channels)        # 256 -> 128
        self.enc2 = EncoderBlock(base_channels, base_channels * 2)    # 128 -> 64
        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4) # 64 -> 32
        self.enc4 = EncoderBlock(base_channels * 4, base_channels * 8) # 32 -> 16
        self.enc5 = EncoderBlock(base_channels * 8, latent_channels)   # 16 -> 8
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(latent_channels),
            ResidualBlock(latent_channels)
        )
        
        # Decoder (8 -> 16 -> 32 -> 64 -> 128 -> 256)
        self.dec5 = DecoderBlock(latent_channels, latent_channels, base_channels * 8)    # 8 -> 16
        self.dec4 = DecoderBlock(base_channels * 8, base_channels * 8, base_channels * 4) # 16 -> 32
        self.dec3 = DecoderBlock(base_channels * 4, base_channels * 4, base_channels * 2) # 32 -> 64
        self.dec2 = DecoderBlock(base_channels * 2, base_channels * 2, base_channels)     # 64 -> 128
        self.dec1 = DecoderBlock(base_channels, base_channels, base_channels)             # 128 -> 256
        
        # Final projection
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, in_channels, 1),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, x):
        # Initial
        x = self.init_relu(self.init_bn(self.init_conv(x)))
        
        # Encoder with skip connections
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)
        x, skip5 = self.enc5(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        x = self.dec5(x, skip5)
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        
        # Final projection
        x = self.final_conv(x)
        
        return x


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch):
    """Train for one epoch with gradient accumulation"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_data, batch_indices, _ in dataloader:
        batch_data = batch_data.to(device)
        
        optimizer.zero_grad()
        reconstructed = model(batch_data)
        loss = criterion(reconstructed, batch_data)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    per_sample_losses = []
    
    with torch.no_grad():
        for batch_data, batch_indices, _ in dataloader:
            batch_data = batch_data.to(device)
            reconstructed = model(batch_data)
            loss = criterion(reconstructed, batch_data)
            total_loss += loss.item()
            num_batches += 1
            
            # Per-sample loss
            for i in range(batch_data.size(0)):
                sample_loss = F.mse_loss(reconstructed[i], batch_data[i]).item()
                per_sample_losses.append(sample_loss)
    
    return total_loss / num_batches if num_batches > 0 else 0.0, per_sample_losses


def plot_losses(train_losses, val_losses, save_path: str):
    """Plot training and validation losses"""
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Train Loss', marker='o', markersize=2, alpha=0.8)
    plt.plot(val_losses, label='Val Loss', marker='s', markersize=2, alpha=0.8)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('U-Net Autoencoder Training', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to {save_path}")


def visualize_reconstruction(model, dataset, device, save_path: str, num_samples: int = 4):
    """Visualize reconstructions"""
    model.eval()
    
    num_samples = min(num_samples, len(dataset))
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    
    with torch.no_grad():
        for i in range(num_samples):
            tensor, idx, max_val = dataset[i]
            tensor = tensor.unsqueeze(0).to(device)
            reconstructed = model(tensor)
            
            # Get first channel for visualization
            original = tensor[0, 0].cpu().numpy()
            recon = reconstructed[0, 0].cpu().numpy()
            error = np.abs(original - recon)
            diff = original - recon
            
            mse = np.mean((original - recon) ** 2)
            
            ax = axes[i] if num_samples > 1 else axes
            
            # Original
            im0 = ax[0].imshow(original, cmap='viridis')
            ax[0].set_title(f'{dataset.protein_names[i]}\nOriginal', fontsize=10)
            ax[0].axis('off')
            plt.colorbar(im0, ax=ax[0], fraction=0.046)
            
            # Reconstructed
            im1 = ax[1].imshow(recon, cmap='viridis')
            ax[1].set_title(f'Reconstructed\nMSE: {mse:.6f}', fontsize=10)
            ax[1].axis('off')
            plt.colorbar(im1, ax=ax[1], fraction=0.046)
            
            # Absolute Error
            im2 = ax[2].imshow(error, cmap='hot')
            ax[2].set_title(f'Absolute Error\nMax: {error.max():.4f}', fontsize=10)
            ax[2].axis('off')
            plt.colorbar(im2, ax=ax[2], fraction=0.046)
            
            # Signed Difference
            vmax = max(abs(diff.min()), abs(diff.max()))
            im3 = ax[3].imshow(diff, cmap='RdBu', vmin=-vmax, vmax=vmax)
            ax[3].set_title(f'Signed Difference', fontsize=10)
            ax[3].axis('off')
            plt.colorbar(im3, ax=ax[3], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Reconstruction visualization saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train U-Net Autoencoder')
    parser.add_argument('--experiment', type=str, choices=['single_channel', 'two_channels'],
                       default='single_channel', help='Which experiment to run')
    parser.add_argument('--protein_dir', type=str, default='.', help='Directory containing protein folders')
    parser.add_argument('--layer', type=int, default=47, help='Layer number')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Max learning rate')
    parser.add_argument('--target_size', type=int, default=256, help='Target size for interpolation')
    parser.add_argument('--base_channels', type=int, default=64, help='Base channels for U-Net')
    parser.add_argument('--latent_channels', type=int, default=512, help='Latent channels')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Find proteins (exclude large ones)
    exclude_proteins = ['7tkv_A', '7pbk_A', '7kdx_B', 'channel_analysis']
    protein_dirs = [d for d in os.listdir(args.protein_dir) 
                   if os.path.isdir(os.path.join(args.protein_dir, d)) 
                   and not d.startswith('.') 
                   and d not in exclude_proteins]
    protein_dirs.sort()
    
    print(f"Found {len(protein_dirs)} proteins: {protein_dirs}")
    
    # Build data paths
    data_paths = []
    for p in protein_dirs:
        path = os.path.join(args.protein_dir, p, f'{p}_pair_block_{args.layer}.npy')
        if os.path.exists(path):
            data_paths.append(path)
    
    # Channel selection
    if args.experiment == 'single_channel':
        channel_indices = [0]
        num_channels = 1
    else:
        channel_indices = [0, 1]
        num_channels = 2
    
    print(f"Experiment: {args.experiment}")
    print(f"Channels: {channel_indices}")
    print(f"Target size: {args.target_size}x{args.target_size}")
    
    # Create dataset
    dataset = InterpolatedDataset(data_paths, target_size=args.target_size, 
                                  channel_indices=channel_indices)
    
    print(f"Dataset size: {len(dataset)}")
    
    # For small datasets, use all for training and validation (to measure overfitting)
    train_size = max(1, int(0.8 * len(dataset)))
    val_size = len(dataset) - train_size
    
    # Actually, let's train on ALL data and validate on ALL data (pure overfit mode)
    print(f"Training on ALL {len(dataset)} samples (overfit mode for near-zero reconstruction)")
    
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = UNetAutoEncoder(
        in_channels=num_channels,
        base_channels=args.base_channels,
        latent_channels=args.latent_channels
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    print(f"Architecture: U-Net with skip connections")
    print(f"  - 5 encoder blocks: {args.base_channels} -> {args.latent_channels} channels")
    print(f"  - 5 decoder blocks with skip connections")
    print(f"  - Residual blocks at each level")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # OneCycleLR scheduler for fast convergence
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos',
        div_factor=10,  # start_lr = max_lr / 10
        final_div_factor=100  # end_lr = max_lr / 1000
    )
    
    print(f"Optimizer: AdamW, Max LR: {args.lr}")
    print(f"Scheduler: OneCycleLR (10% warmup, cosine annealing)")
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80, flush=True)
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch)
        val_loss, per_sample_losses = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model
            torch.save(model.state_dict(), f'model_unet_{args.experiment}_best.pth')
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == args.epochs - 1:
            elapsed = time.time() - start_time
            remaining = epoch_time * (args.epochs - epoch - 1)
            print(f"Epoch {epoch+1:3d}/{args.epochs} | Train: {train_loss:.8f} | Val: {val_loss:.8f} | "
                  f"Best: {best_val_loss:.8f} | LR: {current_lr:.6f} | "
                  f"Time: {elapsed:.1f}s | Est. remaining: {remaining:.1f}s", flush=True)
    
    total_time = time.time() - start_time
    print("=" * 80)
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.10f}")
    print(f"Final train loss: {train_losses[-1]:.10f}")
    print(f"Final val loss: {val_losses[-1]:.10f}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    # Final per-protein losses
    print("\nPer-protein MSE:")
    _, final_per_sample = validate(model, val_loader, criterion, device)
    for i, protein in enumerate(dataset.protein_names):
        if i < len(final_per_sample):
            print(f"  {protein}: {final_per_sample[i]:.10f}")
    
    # Plot losses
    config_suffix = f"_unet_b{args.base_channels}_l{args.latent_channels}_s{args.target_size}"
    plot_path = f'losses_{args.experiment}_layer{args.layer}{config_suffix}.png'
    plot_losses(train_losses, val_losses, plot_path)
    
    # Save model
    model_path = f'model_{args.experiment}_layer{args.layer}{config_suffix}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Visualize reconstructions
    recon_path = f'reconstructions_{args.experiment}_layer{args.layer}{config_suffix}.png'
    visualize_reconstruction(model, dataset, device, recon_path, num_samples=min(7, len(dataset)))
    
    # Save info
    arch_info = {
        "architecture": "U-Net Autoencoder with Skip Connections",
        "in_channels": num_channels,
        "base_channels": args.base_channels,
        "latent_channels": args.latent_channels,
        "target_size": args.target_size,
        "total_params": total_params,
        "best_val_loss": best_val_loss,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "experiment": args.experiment,
        "channel_indices": channel_indices,
        "epochs": args.epochs,
        "per_protein_mse": {dataset.protein_names[i]: final_per_sample[i] 
                           for i in range(min(len(dataset.protein_names), len(final_per_sample)))},
        "key_features": [
            "Skip connections for fine detail preservation",
            "Residual blocks for better gradient flow",
            "OneCycleLR for fast convergence",
            "Bilinear interpolation for size normalization"
        ]
    }
    
    arch_info_path = f'arch_info_{args.experiment}_layer{args.layer}{config_suffix}.json'
    with open(arch_info_path, 'w') as f:
        json.dump(arch_info, f, indent=2)
    print(f"Architecture info saved to {arch_info_path}")


if __name__ == '__main__':
    main()
