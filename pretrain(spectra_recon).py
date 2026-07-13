import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import copy
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# ========================== 配置 ==========================
output_dir = 'output_multiscale_residual_ae_IP_test'
os.makedirs(output_dir, exist_ok=True)

SEED = 42
BATCH_SIZE = 32
PRETRAIN_EPOCHS = 50
LR_PRETRAIN = 1e-4
WEIGHT_DECAY = 1e-4
MASK_RATIO = 0.8
NOISE_STD = 0.05
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SPECTRAL_LENGTH = None

# ========================== 数据加载 ==========================
def load_4class_data(path):
    global SPECTRAL_LENGTH
    df = pd.read_excel(path)
    if 'ID' in df.columns:
        df = df.set_index('ID')
    
    labels = df['Class'].values.astype(np.int64)
    feats = df.iloc[:, 1:].values.astype(np.float32)
    
    mask = np.isin(labels, [1, 2, 3, 4])
    feats = feats[mask]
    labels = labels[mask]
    
    mu = feats.mean(axis=1, keepdims=True)
    std = feats.std(axis=1, keepdims=True) + 1e-8
    feats = (feats - mu) / std
    
    if SPECTRAL_LENGTH is None:
        SPECTRAL_LENGTH = feats.shape[1]
        print(f"检测到光谱长度: {SPECTRAL_LENGTH}")
    
    label_map = {1:0, 2:1, 3:2, 4:3}
    labels = np.array([label_map[l] for l in labels])
    
    print(f"加载 {path} → 样本数: {len(labels)} | 类别分布: {np.bincount(labels)}")
    return feats, labels

class SpectralDataset(Dataset):
    def __init__(self, feats, labels=None, augment=False, mask=False):
        self.feats = feats
        self.labels = labels
        self.augment = augment
        self.mask = mask
    
    def __len__(self):
        return len(self.feats)
    
    def __getitem__(self, idx):
        x = self.feats[idx].copy().astype(np.float32)
        
        if self.augment and np.random.rand() < 0.6:
            noise = np.random.normal(0, NOISE_STD, x.shape).astype(np.float32)
            x += noise
        
        if self.mask and np.random.rand() < 0.75:
            mask_len = int(len(x) * MASK_RATIO)
            mask_idx = np.random.choice(len(x), mask_len, replace=False)
            masked_x = x.copy()
            masked_x[mask_idx] = 0.0
            return torch.FloatTensor(masked_x), torch.FloatTensor(x)
        
        return torch.FloatTensor(x), torch.FloatTensor(x)

# ========================== 模型定义 ==========================
class MultiScaleResidualBlock1D(nn.Module):
    def __init__(self, channels, dilation_rates=[1,3,5]):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.BatchNorm1d(channels),
                nn.GELU()
            ) for d in dilation_rates
        ])
        self.fusion = nn.Sequential(
            nn.Conv1d(channels*len(dilation_rates), channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU()
        )
    
    def forward(self, x):
        residual = x
        feats = [branch(x) for branch in self.branches]
        fused = torch.cat(feats, dim=1)
        return self.fusion(fused) + residual

class MultiScaleEncoder(nn.Module):
    def __init__(self, base_channels=64, num_blocks=4):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.blocks = nn.ModuleList()
        self.skip_channels = []
        channels = base_channels
        for i in range(num_blocks):
            self.blocks.append(MultiScaleResidualBlock1D(channels))
            self.skip_channels.append(channels)
            if i < num_blocks-1:
                self.blocks.append(nn.Sequential(
                    nn.Conv1d(channels, channels*2, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm1d(channels*2),
                    nn.GELU()
                ))
                channels *= 2
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.latent_dim = channels
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.initial(x)
        skips = []
        for layer in self.blocks:
            x = layer(x)
            if isinstance(layer, MultiScaleResidualBlock1D):
                skips.append(x)
        latent = self.avgpool(x).flatten(1)
        return latent, skips

class MultiScaleDecoder(nn.Module):
    def __init__(self, latent_dim=512, spectral_length=200, skip_channels=None, base_channels=64):
        super().__init__()
        self.spectral_length = spectral_length
        self.skip_channels = skip_channels[::-1]
        
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, base_channels*8),
            nn.GELU(),
            nn.Unflatten(1, (base_channels*8, 1))
        )
        
        self.up_path = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        channels = base_channels*8
        for sc in self.skip_channels:
            self.up_path.append(nn.Sequential(
                nn.ConvTranspose1d(channels, channels//2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(channels//2),
                nn.GELU()
            ))
            channels //= 2
            self.up_path.append(MultiScaleResidualBlock1D(channels))
            self.skip_convs.append(nn.Conv1d(sc, channels, kernel_size=1, bias=False))
        
        self.final_in_channels = channels
        self.final = nn.Sequential(
            nn.Conv1d(self.final_in_channels, self.final_in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(self.final_in_channels),
            nn.GELU(),
            nn.Conv1d(self.final_in_channels, 1, kernel_size=1, bias=True)
        )
    
    def forward(self, latent, skip_connections):
        x = self.latent_proj(latent)
        skip_connections = skip_connections[::-1]
        skip_idx = 0
        for layer in self.up_path:
            x = layer(x)
            if isinstance(layer, MultiScaleResidualBlock1D):
                skip = skip_connections[skip_idx]
                if x.shape[-1] != skip.shape[-1]:
                    skip = nn.functional.interpolate(skip, size=x.shape[-1], mode='linear', align_corners=False)
                skip = self.skip_convs[skip_idx](skip)
                x = x + skip
                skip_idx += 1
        if x.shape[-1] != self.spectral_length:
            x = nn.functional.interpolate(x, size=self.spectral_length, mode='linear', align_corners=False)
        return self.final(x).squeeze(1)

class MultiScaleResidualAutoencoder(nn.Module):
    def __init__(self, latent_dim=512, spectral_length=200, skip_channels=None):
        super().__init__()
        self.encoder = MultiScaleEncoder(base_channels=64, num_blocks=4)
        self.decoder = MultiScaleDecoder(latent_dim=latent_dim, spectral_length=spectral_length, skip_channels=self.encoder.skip_channels)
    
    def forward(self, x):
        latent, skips = self.encoder(x)
        recon = self.decoder(latent, skips)
        return recon, latent

# ========================== 预训练 ==========================
def pretrain_encoder(pretrain_loader, save_path=None):
    model = MultiScaleResidualAutoencoder(latent_dim=512, spectral_length=SPECTRAL_LENGTH).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR_PRETRAIN, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20)
    
    best_loss = float('inf')
    best_encoder_state = None
    
    print("开始第一阶段：多尺度残差自监督预训练 Encoder...\n")
    
    for epoch in range(PRETRAIN_EPOCHS):
        model.train()
        total_loss = 0.0
        pbar = tqdm(pretrain_loader, desc=f"Pretrain Epoch {epoch+1}/{PRETRAIN_EPOCHS}")
        for masked, original in pbar:
            masked, original = masked.to(DEVICE), original.to(DEVICE)
            optimizer.zero_grad()
            recon, _ = model(masked)
            loss = criterion(recon, original)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(Loss=f"{loss.item():.5f}")
        avg_loss = total_loss / len(pretrain_loader)
        scheduler.step()
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_encoder_state = copy.deepcopy(model.encoder.state_dict())
        print(f"Pretrain Epoch {epoch+1:3d} | Avg Loss: {avg_loss:.6f} | Best: {best_loss:.6f}")
    
    if save_path and best_encoder_state:
        torch.save(best_encoder_state, save_path)
        print(f"\n最佳 Encoder 权重已保存至: {save_path}")
    return best_encoder_state

# ========================== 主程序 ==========================
def main_pretrain():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    print("加载训练数据用于预训练...")
    X_train, _ = load_4class_data('Train_IndianPines.xlsx')
    
    pretrain_ds = SpectralDataset(X_train, augment=True, mask=True)
    pretrain_loader = DataLoader(pretrain_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    
    encoder_save_path = os.path.join(output_dir, 'encoder_multiscale_residual_best.pth')
    best_encoder = pretrain_encoder(pretrain_loader, encoder_save_path)
    
    print("\n=== 第一阶段多尺度残差预训练完成 ===")

if __name__ == "__main__":
    main_pretrain()
