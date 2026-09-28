import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class PatchDescriptorNet(nn.Module):
    def __init__(self):
        super(PatchDescriptorNet, self).__init__()
        # Architecture < 500k parameters
        # Conv1: 1 input channel -> 32 output channels (param count: 32 * 1 * 9 = 288)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Conv2: 32 -> 64 (param count: 64 * 32 * 9 = 18,432)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Conv3: 64 -> 128 (param count: 128 * 64 * 9 = 73,728)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Max pooling 2x2 reduces 32x32 -> 16x16 -> 8x8 -> 4x4
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected: 128 * 4 * 4 (2048) -> 128 (param count: 2048 * 128 = 262,144)
        # Total parameters: ~355k
        self.fc = nn.Linear(128 * 4 * 4, 128)

    def forward(self, x):
        # x is expected to be (Batch, 1, 32, 32)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        # L2 normalize
        x = F.normalize(x, p=2, dim=1)
        return x

def train_descriptor(data_path="data/training/patch_triplets.npz", 
                     save_path="registration_engine/models/patch_descriptor.pt",
                     epochs=30, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Load dataset
    print(f"Loading data from {data_path}")
    data = np.load(data_path)
    anchors = data['anchors'].astype(np.float32) / 255.0
    positives = data['positives'].astype(np.float32) / 255.0
    negatives = data['negatives'].astype(np.float32) / 255.0
    
    # Reshape to (N, 1, 32, 32)
    anchors = np.expand_dims(anchors, axis=1)
    positives = np.expand_dims(positives, axis=1)
    negatives = np.expand_dims(negatives, axis=1)
    
    # Train/Val split (90/10)
    num_samples = anchors.shape[0]
    indices = np.random.permutation(num_samples)
    split_idx = int(0.9 * num_samples)
    
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    # Convert to tensors
    def to_tensor(arr, idx):
        return torch.tensor(arr[idx], dtype=torch.float32)
        
    train_anchors = to_tensor(anchors, train_idx)
    train_positives = to_tensor(positives, train_idx)
    train_negatives = to_tensor(negatives, train_idx)
    
    val_anchors = to_tensor(anchors, val_idx)
    val_positives = to_tensor(positives, val_idx)
    val_negatives = to_tensor(negatives, val_idx)
    
    train_dataset = torch.utils.data.TensorDataset(train_anchors, train_positives, train_negatives)
    val_dataset = torch.utils.data.TensorDataset(val_anchors, val_positives, val_negatives)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = PatchDescriptorNet().to(device)
    criterion = nn.TripletMarginLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for a, p, n in train_loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            
            optimizer.zero_grad()
            
            out_a = model(a)
            out_p = model(p)
            out_n = model(n)
            
            loss = criterion(out_a, out_p, out_n)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * a.size(0)
            
        train_loss /= len(train_dataset)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for a, p, n in val_loader:
                a, p, n = a.to(device), p.to(device), n.to(device)
                
                out_a = model(a)
                out_p = model(p)
                out_n = model(n)
                
                loss = criterion(out_a, out_p, out_n)
                val_loss += loss.item() * a.size(0)
                
        val_loss /= len(val_dataset)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model with val_loss: {val_loss:.4f}")
            
    print(f"Training complete. Best model saved to {save_path} with val_loss: {best_val_loss:.4f}")
