#!/usr/bin/env python3
"""
VulTorch Example: Training a Multi-Layer Perceptron (MLP) on AMD Radeon RX 7900 XTX via Vulkan
"""

import numpy as np
import time
import vultorch as torch
import vultorch.nn as nn
import vultorch.optim as optim
from vultorch.utils.data import TensorDataset, DataLoader

def generate_synthetic_mnist(num_samples=1000):
    """Generates synthetic 28x28 digit classification data."""
    np.random.seed(42)
    # 10 classes
    X = np.random.randn(num_samples, 28 * 28).astype(np.float32)
    # Create simple class signals
    y = np.random.randint(0, 10, size=num_samples)
    for c in range(10):
        mask = (y == c)
        X[mask, c * 70:(c + 1) * 70] += 2.5
    return X, y

class DigitClassifier(nn.Module):
    def __init__(self, device="vk:0"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 128, device=device),
            nn.ReLU(),
            nn.Linear(128, 64, device=device),
            nn.ReLU(),
            nn.Linear(64, 10, device=device)
        )

    def forward(self, x):
        return self.net(x)

def main():
    device = "vk:0"
    device_name = torch.vulkan.get_device_name(0)
    print("=" * 65)
    print(f"🔥 VulTorch MLP Training Demonstration")
    print(f"🎮 Target Hardware: {device_name} ({device})")
    print(f"🛡️  Proprietary Drivers: NONE (Pure Open Vulkan Compute)")
    print("=" * 65)

    X_train, y_train = generate_synthetic_mnist(800)
    X_val, y_val = generate_synthetic_mnist(200)

    train_ds = TensorDataset(torch.tensor(X_train, device=device), torch.tensor(y_train, device=device))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    model = DigitClassifier(device=device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()

    for epoch in range(1, 11):
        total_loss = 0.0
        batches = 0

        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / batches

        # Validation accuracy
        with torch.no_grad():
            val_x = torch.tensor(X_val, device=device)
            val_preds = model(val_x).numpy()
            pred_classes = np.argmax(val_preds, axis=1)
            acc = np.mean(pred_classes == y_val) * 100.0

        print(f" Epoch {epoch:2d}/10 | Train Loss: {avg_loss:.4f} | Val Accuracy: {acc:.1f}%")

    elapsed = time.time() - start_time
    print("-" * 65)
    print(f"✨ Training complete in {elapsed:.2f}s on {device_name} via Vulkan!")
    print("=" * 65)

if __name__ == "__main__":
    main()
