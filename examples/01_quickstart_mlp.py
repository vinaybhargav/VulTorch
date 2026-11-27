"""
Example 01: Quickstart MLP
Drop-in PyTorch-compatible MLP training & evaluation on Vulkan.
"""

import vultorch as torch
import vultorch.nn as nn
import vultorch.optim as optim
from vultorch.utils.data import TensorDataset, DataLoader


def main():
    device = "vk:0" if torch.vulkan.is_available() else "cpu"
    print(f"Running on device: {device}")

    # 1. Model definition
    model = nn.Sequential(
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)

    # 2. Synthetic dataset & dataloader
    x_data = torch.randn(256, 28 * 28)
    y_data = torch.randint(0, 10, (256,), dtype=torch.int64)
    dataset = TensorDataset(x_data, y_data)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 3. Training loop
    model.train()
    for epoch in range(2):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}: Loss = {total_loss / len(loader):.4f}")

    print("Example 01 completed successfully!")


if __name__ == "__main__":
    main()
