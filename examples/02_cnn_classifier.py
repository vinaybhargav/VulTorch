"""
Example 02: CNN Classifier
2D Convolution, Batch Normalization, Max Pooling, and Adaptive Average Pooling.
"""

import vultorch as torch
import vultorch.nn as nn
import vultorch.optim as optim


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Linear(32 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        flat = feat.reshape((feat.shape[0], -1))
        return self.classifier(flat)


def main():
    device = "vk:0" if torch.vulkan.is_available() else "cpu"
    print(f"Running on device: {device}")

    model = SimpleCNN(num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(8, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (8,), dtype=torch.int64, device=device)

    model.train()
    for step in range(3):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        print(f"Step {step + 1}: Loss = {loss.item():.4f}")

    print("Example 02 completed successfully!")


if __name__ == "__main__":
    main()
