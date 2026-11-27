"""
Example 04: Mamba & Snake Activations
State-Space Sequence Models and Periodic Wavelet-like Activations.
"""

import vultorch as torch
import vultorch.nn as nn
import vultorch.optim as optim


class SnakeMambaModel(nn.Module):
    def __init__(self, d_model: int = 32):
        super().__init__()
        self.mamba = nn.Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.snake = nn.Snake(d_model, channel_dim=-1)
        self.linear = nn.Linear(d_model, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, D)
        h = self.mamba(x)
        h = self.snake(h)
        return self.linear(h.mean(dim=1))


def main():
    device = "vk:0" if torch.vulkan.is_available() else "cpu"
    print(f"Running on device: {device}")

    model = SnakeMambaModel(d_model=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(4, 16, 32, device=device)

    model.train()
    for step in range(3):
        optimizer.zero_grad()
        out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        print(f"Step {step + 1}: Loss = {loss.item():.4f}")

    print("Example 04 completed successfully!")


if __name__ == "__main__":
    main()
