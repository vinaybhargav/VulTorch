"""
Example 03: Transformer with AMP
Multihead Attention, Rotary Embeddings, SwiGLU, Autocast & GradScaler.
"""

import vultorch as torch
import vultorch.nn as nn
import vultorch.optim as optim
from vultorch.optim import lr_scheduler


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 64, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.swiglu = nn.SwiGLU()
        self.linear2 = nn.Linear(d_model * 2, d_model)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        mlp_out = self.linear2(self.swiglu(self.linear1(x)))
        x = self.norm2(x + mlp_out)
        return x


def main():
    device = "vk:0" if torch.vulkan.is_available() else "cpu"
    print(f"Running on device: {device}")

    model = TransformerBlock(d_model=64, num_heads=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = torch.amp.GradScaler(enabled=True)

    x = torch.randn(4, 16, 64, device=device)

    model.train()
    for step in range(3):
        optimizer.zero_grad()
        with torch.amp.autocast(device_type="vk", enabled=True):
            out = model(x)
            loss = out.sum()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        print(f"Step {step + 1}: Loss = {loss.item():.4f}")

    print("Example 03 completed successfully!")


if __name__ == "__main__":
    main()
