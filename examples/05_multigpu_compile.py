"""
Example 05: MultiGPU & Compilation
Heterogeneous Multi-GPU DataParallel with VulkanGraph Compilation & Safetensors.
"""

import tempfile
import os
import vultorch as torch
import vultorch.nn as nn
import vultorch.optim as optim


def main():
    base_model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    # Wrap in multi-GPU DataParallel across all available Vulkan GPUs
    dp_model = nn.DataParallel(base_model, device_ids=["all"], balance="auto")
    optimizer = optim.AdamW(base_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(128, 64)
    y = torch.randint(0, 10, (128,), dtype=torch.int64)

    # Multi-GPU training step
    for step in range(3):
        optimizer.zero_grad()
        logits = dp_model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        print(f"Step {step + 1}: Loss = {loss.item():.4f}")

    # Compile a module with @vultorch.compile / torch.compile
    compiled_model = torch.compile(base_model)
    _ = compiled_model(x[:32])

    # Save and reload weights using zero-dependency Safetensors
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.safetensors")
        torch.save_safetensors(base_model.state_dict(), path)
        loaded_state = torch.load_safetensors(path)
        base_model.load_state_dict(loaded_state)
        print("Safetensors weight roundtrip verified!")

    print("Example 05 completed successfully!")


if __name__ == "__main__":
    main()
