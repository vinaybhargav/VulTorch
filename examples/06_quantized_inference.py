"""
Example 06: Quantized Inference
In-Place int4 / int8 Weight Quantization on Vulkan.
"""

import vultorch as torch
import vultorch.nn as nn


def main():
    device = "vk:0" if torch.vulkan.is_available() else "cpu"
    print(f"Running on device: {device}")

    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
    ).to(device)

    x = torch.randn(16, 256, device=device)

    # 1. Full precision inference
    out_fp32 = model(x)

    # 2. Quantize model weights to int8 in-place
    model_q8 = torch.quantize(model, bits=8)
    out_q8 = model_q8(x)

    # Verify output similarity
    cos_sim = (out_fp32 * out_q8).sum() / (out_fp32.norm() * out_q8.norm() + 1e-6)
    print(f"Quantized int8 Cosine Similarity: {cos_sim.item():.4f}")

    print("Example 06 completed successfully!")


if __name__ == "__main__":
    main()
