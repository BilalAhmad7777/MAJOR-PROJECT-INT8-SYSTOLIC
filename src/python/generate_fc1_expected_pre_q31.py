
import json
import numpy as np
import torch
from pathlib import Path
from torchvision import datasets, transforms

ROOT = Path("src/python/models/int8/")
OUT = Path("verification/test_vectors/fc1_hardware/")

with open(ROOT / "quantization_metadata.json") as f:
    meta = json.load(f)

input_scale = meta["input_scale"]
weight_scale = meta["fc1_weight_scale"]
hidden_scale = meta["hidden_activation_scale"]

weights = np.load(ROOT / "fc1_weights_int8.npy").astype(np.int64)
biases = np.load(ROOT / "fc1_bias_int32.npy").astype(np.int64)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

# Use exactly the same four images as the hardware generator.
images = torch.stack([dataset[i][0] for i in range(4)])
labels = np.array([dataset[i][1] for i in range(4)])

x = images.numpy().reshape(4, 784)

# Match hardware input quantization exactly.
x_q = np.clip(
    np.rint(x / input_scale), -127, 127
).astype(np.int64)

# Integer FC1 computation.
accumulators = x_q @ weights.T
biased = accumulators + biases.reshape(1, 32)

# ReLU and requantization.
relu_values = np.maximum(biased, 0)

multiplier = (
    input_scale * weight_scale / hidden_scale
)

expected = np.clip(
    np.rint(relu_values.astype(np.float64) * multiplier),
    0, 127
).astype(np.int8)

OUT.mkdir(parents=True, exist_ok=True)

# Export expected INT8 outputs.
np.savetxt(
    OUT / "expected_hidden_int8.hex",
    expected.reshape(-1),
    fmt="%02x"
)

# Export INT32 accumulators.
# Export INT32 accumulators as 32-bit two's-complement hexadecimal.
(OUT / "fc1_accumulators_int32.hex").write_text(
    "".join(
        f"{int(v) & 0xFFFFFFFF:08x}\n"
        for v in accumulators.reshape(-1)
    )
)

# Export biases as 32-bit two's-complement hexadecimal.
(OUT / "fc1_bias_int32.hex").write_text(
    "".join(
        f"{int(v) & 0xFFFFFFFF:08x}\n"
        for v in biases
    )
)

# Export image labels.
with open(OUT / "expected_hidden_labels.txt", "w") as f:
    f.write("\n".join(map(str, labels)) + "\n")

print("FC1 expected-output generation complete.")
print("Image labels:", labels.tolist())
print("Accumulator shape:", accumulators.shape)
print("Expected output shape:", expected.shape)
print("Expected sample 0, first 4 neurons:", expected[0, :4])
print("Multiplier:", multiplier)
print("Files saved in:", OUT)
