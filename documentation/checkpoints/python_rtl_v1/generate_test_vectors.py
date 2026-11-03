import json
import numpy as np
import torch
from pathlib import Path
from torchvision import datasets, transforms

BASE = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE / "src/python/models"
INT8_DIR = MODEL_DIR / "int8"
DATA_DIR = BASE / "data/MNIST"
OUT = BASE / "verification/test_vectors/fc1_hardware"

OUT.mkdir(parents=True, exist_ok=True)

with open(INT8_DIR / "quantization_metadata.json") as f:
    meta = json.load(f)

input_scale = meta["input_scale"]

weights = np.load(
    INT8_DIR / "fc1_weights_int8.npy"
).astype(np.int8)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset = datasets.MNIST(
    root=str(DATA_DIR),
    train=False,
    download=True,
    transform=transform
)

# Select four reproducible test images.
images = torch.stack([dataset[i][0] for i in range(4)])
labels = np.array([dataset[i][1] for i in range(4)])

x = images.numpy().reshape(4, 784)

x_q = np.clip(
    np.rint(x / input_scale), -127, 127
).astype(np.int8)


def pack_int8_matrix(matrix):
    """Pack row-major INT8 values into a 128-bit RTL input bus."""
    value = 0

    for i in range(4):
        for j in range(4):
            byte = int(matrix[i, j]) & 0xFF
            value |= byte << ((i * 4 + j) * 8)

    return f"{value:032x}"


def pack_int32_matrix(matrix):
    """Pack row-major INT32 values into a 512-bit result bus."""
    value = 0

    for i in range(4):
        for j in range(4):
            word = int(matrix[i, j]) & 0xFFFFFFFF
            value |= word << ((i * 4 + j) * 32)

    return f"{value:0128x}"


activation_lines = []
weight_lines = []
expected_lines = []
metadata_lines = []

tile_count = 0

# FC1: (4 x 784) multiplied by (784 x 32).
# Each hardware transaction computes one 4x4 tile product.
for output_start in range(0, 32, 4):
    for k in range(0, 784, 4):

        a_tile = np.zeros((4, 4), dtype=np.int8)
        b_tile = np.zeros((4, 4), dtype=np.int8)

        a_tile[:, :] = x_q[:, k:k+4]

        # B[k][j] = W[output_channel][k]
        b_tile[:, :] = weights[
            output_start:output_start+4, k:k+4
        ].T

        expected = (
            a_tile.astype(np.int64)
            @ b_tile.astype(np.int64)
        ).astype(np.int32)

        activation_lines.append(pack_int8_matrix(a_tile))
        weight_lines.append(pack_int8_matrix(b_tile))
        expected_lines.append(pack_int32_matrix(expected))

        metadata_lines.append(
            f"{tile_count} {output_start} {k}"
        )

        tile_count += 1


(OUT / "activation.hex").write_text(
    "\n".join(activation_lines) + "\n"
)

(OUT / "weights.hex").write_text(
    "\n".join(weight_lines) + "\n"
)

(OUT / "expected_partial_sums.hex").write_text(
    "\n".join(expected_lines) + "\n"
)

(OUT / "tile_index.txt").write_text(
    "tile_id output_start reduction_start\n"
    + "\n".join(metadata_lines) + "\n"
)

np.savetxt(OUT / "input_labels.txt", labels, fmt="%d")
np.save(OUT / "input_images_int8.npy", x_q)

print("===== HARDWARE VECTOR GENERATION =====")
print("Selected MNIST labels:", labels.tolist())
print("Activation matrix shape:", x_q.shape)
print("FC1 weight shape:", weights.shape)
print("Tile size: 4x4")
print("Reduction tiles per output group:", 196)
print("Output groups:", 8)
print("Total hardware transactions:", tile_count)
print("\nFiles generated:")
for path in sorted(OUT.iterdir()):
    print(path.name, "-", path.stat().st_size, "bytes")

print("\n[PASS] Hardware input and expected-output vectors generated.")
