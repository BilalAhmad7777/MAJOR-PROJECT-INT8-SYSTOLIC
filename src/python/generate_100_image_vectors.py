
import json
import numpy as np
import torch
from pathlib import Path
from torchvision import datasets, transforms
from fixed_point import q31_multiplier, requantize_q31

BASE = Path(__file__).resolve().parents[2]
INT8_DIR = BASE / "src/python/models/int8"
DATA_DIR = BASE / "data/MNIST"
OUT = BASE / "verification/test_vectors/mnist_1000"

NUM_IMAGES = 1000
BATCH_SIZE = 4
NUM_GROUPS = 8
TILES_PER_GROUP = 196

OUT.mkdir(parents=True, exist_ok=True)

with open(INT8_DIR / "quantization_metadata.json") as f:
    meta = json.load(f)

input_scale = meta["input_scale"]

weights = np.load(
    INT8_DIR / "fc1_weights_int8.npy"
).astype(np.int64)

fc1_bias = np.load(
    INT8_DIR / "fc1_bias_int32.npy"
).astype(np.int64)

fc2_weights = np.load(
    INT8_DIR / "fc2_weights_int8.npy"
).astype(np.int64)

fc2_bias = np.load(
    INT8_DIR / "fc2_bias_int32.npy"
).astype(np.int64)

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

# Use 1000 reproducible test images.
images = torch.stack([dataset[i][0] for i in range(NUM_IMAGES)])
labels = np.array([dataset[i][1] for i in range(NUM_IMAGES)])

x = images.numpy().reshape(NUM_IMAGES, 784)

x_q = np.clip(
    np.rint(x / input_scale), -127, 127
).astype(np.int8)

# Python INT8 golden inference.
acc1 = x_q.astype(np.int64) @ weights.T
biased1 = acc1 + fc1_bias.reshape(1, 32)

hidden_scale = meta["hidden_activation_scale"]
multiplier = q31_multiplier(
    input_scale, meta["fc1_weight_scale"], hidden_scale
)

hidden = requantize_q31(np.maximum(biased1, 0), multiplier).astype(np.int8)

logits = (
    hidden.astype(np.int64) @ fc2_weights.T
    + fc2_bias.reshape(1, 10)
)

predictions = np.argmax(logits, axis=1)

# Pack row-major INT8 matrices into 128-bit buses.
def pack_int8_matrix(matrix):
    value = 0
    for i in range(4):
        for j in range(4):
            byte = int(matrix[i, j]) & 0xFF
            value |= byte << ((i * 4 + j) * 8)
    return f"{value:032x}"

activation_lines = []
weight_lines = []
metadata_lines = []

# One file contains all 25 batches, in order.
for batch_start in range(0, NUM_IMAGES, BATCH_SIZE):
    batch = x_q[batch_start:batch_start + BATCH_SIZE]

    for output_start in range(0, 32, 4):
        for k in range(0, 784, 4):
            a_tile = np.zeros((4, 4), dtype=np.int8)
            b_tile = np.zeros((4, 4), dtype=np.int8)

            a_tile[:, :] = batch[:, k:k + 4]
            b_tile[:, :] = weights[
                output_start:output_start + 4, k:k + 4
            ].T

            activation_lines.append(pack_int8_matrix(a_tile))
            weight_lines.append(pack_int8_matrix(b_tile))

            metadata_lines.append(
                f"{batch_start} {output_start} {k}"
            )

# FC1 bias values.
(OUT / "fc1_bias_int32.hex").write_text(
    "".join(f"{int(v) & 0xFFFFFFFF:08x}\n" for v in fc1_bias)
)

# FC2 parameters.
(OUT / "fc2_weights_int8.hex").write_text(
    "".join(
        f"{int(v) & 0xFF:02x}\n"
        for v in fc2_weights.reshape(-1)
    )
)

(OUT / "fc2_bias_int32.hex").write_text(
    "".join(f"{int(v) & 0xFFFFFFFF:08x}\n" for v in fc2_bias)
)

# FC2 expected logits, labels and Python predictions.
(OUT / "expected_logits_int32.hex").write_text(
    "".join(
        f"{int(v) & 0xFFFFFFFF:08x}\n"
        for v in logits.reshape(-1)
    )
)

np.savetxt(OUT / "labels.txt", labels, fmt="%d")
np.savetxt(OUT / "predictions_python.txt", predictions, fmt="%d")

np.save(OUT / "input_images_int8.npy", x_q)
np.save(OUT / "hidden_activations_int8.npy", hidden)

(OUT / "activation.hex").write_text(
    "\n".join(activation_lines) + "\n"
)

(OUT / "weights.hex").write_text(
    "\n".join(weight_lines) + "\n"
)

(OUT / "tile_index.txt").write_text(
    "batch_start output_start reduction_start\n"
    + "\n".join(metadata_lines) + "\n"
)

print("===== 1000-IMAGE VECTOR GENERATION =====")
print("Images:", NUM_IMAGES)
print("Batches:", NUM_IMAGES // BATCH_SIZE)
print("Python accuracy:", f"{np.mean(predictions == labels) * 100:.2f}%")
print("FC1 activation tiles:", len(activation_lines))
print("Expected logits:", logits.shape)
print("Output directory:", OUT)
print("[PASS] 1000-image vectors generated.")
