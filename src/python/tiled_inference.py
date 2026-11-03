import json
import numpy as np
import torch
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from fixed_point import q31_multiplier, requantize_q31

BASE = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE / "src/python/models"
INT8_DIR = MODEL_DIR / "int8"
DATA_DIR = BASE / "data/MNIST"
OUTPUT_DIR = BASE / "verification/test_vectors"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TILE = 4
BATCH = 4

with open(INT8_DIR / "quantization_metadata.json") as f:
    meta = json.load(f)

input_scale = meta["input_scale"]
w1_scale = meta["fc1_weight_scale"]
hidden_scale = meta["hidden_activation_scale"]
w2_scale = meta["fc2_weight_scale"]

w1 = np.load(INT8_DIR / "fc1_weights_int8.npy").astype(np.int32)
w2 = np.load(INT8_DIR / "fc2_weights_int8.npy").astype(np.int32)

b1 = np.load(INT8_DIR / "fc1_bias_float32.npy")
b2 = np.load(INT8_DIR / "fc2_bias_float32.npy")

b1_q = np.rint(b1 / (input_scale * w1_scale)).astype(np.int32)
b2_q = np.rint(b2 / (hidden_scale * w2_scale)).astype(np.int32)
fc1_multiplier = q31_multiplier(input_scale, w1_scale, hidden_scale)


def quantize_input(images):
    x = images.reshape(images.shape[0], -1).astype(np.float64)
    return np.clip(
        np.rint(x / input_scale), -127, 127
    ).astype(np.int32)


def tiled_linear(x, weights, bias):
    """
    Compute X @ W.T using 4x4 tiles.
    Accumulate every K tile before adding bias.
    """
    batch_size, input_features = x.shape
    output_features = weights.shape[0]

    output = np.zeros(
        (batch_size, output_features), dtype=np.int64
    )

    for batch_start in range(0, batch_size, TILE):
        batch_end = min(batch_start + TILE, batch_size)

        for output_start in range(0, output_features, TILE):
            output_end = min(output_start + TILE, output_features)

            partial = np.zeros(
                (batch_end - batch_start,
                 output_end - output_start),
                dtype=np.int64
            )

            for k in range(0, input_features, TILE):
                k_end = min(k + TILE, input_features)

                a_tile = x[batch_start:batch_end, k:k_end]
                b_tile = weights[output_start:output_end, k:k_end].T

                # The final tile is padded if necessary.
                a_pad = np.zeros((TILE, TILE), dtype=np.int32)
                b_pad = np.zeros((TILE, TILE), dtype=np.int32)

                a_pad[:a_tile.shape[0], :a_tile.shape[1]] = a_tile
                b_pad[:b_tile.shape[0], :b_tile.shape[1]] = b_tile

                tile_result = a_pad.astype(np.int64) @ b_pad.astype(np.int64)

                partial += tile_result[
                    :batch_end - batch_start,
                    :output_end - output_start
                ]

            output[
                batch_start:batch_end,
                output_start:output_end
            ] = partial + bias[output_start:output_end]

    if np.any(output < -(2**31)) or np.any(output > 2**31 - 1):
        raise OverflowError("INT32 accumulator overflow")

    return output.astype(np.int32)


def tiled_forward(images):
    # FC1
    x_q = quantize_input(images)
    acc1 = tiled_linear(x_q, w1, b1_q)
    acc1 = np.maximum(acc1, 0)

    hidden_q = requantize_q31(acc1, fc1_multiplier)

    # FC2
    acc2 = tiled_linear(hidden_q, w2, b2_q)

    return acc2


def reference_forward(images):
    """Untiled integer calculation for comparison."""
    x_q = quantize_input(images)

    acc1 = x_q.astype(np.int64) @ w1.T.astype(np.int64)
    acc1 += b1_q
    acc1 = np.maximum(acc1, 0)

    hidden_q = requantize_q31(acc1, fc1_multiplier)

    acc2 = hidden_q.astype(np.int64) @ w2.T.astype(np.int64)
    acc2 += b2_q

    return acc2


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_data = datasets.MNIST(
    root=str(DATA_DIR),
    train=False,
    download=True,
    transform=transform
)

loader = DataLoader(test_data, batch_size=BATCH, shuffle=False)

correct = 0
total = 0
mismatches = 0

print("===== 4x4 TILED MNIST INFERENCE =====")
print("FC1:", 784, "input features -> 32 outputs")
print("FC2:", 32, "input features -> 10 outputs")
print("Tile size:", TILE, "x", TILE)

for images, labels in loader:
    image_array = images.numpy()

    tiled_logits = tiled_forward(image_array)
    reference_logits = reference_forward(image_array)

    if not np.array_equal(tiled_logits, reference_logits):
        mismatches += np.count_nonzero(tiled_logits != reference_logits)

    predictions = np.argmax(tiled_logits, axis=1)
    correct += np.sum(predictions == labels.numpy())
    total += len(labels)

accuracy = 100.0 * correct / total

print("\n===== RESULTS =====")
print("Test images:", total)
print("Correct:", correct)
print("Accuracy:", f"{accuracy:.2f}%")
print("Tiled/reference numerical mismatches:", mismatches)

if mismatches == 0:
    print("\n[PASS] Tiled arithmetic matches untiled integer reference")
else:
    print("\n[FAIL] Tiled arithmetic mismatch detected")

# Save one reproducible 4x4 hardware test case.
sample_images, sample_labels = next(iter(loader))
sample_x = quantize_input(sample_images.numpy())

np.savetxt(
    OUTPUT_DIR / "mnist_sample_input_int8.txt",
    sample_x[0].reshape(-1),
    fmt="%d"
)

np.savetxt(
    OUTPUT_DIR / "mnist_sample_labels.txt",
    sample_labels[:1].numpy(),
    fmt="%d"
)

np.save(
    OUTPUT_DIR / "mnist_sample_fc1_input_tiles.npy",
    sample_x[:4]
)

print("Sample input vectors saved in:", OUTPUT_DIR)
