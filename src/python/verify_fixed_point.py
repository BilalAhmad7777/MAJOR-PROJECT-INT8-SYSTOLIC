import json
import numpy as np
import torch
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

BASE = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE / "src/python/models"
INT8_DIR = MODEL_DIR / "int8"
DATA_DIR = BASE / "data/MNIST"

with open(INT8_DIR / "quantization_metadata.json") as f:
    meta = json.load(f)

input_scale = meta["input_scale"]
w1_scale = meta["fc1_weight_scale"]
hidden_scale = meta["hidden_activation_scale"]

w1 = np.load(INT8_DIR / "fc1_weights_int8.npy").astype(np.int32)
b1 = np.load(INT8_DIR / "fc1_bias_int32.npy").astype(np.int32)

multiplier = (input_scale * w1_scale) / hidden_scale
q31 = int(np.rint(multiplier * (2**31)))

print("Floating multiplier:", multiplier)
print("Q31 integer multiplier:", q31)
print("Represented multiplier:", q31 / (2**31))

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

loader = DataLoader(test_data, batch_size=256, shuffle=False)

total_values = 0
mismatches = 0
max_error = 0
first_mismatch = None

for images, _ in loader:
    x = images.numpy().reshape(-1, 784).astype(np.float64)

    x_q = np.clip(
        np.rint(x / input_scale), -127, 127
    ).astype(np.int32)

    acc = x_q @ w1.T
    acc = acc + b1
    acc = np.maximum(acc, 0).astype(np.int64)

    # Python floating-point reference.
    reference = np.clip(
        np.rint(acc.astype(np.float64) * multiplier),
        0, 127
    ).astype(np.int32)

    # Q31 integer multiplication with ties-to-even rounding.
    product = acc * np.int64(q31)
    quotient = product >> 31
    remainder = product & ((1 << 31) - 1)
    half = 1 << 30

    rounded = quotient + (
        (remainder > half) |
        ((remainder == half) & ((quotient & 1) != 0))
    ).astype(np.int64)

    fixed = np.clip(rounded, 0, 127).astype(np.int32)

    difference = np.abs(reference.astype(np.int64) - fixed.astype(np.int64))

    count = int(np.count_nonzero(difference))
    mismatches += count
    total_values += reference.size
    max_error = max(max_error, int(difference.max()))

    if count and first_mismatch is None:
        index = np.argwhere(reference != fixed)[0]
        first_mismatch = {
            "reference": int(reference[tuple(index)]),
            "fixed": int(fixed[tuple(index)]),
            "accumulator": int(acc[tuple(index)])
        }

print("\n===== Q31 VERIFICATION =====")
print("Values checked:", total_values)
print("Mismatches:", mismatches)
print("Maximum absolute difference:", max_error)
print("First mismatch:", first_mismatch)

if mismatches == 0:
    print("\n[PASS] Q31 MATCHES PYTHON ON ALL TEST IMAGES")
else:
    print("\n[FAIL] Q31 DOES NOT MATCH PYTHON EXACTLY")
