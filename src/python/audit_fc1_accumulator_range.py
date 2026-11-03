import json
import numpy as np
import torch
from pathlib import Path
from torchvision import datasets, transforms

ROOT = Path("src/python/models/int8")

with open(ROOT / "quantization_metadata.json") as f:
    meta = json.load(f)

input_scale = meta["input_scale"]

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

raw_min = float("inf")
raw_max = float("-inf")
biased_min = float("inf")
biased_max = float("-inf")

raw_min_location = None
raw_max_location = None
biased_min_location = None
biased_max_location = None

raw_overflow_count = 0
biased_overflow_count = 0
total_values = 0

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1

print("Auditing all 10,000 MNIST test images...")

for image_index in range(len(dataset)):
    image, label = dataset[image_index]

    x = image.numpy().reshape(1, 784)

    x_q = np.clip(
        np.rint(x / input_scale),
        -127,
        127
    ).astype(np.int64)

    accumulators = x_q @ weights.T
    biased = accumulators + biases.reshape(1, 32)

    total_values += accumulators.size

    raw_overflow_count += int(
        np.count_nonzero(
            (accumulators < INT32_MIN) |
            (accumulators > INT32_MAX)
        )
    )

    biased_overflow_count += int(
        np.count_nonzero(
            (biased < INT32_MIN) |
            (biased > INT32_MAX)
        )
    )

    raw_min_index = int(np.argmin(accumulators))
    raw_max_index = int(np.argmax(accumulators))

    biased_min_index = int(np.argmin(biased))
    biased_max_index = int(np.argmax(biased))

    current_raw_min = int(accumulators.flat[raw_min_index])
    current_raw_max = int(accumulators.flat[raw_max_index])
    current_biased_min = int(biased.flat[biased_min_index])
    current_biased_max = int(biased.flat[biased_max_index])

    if current_raw_min < raw_min:
        raw_min = current_raw_min
        raw_min_location = (image_index, raw_min_index)

    if current_raw_max > raw_max:
        raw_max = current_raw_max
        raw_max_location = (image_index, raw_max_index)

    if current_biased_min < biased_min:
        biased_min = current_biased_min
        biased_min_location = (image_index, biased_min_index)

    if current_biased_max > biased_max:
        biased_max = current_biased_max
        biased_max_location = (image_index, biased_max_index)

print("\n===== FC1 ACCUMULATOR RANGE AUDIT =====")
print("Images checked:", len(dataset))
print("FC1 outputs per image:", weights.shape[0])
print("Total accumulator values:", total_values)

print("\nRAW ACCUMULATORS")
print("Minimum:", raw_min, "Location:", raw_min_location)
print("Maximum:", raw_max, "Location:", raw_max_location)
print("INT32 overflow values:", raw_overflow_count)

print("\nBIASED ACCUMULATORS")
print("Minimum:", biased_min, "Location:", biased_min_location)
print("Maximum:", biased_max, "Location:", biased_max_location)
print("INT32 overflow values:", biased_overflow_count)

print("\nINT32 RANGE")
print("Minimum:", INT32_MIN)
print("Maximum:", INT32_MAX)

if raw_overflow_count == 0 and biased_overflow_count == 0:
    print("\n[PASS] No INT32 overflow in the MNIST test dataset.")
else:
    print("\n[FAIL] INT32 overflow detected.")

print("\nNOTE: This audit covers the 10,000 MNIST test images only.")
