import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "src" / "python" / "models"
INT8_DIR = MODEL_DIR / "int8"
DATA_DIR = BASE_DIR / "data" / "MNIST"

DEVICE = "cpu"
BATCH_SIZE = 256


class MNISTNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(x.size(0), 784)
        hidden = self.relu(self.fc1(x))
        return self.fc2(hidden)


print("Loading model and INT8 parameters...")

checkpoint = torch.load(
    MODEL_DIR / "mnist_fc_model.pth",
    map_location=DEVICE,
    weights_only=True
)

model = MNISTNetwork()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

w1 = np.load(INT8_DIR / "fc1_weights_int8.npy").astype(np.int32)
w2 = np.load(INT8_DIR / "fc2_weights_int8.npy").astype(np.int32)

b1 = np.load(INT8_DIR / "fc1_bias_float32.npy")
b2 = np.load(INT8_DIR / "fc2_bias_float32.npy")

import json

with open(INT8_DIR / "quantization_metadata.json") as f:
    metadata = json.load(f)

input_scale = metadata["input_scale"]
w1_scale = metadata["fc1_weight_scale"]
w2_scale = metadata["fc2_weight_scale"]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST(
    root=str(DATA_DIR),
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root=str(DATA_DIR),
    train=False,
    download=True,
    transform=transform
)

# Calibrate hidden-layer activation scale using training images.
print("Calibrating hidden activation scale...")

calibration_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)

max_hidden = 0.0

with torch.no_grad():
    for images, _ in calibration_loader:
        hidden = model.relu(model.fc1(images.view(-1, 784)))
        max_hidden = max(max_hidden, hidden.max().item())

hidden_scale = max_hidden / 127.0 if max_hidden > 0 else 1.0

# Convert floating-point biases into accumulator units.
b1_int32 = np.rint(
    b1.astype(np.float64) / (input_scale * w1_scale)
).astype(np.int32)

b2_int32 = np.rint(
    b2.astype(np.float64) / (hidden_scale * w2_scale)
).astype(np.int32)

print("Input scale:", input_scale)
print("FC1 weight scale:", w1_scale)
print("Hidden activation scale:", hidden_scale)
print("FC2 weight scale:", w2_scale)

# Integer-only layer computation.
def int8_forward(images):
    x = images.numpy().reshape(-1, 784).astype(np.float64)

    # Quantize input.
    x_q = np.clip(
        np.rint(x / input_scale), -127, 127
    ).astype(np.int32)

    # FC1: INT8 x INT8 -> INT32 accumulation.
    acc1 = x_q @ w1.T
    acc1 = acc1 + b1_int32

    # ReLU.
    acc1 = np.maximum(acc1, 0)

    # Convert hidden activations to INT8.
    hidden_q = np.clip(
        np.rint(
            acc1.astype(np.float64)
            * (input_scale * w1_scale)
            / hidden_scale
        ),
        0, 127
    ).astype(np.int32)

    # FC2: INT8 x INT8 -> INT32 accumulation.
    acc2 = hidden_q @ w2.T
    acc2 = acc2 + b2_int32

    # All logits share the same positive scale,
    # so argmax can be taken directly on integer accumulators.
    return acc2


print("\n===== INT8 MNIST EVALUATION =====")

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        integer_logits = int8_forward(images)
        predictions = np.argmax(integer_logits, axis=1)

        correct += np.sum(predictions == labels.numpy())
        total += labels.size(0)

accuracy = 100.0 * correct / total

print("\n===== RESULTS =====")
print("Total test images:", total)
print("Correct predictions:", correct)
print("Incorrect predictions:", total - correct)
print(f"INT8 inference accuracy: {accuracy:.2f}%")

print("\nEvaluation complete. Quantization metadata was not modified.")
