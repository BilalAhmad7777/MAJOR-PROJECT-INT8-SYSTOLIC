import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "src" / "python" / "models"
DATA_DIR = BASE_DIR / "data" / "MNIST"

MODEL_PATH = MODEL_DIR / "mnist_fc_model.pth"
OUTPUT_DIR = MODEL_DIR / "int8"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)


class MNISTNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(x.size(0), 784)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def quantize_symmetric(tensor):
    """Symmetric signed INT8 quantization."""
    tensor = tensor.detach().cpu().to(torch.float64)
    max_abs = tensor.abs().max().item()

    if max_abs == 0:
        scale = 1.0
    else:
        scale = max_abs / 127.0

    quantized = torch.round(tensor / scale).clamp(-127, 127)
    return quantized.to(torch.int8), scale


print("Loading trained model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

model = MNISTNetwork()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Quantizing weights...")

state = model.state_dict()

w1_int8, w1_scale = quantize_symmetric(state["fc1.weight"])
w2_int8, w2_scale = quantize_symmetric(state["fc2.weight"])

# Preserve biases as floating-point values for now.
b1_float = state["fc1.bias"].cpu().numpy().astype(np.float32)
b2_float = state["fc2.bias"].cpu().numpy().astype(np.float32)

# Load representative MNIST images for input calibration.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

calibration_dataset = datasets.MNIST(
    root=str(DATA_DIR),
    train=True,
    download=True,
    transform=transform
)

calibration_loader = DataLoader(
    calibration_dataset,
    batch_size=1000,
    shuffle=False
)

max_input = 0.0

with torch.no_grad():
    for images, _ in calibration_loader:
        max_input = max(max_input, images.abs().max().item())

input_scale = max_input / 127.0

# Save quantized weights and metadata.
np.save(OUTPUT_DIR / "fc1_weights_int8.npy", w1_int8.numpy())
np.save(OUTPUT_DIR / "fc2_weights_int8.npy", w2_int8.numpy())

np.save(OUTPUT_DIR / "fc1_bias_float32.npy", b1_float)
np.save(OUTPUT_DIR / "fc2_bias_float32.npy", b2_float)

metadata = {
    "architecture": "784-32-10",
    "quantization": "symmetric_signed_int8",
    "input_scale": input_scale,
    "fc1_weight_scale": w1_scale,
    "fc2_weight_scale": w2_scale,
    "fc1_weight_shape": list(w1_int8.shape),
    "fc2_weight_shape": list(w2_int8.shape),
    "bias_format": "float32_preserved_for_software_inference",
    "hardware_note": (
        "Current RTL does not yet implement bias addition or "
        "full-layer accumulation across 4-element reduction tiles."
    )
}

with open(OUTPUT_DIR / "quantization_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("\n===== QUANTIZATION COMPLETE =====")
print("FC1 weights:", tuple(w1_int8.shape))
print("FC2 weights:", tuple(w2_int8.shape))
print("Input scale:", input_scale)
print("FC1 weight scale:", w1_scale)
print("FC2 weight scale:", w2_scale)
print("FC1 INT8 range:", int(w1_int8.min()), int(w1_int8.max()))
print("FC2 INT8 range:", int(w2_int8.min()), int(w2_int8.max()))
print("Output directory:", OUTPUT_DIR)
