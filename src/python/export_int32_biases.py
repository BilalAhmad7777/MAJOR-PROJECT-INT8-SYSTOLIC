import json
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
INT8_DIR = BASE_DIR / "src" / "python" / "models" / "int8"

# Load original floating-point biases.
b1 = np.load(INT8_DIR / "fc1_bias_float32.npy").astype(np.float64)
b2 = np.load(INT8_DIR / "fc2_bias_float32.npy").astype(np.float64)

# Load the exact scales used by the evaluation script.
with open(INT8_DIR / "quantization_metadata.json") as f:
    metadata = json.load(f)

input_scale = metadata["input_scale"]
w1_scale = metadata["fc1_weight_scale"]
w2_scale = metadata["fc2_weight_scale"]
hidden_scale = metadata["hidden_activation_scale"]

# Convert biases into INT32 accumulator units.
b1_int32 = np.rint(
    b1 / (input_scale * w1_scale)
).astype(np.int32)

b2_int32 = np.rint(
    b2 / (hidden_scale * w2_scale)
).astype(np.int32)

# Save the actual hardware bias arrays.
np.save(INT8_DIR / "fc1_bias_int32.npy", b1_int32)
np.save(INT8_DIR / "fc2_bias_int32.npy", b2_int32)

# Calculate the FC1 requantization multiplier.
fc1_multiplier = (
    input_scale * w1_scale
) / hidden_scale

print("===== INT32 BIAS EXPORT =====")
print("FC1 bias shape:", b1_int32.shape)
print("FC2 bias shape:", b2_int32.shape)
print("FC1 bias range:", b1_int32.min(), "to", b1_int32.max())
print("FC2 bias range:", b2_int32.min(), "to", b2_int32.max())
print("FC1 requantization multiplier:", fc1_multiplier)

print("\n[PASS] INT32 bias files exported.")

print("\nSaved files:")
print("fc1_bias_int32.npy")
print("fc2_bias_int32.npy")
