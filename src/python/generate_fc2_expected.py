import numpy as np
from pathlib import Path

ROOT = Path("src/python/models/int8")
OUT = Path("verification/test_vectors/fc2_hardware")
FC1_OUT = Path("verification/test_vectors/fc1_hardware")

OUT.mkdir(parents=True, exist_ok=True)

# Read FC1 INT8 hexadecimal outputs
with open(FC1_OUT / "expected_hidden_int8.hex") as f:
    raw_hidden = [int(line.strip(), 16) for line in f if line.strip()]

hidden = np.array(
    [v if v < 128 else v - 256 for v in raw_hidden],
    dtype=np.int8
).reshape(4, 32)

# Load FC2 parameters
weights = np.load(ROOT / "fc2_weights_int8.npy").astype(np.int64)
biases = np.load(ROOT / "fc2_bias_int32.npy").astype(np.int64)

# Calculate INT32 logits
logits = hidden.astype(np.int64) @ weights.T + biases.reshape(1, 10)
predictions = np.argmax(logits, axis=1)

# Save hidden activations
with open(OUT / "fc2_hidden_int8.hex", "w") as f:
    for v in hidden.reshape(-1):
        f.write(f"{int(v) & 0xFF:02x}\n")

# Save weights
with open(OUT / "fc2_weights_int8.hex", "w") as f:
    for v in weights.reshape(-1):
        f.write(f"{int(v) & 0xFF:02x}\n")

# Save biases
with open(OUT / "fc2_bias_int32.hex", "w") as f:
    for v in biases:
        f.write(f"{int(v) & 0xFFFFFFFF:08x}\n")

# Save expected logits
with open(OUT / "fc2_expected_logits_int32.hex", "w") as f:
    for v in logits.reshape(-1):
        f.write(f"{int(v) & 0xFFFFFFFF:08x}\n")

print("===== FC2 GOLDEN MODEL =====")
print("Hidden shape:", hidden.shape)
print("Weight shape:", weights.shape)
print("Bias shape:", biases.shape)
print("Logit shape:", logits.shape)
print("\nPredictions:", predictions)

for i, row in enumerate(logits):
    print(f"Image {i}: {row.tolist()} | Prediction: {predictions[i]}")

print("\nFiles saved in:", OUT)
