#!/usr/bin/env python3
"""Export ROM / image files for mnist_accel_v2_top and cross-check them against
the existing golden vectors.  Run from the project root:
    python3 src/python/export_streaming_roms.py
Needs only numpy."""
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
M = BASE / "src/python/models/int8"
V = BASE / "verification/test_vectors/mnist_1000"
OUT = BASE / "rom"
OUT.mkdir(exist_ok=True)

MULT = 638865                      # same constant as fc1_postprocess.v

w1 = np.load(M / "fc1_weights_int8.npy").astype(np.int64)    # [32,784]
w2 = np.load(M / "fc2_weights_int8.npy").astype(np.int64)    # [10,32]
b1 = np.load(M / "fc1_bias_int32.npy").astype(np.int64)      # [32]
b2 = np.load(M / "fc2_bias_int32.npy").astype(np.int64)      # [10]
imgs = np.load(V / "input_images_int8.npy").astype(np.int64) # [1000,784]

def requant(acc):                  # bias already added; ReLU, Q31 mult, RNE, clip 0..127
    a = np.maximum(acc, 0).astype(object)
    out = np.zeros(acc.shape, dtype=np.int64)
    it = np.nditer(acc, flags=["multi_index"])
    for _ in it:
        v = max(int(acc[it.multi_index]), 0) * MULT
        q, r = v >> 31, v & 0x7FFFFFFF
        half = 1 << 30
        if r > half or (r == half and (q & 1)):
            q += 1
        out[it.multi_index] = min(max(q, 0), 127)
    return out

def golden(x):
    h = requant(x @ w1.T + b1)
    return h @ w2.T + b2

# ---- ROM files -------------------------------------------------------------
def tile_line(t):                  # t[k][c] -> 128-bit, byte (k*4+c) at bits [8*(k*4+c)+:8]
    v = 0
    for k in range(4):
        for c in range(4):
            v |= (int(t[k][c]) & 0xFF) << (8 * (k * 4 + c))
    return "%032x" % v

fc1_lines = []
for g in range(8):
    for kt in range(196):
        t = [[w1[g*4+c][kt*4+k] for c in range(4)] for k in range(4)]
        fc1_lines.append(tile_line(t))
fc2_lines = []
for g in range(3):
    for kt in range(8):
        t = [[(w2[g*4+c][kt*4+k] if g*4+c < 10 else 0) for c in range(4)] for k in range(4)]
        fc2_lines.append(tile_line(t))

(OUT / "fc1_weights_tiles.hex").write_text("\n".join(fc1_lines) + "\n")
(OUT / "fc2_weights_tiles.hex").write_text("\n".join(fc2_lines) + "\n")
(OUT / "fc1_bias.hex").write_text("\n".join("%08x" % (int(b) & 0xFFFFFFFF) for b in b1) + "\n")
(OUT / "fc2_bias.hex").write_text("\n".join("%08x" % (int(b) & 0xFFFFFFFF) for b in b2) + "\n")

# image words: image-major, 196 words per image, byte k = pixel kt*4+k
words = []
for im in imgs:
    for kt in range(196):
        v = 0
        for k in range(4):
            v |= (int(im[kt*4+k]) & 0xFF) << (8 * k)
        words.append("%08x" % v)
(OUT / "images_1000_words.hex").write_text("\n".join(words) + "\n")

# ---- cross-checks against the existing project vectors ----------------------
ok = True
old = (BASE / "verification/test_vectors/fc1_hardware/weights.hex")
if old.exists():
    ol = old.read_text().split()[:1568]
    # old files store the 16 weights in the same tile order; compare as byte multisets per line
    same = all(sorted(a[i:i+2] for i in range(0, 32, 2)) == sorted(b[i:i+2] for i in range(0, 32, 2))
               for a, b in zip(ol, fc1_lines))
    print("FC1 tile ROM matches existing weights.hex tiles (byte-multiset check):", same)
    ok &= same

g = golden(imgs)
exp = np.array([int(l, 16) for l in (V / "expected_logits_int32.hex").read_text().split()], dtype=np.int64)
exp = np.where(exp >= 2**31, exp - 2**32, exp).reshape(-1, 10)
m = np.array_equal(g[:len(exp)], exp)
print("Python re-implementation == expected_logits_int32.hex:", m)
ok &= m
pred = [int(x) for x in (V / "predictions_python.txt").read_text().split()]
p = list(g.argmax(axis=1))
print("argmax == predictions_python.txt:", p[:len(pred)] == pred)
ok &= (p[:len(pred)] == pred)
print("ALL EXPORT CHECKS PASS" if ok else "EXPORT CHECK FAILED - do not simulate yet")
