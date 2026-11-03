#!/usr/bin/env python3
"""Behavioural check of mnist_accel_v2_top's DATAFLOW (index math, ROM layout, FC2 padding,
accumulation order, post-processing, argmax) using the exported ROM files exactly as the RTL
reads them.  Array timing itself is covered by stream_model.py / the array testbench.
Run from the project root after export_streaming_roms.py."""
import numpy as np
from pathlib import Path
R = Path("rom"); V = Path("verification/test_vectors/mnist_1000")
B = 8
def rd(f): return [int(x, 16) for x in (R/f).read_text().split()]
def s8(v): return v - 256 if v >= 128 else v
def s32(v): return v - 2**32 if v >= 2**31 else v
fc1_w = rd("fc1_weights_tiles.hex"); fc2_w = rd("fc2_weights_tiles.hex")
b1 = [s32(x) for x in rd("fc1_bias.hex")]; b2 = [s32(x) for x in rd("fc2_bias.hex")]
words = rd("images_1000_words.hex")
MULT = 638865
def post(acc, bias):
    v = max(acc + bias, 0) * MULT
    q, r = v >> 31, v & 0x7FFFFFFF
    if r > (1 << 30) or (r == (1 << 30) and (q & 1)): q += 1
    return min(max(q, 0), 127)
def tile_bus(rom, idx):          # bus byte (k*4+c)
    return [[s8((rom[idx] >> (8*(k*4+c))) & 0xFF) for c in range(4)] for k in range(4)]
exp = [s32(int(l, 16)) for l in (V/"expected_logits_int32.hex").read_text().split()]
pred_exp = [int(x) for x in (V/"predictions_python.txt").read_text().split()]
bad = 0
for batch in range(0, 4):            # 4 batches = 32 images (pure python, keep it quick)
    img_mem = words[batch*B*196:(batch+1)*B*196]
    hidden = [0]*(B*32)
    for grp in range(8):
        acc = [[0]*4 for _ in range(B)]
        for kt in range(196):
            W = tile_bus(fc1_w, grp*196+kt)
            for i in range(B):
                wd = img_mem[i*196+kt]
                a = [s8((wd >> (8*k)) & 0xFF) for k in range(4)]
                for c in range(4):
                    acc[i][c] += sum(a[k]*W[k][c] for k in range(4))
        for i in range(B):
            for c in range(4):
                hidden[i*32+grp*4+c] = post(acc[i][c], b1[grp*4+c])
    logits = [[0]*10 for _ in range(B)]
    for grp in range(3):
        acc = [[0]*4 for _ in range(B)]
        for kt in range(8):
            W = tile_bus(fc2_w, grp*8+kt)
            for i in range(B):
                a = [hidden[i*32+kt*4+k] for k in range(4)]
                for c in range(4):
                    acc[i][c] += sum(a[k]*W[k][c] for k in range(4))
        for i in range(B):
            for c in range(4):
                n = grp*4+c
                if n < 10: logits[i][n] = acc[i][c] + b2[n]
    for i in range(B):
        g = batch*B+i
        if logits[i] != exp[g*10:g*10+10]: bad += 1; print("logit mismatch image", g)
        best = max(range(10), key=lambda n: (logits[i][n], -n))
        if best != pred_exp[g]: bad += 1; print("pred mismatch image", g)
print("DATAFLOW CHECK:", "PASS (32 images bit-exact)" if bad == 0 else "FAIL %d" % bad)
