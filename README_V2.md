# v2 streaming-array upgrade (adds files only, overwrites nothing)

## What this adds
| Synopsis item | New file |
|---|---|
| 4x4 array, 16 MACs/cycle, weights held across tiles, no bubbles between tiles | `src/rtl/ws_pe_stream.v`, `src/rtl/systolic_array_ws_4x4.v` |
| Real weight memory (FC1 + FC2 ROMs), image buffer, tile-address generator, accumulators, INT8 output (hidden) buffer, argmax, perf counters, FSM | `src/rtl/mnist_accel_v2_top.v` |
| FC1 **and** FC2 tiled on the same array | same top file |
| ROM/image export + cross-check against your existing golden vectors | `src/python/export_streaming_roms.py` |
| Tests | `testbench/tb_systolic_array_ws_4x4.v`, `testbench/tb_mnist_accel_v2.v` |
| One-command run / synthesis | `scripts/run_v2_regression.sh`, `synthesis/synth_v2.ys` |
| Pre-simulation checks (Python only) | `tools/stream_model.py`, `tools/top_dataflow_check.py` |

`fc1_postprocess.v` (bias + ReLU + Q31 requant) is reused unchanged, so the numerics are identical to your verified design.

## Install and run
```bash
cd ~/Desktop/INT8_AI_Accelerator
unzip -o ~/Downloads/int8_v2_streaming_upgrade.zip      # adds files, existing ones untouched
bash scripts/run_v2_regression.sh 2                      # quick: 2 batches = 16 images
bash scripts/run_v2_regression.sh                        # full: 125 batches = 1000 images
```
Always run from the project root (the ROM files are opened with relative paths).

## How it works
* Batch = 8 images. For each group of 4 neurons and each 4-wide reduction tile, one 4x4 weight tile is
  placed on the weight bus and 8 activation vectors stream through, one per clock.
* A `first` token travels with the activations; each PE captures its weight when the token reaches it, so
  the next tile follows with no pipeline bubble (this needs >= 7 vectors per tile -> BATCH >= 7).
* Row k of the array sees its activation k cycles late (input skew); columns are re-aligned at the output
  (deskew). Results leave 7 clocks after the vector entered, one 4 x INT32 row per clock.
* Accumulators (BATCH x 4 x INT32) add the rows of all 196 tiles; then bias + ReLU + requant -> INT8 hidden buffer.
* FC2 reuses the array (weights padded 10 -> 12 neurons), logits = INT32 + bias, then serial argmax.

## Interface of `mnist_accel_v2_top`
* Load images while `busy=0`: `img_wr_en`, `img_wr_addr = image*196 + k_tile`, `img_wr_data = {p[4k+3],p[4k+2],p[4k+1],p[4k]}`.
* Pulse `start`; wait for `done`; read `predicted_class` (4 bits per image) and `batch_logits`.
* `perf_cycles`, `perf_mac_ops` = measured busy cycles and PE MACs of the last run.

## Things you may want to adjust
* `BATCH` parameter (must be >= 7; the testbench is written for 8). Other values also need the TB widths changed.
* ROM file names: parameters `FC1_W_FILE`, `FC2_W_FILE`, `FC1_B_FILE`, `FC2_B_FILE`.
* Different network: change the constants 196 / 195 / 8 groups / 3 groups in the top and re-export the ROMs.

## Known limitations
* Single image buffer: loading the next batch cannot overlap with compute (add a second bank to hide it).
* Weights and image buffer are synthesised as registers/logic by Yosys, so the generic cell count will be much
  larger than the old 83k. For physical design they would become SRAM/ROM macros.
* One idle gap per neuron group (drain + 8-cycle post-processing), so utilisation is slightly under 16.
* The Verilog has NOT been simulated by the author (no simulator was available). Python models checked the
  array timing and the top-level dataflow; the Verilog itself still needs your first run.
