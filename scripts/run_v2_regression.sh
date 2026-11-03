#!/usr/bin/env bash
# Streaming-array upgrade: export ROMs, compile, run array test + full MNIST test.
# Usage (from anywhere):  bash scripts/run_v2_regression.sh [num_batches]   (default 125 = 1000 images)
set -e
cd "$(dirname "$0")/.."
mkdir -p simulation

echo "== 1/3 export ROMs and cross-check against existing golden vectors =="
python3 src/python/export_streaming_roms.py

echo "== 2/3 streaming array unit test =="
iverilog -g2012 -s tb_systolic_array_ws_4x4 -o simulation/ws_stream_array.vvp \
    src/rtl/ws_pe_stream.v src/rtl/systolic_array_ws_4x4.v testbench/tb_systolic_array_ws_4x4.v
vvp simulation/ws_stream_array.vvp | tee simulation/ws_stream_array.log

echo "== 3/3 full accelerator vs Python golden model =="
iverilog -g2012 -s tb_mnist_accel_v2 -o simulation/mnist_accel_v2.vvp \
    src/rtl/ws_pe_stream.v src/rtl/systolic_array_ws_4x4.v src/rtl/fc1_postprocess.v \
    src/rtl/mnist_accel_v2_top.v testbench/tb_mnist_accel_v2.v
vvp simulation/mnist_accel_v2.vvp +BATCHES=${1:-125} | tee simulation/mnist_accel_v2.log
