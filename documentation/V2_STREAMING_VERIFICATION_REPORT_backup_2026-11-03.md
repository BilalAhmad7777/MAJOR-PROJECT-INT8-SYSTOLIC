# INT8 AI Accelerator — V2 Streaming Verification Report

**Project:** INT8 MNIST inference accelerator
**Verification date:** 3 November 2026
**RTL simulation:** Icarus Verilog
**Generic synthesis:** Yosys
**Software reference:** Python / NumPy / PyTorch

## 1. Project Overview

V2 implements a streaming INT8 neural-network inference accelerator for MNIST.

The neural network is a fully connected model:

- Input: 784 features (28 × 28 image)
- FC1: 784 → 32
- ReLU and fixed-point requantization
- FC2: 32 → 10
- Output: 10 classification logits

Numeric representation:

- Signed INT8 activations and weights
- INT32 accumulation
- Fixed-point Q31 requantization
- ReLU activation and output clipping

## 2. Hardware Architecture

The V2 design includes:

- 4×4 weight-stationary streaming processing-element array
- Streaming PE and array control logic
- MNIST accelerator top-level
- FC1 post-processing
- Image and intermediate activation storage
- FC1 and FC2 weight and bias ROMs
- Batch processing and prediction generation
- Performance counters

The standalone streaming-array test verified a constant input-to-output latency of 8 clock edges.

The array test processed 12 tiles × 8 vectors, counted 1,536 MAC operations and reached 16 simultaneously active PEs.

## 3. Software Reference and ROM Generation

The Python flow provides:

- Trained MNIST model parameters
- INT8 weights and biases
- Quantized inference reference
- Expected INT32 logits
- Expected predictions
- Packed image and weight ROM data

The V2 ROM exporter is:

`src/python/export_streaming_roms.py`

The exporter checks:

1. FC1 weight tile consistency against existing weight data.
2. Python reimplementation against `expected_logits_int32.hex`.
3. Argmax predictions against `predictions_python.txt`.

Latest exporter result:

**ALL EXPORT CHECKS PASS**

The Python reference and exported verification data are therefore consistent with the checked golden vectors.

## 4. Reproducible Regression

Run the complete V2 regression from the project root:

```bash
bash scripts/run_v2_regression.sh
