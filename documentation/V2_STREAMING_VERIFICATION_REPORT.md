# INT8 AI Accelerator — V2 Streaming Verification Report

**Project:** INT8 MNIST inference accelerator
**Verification date:** 3 November 2026
**RTL simulation:** Icarus Verilog
**Generic synthesis:** Yosys
**Software reference:** Python / NumPy / PyTorch

## 1. Project Overview

V2 implements a streaming INT8 neural-network inference accelerator for MNIST.

The network is a fully connected model:

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

FC1 and FC2 reuse the same 4×4 processing-element array. FC1 tile generation and image/weight preparation are handled by the Python flow rather than an autonomous raw-pixel preprocessing pipeline in RTL.

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

The Python reference and exported verification data are consistent with the checked golden vectors.

## 4. Reproducible Regression

Run the complete V2 regression from the project root:

`bash scripts/run_v2_regression.sh`

This script runs the ROM exporter checks, the 4x4 streaming array testbench, and the full accelerator-versus-Python golden-model regression.

### Latest Full Regression Results

The latest recorded full V2 regression completed with:

- **Total checks:** 11,001
- **Passed:** 11,001
- **Failed:** 0
- **Images tested:** 1,000
- **Logits matched against Python:** 10,000
- **Predictions matched against Python:** 1,000
- **MNIST label accuracy:** 948/1,000 (94.8%)

All accelerator logits and predictions matched the Python golden model exactly.

### Execution and Throughput

The full regression used 125 batches of 8 images each.

- **Cycles per batch:** 13,015
- **Average cycles per image:** approximately 1,626
- **MAC operations per batch:** 203,776
- **Average throughput:** 15.66 MACs/cycle
- **Peak throughput:** 16 MACs/cycle

The measured average throughput is close to the 16-MAC/cycle peak of the 4x4 processing-element array.

## 5. Generic Logic Synthesis

The V2 RTL was processed using Yosys generic synthesis.

- **Generic mapped cells:** 258,835
- **Undriven wires:** 0
- **Yosys check:** 0 problems reported

These are generic synthesis results, not physical area, power, timing, or post-layout PPA measurements. No physical-design implementation or technology-specific signoff is claimed.

## 6. Known Warnings and Limitations

### FC2 Weight ROM Depth Warning

Icarus Verilog reports that `fc2_weights_tiles.hex` contains 24 entries while the corresponding memory is declared with indices `[0:31]`.

The current design uses the 24 populated entries. This warning is non-fatal, and the verified regression completes successfully.

### Functional Scope

- The RTL does not autonomously perform raw MNIST image preprocessing.
- FC1 tile generation and input/weight preparation are handled by the Python flow.
- The processing-element array uses a sequential weight-stationary execution schedule; it should not be described as a fully pipelined 2D systolic wavefront.

## 7. Conclusion

The V2 INT8 MNIST accelerator has passed its recorded full regression against the Python golden model.

The verification demonstrates bit-exact agreement for all 10,000 tested logits and 1,000 predictions, with 11,001 checks passing and zero failures.

The implementation provides a verified 4x4 weight-stationary processing-element array, INT8 inference datapath, and reproducible Python-to-RTL validation flow.

The results in this report describe the verified functional and generic synthesis state of the project. They do not claim physical-design completion or silicon-level performance.
