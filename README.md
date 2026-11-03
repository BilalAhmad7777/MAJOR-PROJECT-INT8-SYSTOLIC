# INT8 AI Accelerator — 4×4 Weight-Stationary Matrix Engine

A Verilog-based INT8 neural-network accelerator project for edge inference. This project implements a 4×4 weight-stationary matrix engine and an MNIST fully connected inference pipeline, supported by Python reference models and Verilog testbenches.

**Verification status — 3 November 2026:** Functional RTL verification and generic Yosys synthesis are complete. OpenLane/Sky130 physical design and physical PPA measurements have not been performed.

## 1. Architecture

The neural network is a fully connected MNIST model:

- Input: 784 values (28×28 image)
- FC1: 784 → 32
- ReLU and requantization
- FC2: 32 → 10
- Output: 10 classification logits

### Numeric representation

- Signed INT8 activations and weights
- Signed INT32 accumulation
- INT8 post-processing output
- Fixed-point Q31 requantization in the hardware verification flow

### Hardware organization

- 4×4 weight-stationary processing-element matrix
- Weights are held across reduction cycles.
- Each matrix operation computes 16 output values over four reduction cycles.
- FC1 uses externally generated input and weight tiles.
- V2 uses the streaming 4×4 weight-stationary PE array for tiled inference.
- Control logic includes IDLE, LOAD, COMPUTE and DONE states.

**Architecture clarification:** The current matrix engine is a sequential weight-stationary matrix engine. It is not a fully pipelined two-dimensional systolic wavefront with neighbour-to-neighbour PE forwarding. FC1 tile generation is external to the RTL, and the RTL does not accept raw MNIST pixels as a complete autonomous input pipeline.

## 2. Repository structure

```text
INT8_AI_Accelerator/
├── data/
├── documentation/
├── physical_design/
├── reference-asic/
├── reference-fpga/
├── reports/
├── results/
├── rom/
├── scripts/
├── simulation/
├── src/
├── synthesis/
├── testbench/
├── tools/
├── verification/
├── README.md
└── README_V2.md
```

## 3. Environment

Verified development environment:

| Tool / package | Version |
|---|---:|
| Python | 3.12.16 |
| PyTorch | 2.15.0 |
| torchvision | 0.29.2 |
| NumPy | 2.5.4 |
| Icarus Verilog | 13.0 |
| Yosys | 0.70+post |

There is currently no `requirements.txt` or other Python dependency manifest in the repository.

For a fresh Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.15.0 torchvision==0.29.2 numpy==2.5.4
```

The commands below assume `.venv` is located at the project root. Icarus Verilog and Yosys must be installed separately.

## 4. Legacy V1 RTL verification (historical)

The following commands reproduce the earlier V1 verification flow. For the current V2 implementation, use the V2 regression command in Section 4A.

Compile each testbench from the current RTL sources and run it with Icarus Verilog.

### FC1 controller protocol

```bash
iverilog -g2012 -s tb_fc1_controller_protocol -o /tmp/fc1_protocol.vvp src/rtl/accelerator_top.v src/rtl/fc1_controller.v src/rtl/fc1_postprocess.v src/rtl/fc1_tile_accumulator.v src/rtl/fc2_classifier.v src/rtl/mnist_accelerator_top.v src/rtl/relu.v src/rtl/requantize.v src/rtl/systolic_array_4x4.v src/rtl/weight_stationary_array_4x4.v src/rtl/weight_stationary_pe.v src/rtl/weight_stationary_array_4x4_fsm.v testbench/tb_fc1_controller_protocol.v
vvp /tmp/fc1_protocol.vvp
```

### FC1 exhaustive tile verification

```bash
iverilog -g2012 -s tb_fc1_all_tiles -o /tmp/fc1_all_tiles.vvp src/rtl/accelerator_top.v src/rtl/fc1_controller.v src/rtl/fc1_postprocess.v src/rtl/fc1_tile_accumulator.v src/rtl/fc2_classifier.v src/rtl/mnist_accelerator_top.v src/rtl/relu.v src/rtl/requantize.v src/rtl/systolic_array_4x4.v src/rtl/weight_stationary_array_4x4.v src/rtl/weight_stationary_pe.v src/rtl/weight_stationary_array_4x4_fsm.v testbench/tb_fc1_all_tiles.v
vvp /tmp/fc1_all_tiles.vvp
```

### FC2 classifier

```bash
iverilog -g2012 -s tb_fc2_classifier -o /tmp/fc2_classifier.vvp src/rtl/accelerator_top.v src/rtl/fc1_controller.v src/rtl/fc1_postprocess.v src/rtl/fc1_tile_accumulator.v src/rtl/fc2_classifier.v src/rtl/mnist_accelerator_top.v src/rtl/relu.v src/rtl/requantize.v src/rtl/systolic_array_4x4.v src/rtl/weight_stationary_array_4x4.v src/rtl/weight_stationary_pe.v src/rtl/weight_stationary_array_4x4_fsm.v testbench/tb_fc2_classifier.v
vvp /tmp/fc2_classifier.vvp
```

### Integrated MNIST accelerator top

```bash
iverilog -g2012 -s tb_mnist_accelerator_top -o /tmp/mnist_top.vvp src/rtl/accelerator_top.v src/rtl/fc1_controller.v src/rtl/fc1_postprocess.v src/rtl/fc1_tile_accumulator.v src/rtl/fc2_classifier.v src/rtl/mnist_accelerator_top.v src/rtl/relu.v src/rtl/requantize.v src/rtl/systolic_array_4x4.v src/rtl/weight_stationary_array_4x4.v src/rtl/weight_stationary_pe.v src/rtl/weight_stationary_array_4x4_fsm.v testbench/tb_mnist_accelerator_top.v
vvp /tmp/mnist_top.vvp
```

### 1,000-image RTL verification

```bash
iverilog -g2012 -s tb_mnist_1000_images -o /tmp/mnist_1000.vvp src/rtl/accelerator_top.v src/rtl/fc1_controller.v src/rtl/fc1_postprocess.v src/rtl/fc1_tile_accumulator.v src/rtl/fc2_classifier.v src/rtl/mnist_accelerator_top.v src/rtl/relu.v src/rtl/requantize.v src/rtl/systolic_array_4x4.v src/rtl/weight_stationary_array_4x4.v src/rtl/weight_stationary_pe.v src/rtl/weight_stationary_array_4x4_fsm.v testbench/tb_mnist_1000_images.v
vvp /tmp/mnist_1000.vvp
```

The 1,000-image testbench uses vectors in `verification/test_vectors/mnist_1000/`.

### Rebuilding the MNIST model and hardware vectors

Run these commands from the project root, in order:

```bash
.venv/bin/python src/python/train_mnist.py
.venv/bin/python src/python/quantize_model.py
.venv/bin/python src/python/export_int32_biases.py
.venv/bin/python src/python/generate_fc1_expected.py
.venv/bin/python src/python/generate_fc2_expected.py
.venv/bin/python src/python/generate_100_image_vectors.py
```

The FC1 and FC2 expected-output scripts generate the smaller four-image reference checks. The 1,000-image generator independently computes both network layers and creates the separate RTL regression dataset.

Warning: These commands overwrite generated artifacts. Back up verified files before rebuilding.

## 4A. Running the V2 regression

Run the complete V2 verification flow from the project root:

```bash
bash scripts/run_v2_regression.sh
```

This runs the ROM exporter checks, the 4x4 streaming array test, and the integrated MNIST accelerator comparison against the Python golden model.

## 5. Running Python reference checks

```bash
# Legacy 4x4 INT8 arithmetic self-test (not the MNIST reference)
.venv/bin/python src/python/golden_model.py
.venv/bin/python src/python/evaluate_int8.py
.venv/bin/python src/python/verify_fixed_point.py
.venv/bin/python src/python/tiled_inference.py
```

Some scripts load or download the MNIST dataset. The project includes a local MNIST dataset under `data/`.

`evaluate_int8.py` evaluates the INT8 software model and does not modify quantization metadata. Other scripts may generate or export model parameters and verification vectors; review their output paths before rerunning them if you want to preserve existing artifacts.

## 6. Verified results

The following results were reproduced or checked against the current project sources on 3 November 2026.

| Verification | Result |
|---|---:|
| FC1 controller protocol | 4 passed, 0 failed |
| FC1 exhaustive tile outputs | 25,088 passed, 0 failed |
| FC2 classifier | 40 passed, 0 failed |
| Integrated MNIST top | 45 passed, 0 failed |
| 1,000-image RTL regression | 11,001 passed, 0 failed |
| Q31 fixed-point comparison | 320,000 values, 0 mismatches |

### MNIST evaluation

- Python INT8 reference accuracy: **95.42%** (9,542 / 10,000).
- V2 RTL classification accuracy: **94.80%** (948 / 1,000).
- In the 1,000-image RTL regression, all 10,000 logits and all 1,000 predictions matched the Python reference.
- The 94.80% figure is classification accuracy against MNIST labels, not the RTL-to-Python equivalence rate.

### Generic synthesis

The final Yosys generic synthesis reported **258,835 generic mapped cells** for `mnist_accelerator_top`, with zero problems reported by the Yosys consistency checks.

This is a generic synthesis cell count. It is not a Sky130 cell-area result and must not be interpreted as physical area, power, frequency or timing.

For current V2 test details, see `documentation/V2_STREAMING_VERIFICATION_REPORT.md`. Historical V1 verification details are available in `documentation/FINAL_VERIFICATION_REPORT.md`.

## 7. Current limitations and future work

1. Raw MNIST image preprocessing and input preparation are handled by the Python flow and exported verification data.
2. The current implementation uses pre-generated ROM data for weights, biases and test images.
3. The FC2 ROM contains 24 entries while its declared memory range is 32 entries, producing a non-fatal Icarus `$readmemh` warning during simulation.
4. Generic Yosys synthesis has been completed, but physical implementation has not.
5. OpenLane and SkyWater 130nm physical design have not been performed.
6. No physical area, power, clock frequency, critical-path timing or DRC results are claimed.

Potential future work includes integrating raw-image preprocessing, improving ROM configurability, adding broader overflow handling, and eventually performing physical implementation and PPA analysis.

## 8. Detailed documentation

- `documentation/V2_STREAMING_VERIFICATION_REPORT.md` — Current V2 architecture, regression results, performance and synthesis.
- `documentation/FINAL_VERIFICATION_REPORT.md` — Historical V1 verification record and development history.
