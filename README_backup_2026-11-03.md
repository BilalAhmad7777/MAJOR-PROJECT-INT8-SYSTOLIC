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
- FC2 uses a sequential MAC architecture.
- Control logic includes IDLE, LOAD, COMPUTE and DONE states.

**Architecture clarification:** The current matrix engine is a sequential weight-stationary matrix engine. It is not a fully pipelined two-dimensional systolic wavefront with neighbour-to-neighbour PE forwarding. FC1 tile generation is external to the RTL, and the RTL does not accept raw MNIST pixels as a complete autonomous input pipeline.

## 2. Repository structure

```text
.
├── src/
│   ├── rtl/                 # Verilog RTL modules
│   └── python/              # Training, quantization and reference scripts
│       └── models/          # Trained model and INT8 parameters
├── testbench/               # Verilog testbenches
├── verification/
│   └── test_vectors/        # FC1, FC2 and MNIST verification vectors
├── simulation/              # Simulation executables and outputs
├── synthesis/
│   ├── logs/                 # Yosys synthesis logs
│   └── results/              # Synthesized netlists and reports
├── documentation/
│   ├── FINAL_VERIFICATION_REPORT.md
│   └── checkpoints/          # Verification and development checkpoints
├── reference-asic/           # ASIC reference material
├── reference-fpga/           # FPGA reference material
└── README.md
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

## 4. Running RTL verification

Compile each testbench from the current RTL sources and run it with Icarus Verilog.

### FC1 controller protocol

```bash
iverilog -g2012 -s tb_fc1_controller_protocol -o /tmp/fc1_protocol.vvp src/rtl/*.v testbench/tb_fc1_controller_protocol.v
vvp /tmp/fc1_protocol.vvp
```

### FC1 exhaustive tile verification

```bash
iverilog -g2012 -s tb_fc1_all_tiles -o /tmp/fc1_all_tiles.vvp src/rtl/*.v testbench/tb_fc1_all_tiles.v
vvp /tmp/fc1_all_tiles.vvp
```

### FC2 classifier

```bash
iverilog -g2012 -s tb_fc2_classifier -o /tmp/fc2_classifier.vvp src/rtl/*.v testbench/tb_fc2_classifier.v
vvp /tmp/fc2_classifier.vvp
```

### Integrated MNIST accelerator top

```bash
iverilog -g2012 -s tb_mnist_accelerator_top -o /tmp/mnist_top.vvp src/rtl/*.v testbench/tb_mnist_accelerator_top.v
vvp /tmp/mnist_top.vvp
```

### 1,000-image RTL verification

```bash
iverilog -g2012 -s tb_mnist_1000_images -o /tmp/mnist_1000.vvp src/rtl/*.v testbench/tb_mnist_1000_images.v
vvp /tmp/mnist_1000.vvp
```

The 1,000-image testbench uses vectors in `verification/test_vectors/mnist_1000/`.

## 5. Running Python reference checks

```bash
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
| 1,000-image RTL regression | 11,250 passed, 0 failed |
| Q31 fixed-point comparison | 320,000 values, 0 mismatches |

### MNIST evaluation

- Python INT8 reference accuracy: **95.42%** (9,542 / 10,000).
- RTL evaluation accuracy: **94.80%** (948 / 1,000).
- In the 1,000-image RTL regression, all 10,000 logits and all 1,000 predictions matched the Python reference.
- The 94.80% figure is classification accuracy against MNIST labels, not the RTL-to-Python equivalence rate.

### Generic synthesis

The final Yosys generic synthesis reported **83,389 generic logic cells** for `mnist_accelerator_top`, with zero problems reported by the Yosys consistency checks.

This is a generic synthesis cell count. It is not a Sky130 cell-area result and must not be interpreted as physical area, power, frequency or timing.

For full test details, see `documentation/FINAL_VERIFICATION_REPORT.md`.

## 7. Current limitations and future work

1. The 4×4 engine is weight-stationary but is not a fully pipelined systolic wavefront.
2. FC1 tile generation and raw image preprocessing are external to the RTL.
3. FC2 is implemented as a sequential MAC architecture.
4. The RTL does not provide a complete raw-pixel-to-classification input pipeline.
5. FC1 accumulation does not currently provide general overflow detection and saturation.
6. OpenLane and SkyWater 130nm physical implementation have not been completed.
7. No physical area, power, clock frequency, critical-path timing or DRC results are claimed.

Potential next steps include completing the raw-input pipeline, improving the systolic dataflow, adding overflow handling, and running OpenLane/SkyWater 130nm physical implementation and PPA analysis.

## 8. Detailed documentation

See `documentation/FINAL_VERIFICATION_REPORT.md` for the detailed verification record, test results, synthesis evidence, checkpoint inventory and known limitations.
