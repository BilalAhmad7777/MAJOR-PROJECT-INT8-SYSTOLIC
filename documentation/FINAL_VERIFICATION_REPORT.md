# INT8 AI Accelerator — Final Verification Report

**Project:** 4×4 Weight-Stationary TPU-Style Accelerator for Edge Inference
**Verification date:** 3 November 2026
**HDL simulation:** Icarus Verilog
**Synthesis:** Yosys
**Software reference:** Python / PyTorch

---

## 1. Project Overview

This project implements an INT8 neural-network inference accelerator using Verilog RTL.

The target neural-network architecture is:

- Input layer: 784 features
- Hidden layer: 32 neurons
- Output layer: 10 neurons
- Network architecture: 784-32-10
- Arithmetic: signed INT8 operands and INT32 accumulation
- Activation: ReLU
- Output processing: fixed-point requantization

The design is verified against a Python reference implementation using MNIST data.

## 2. Hardware Architecture

### 2.1 Weight-Stationary 4×4 Matrix Engine

The matrix engine contains:

- 4×4 processing elements
- Signed INT8 activations
- Signed INT8 weights
- Signed INT32 accumulators
- Four reduction cycles per 4×4 matrix operation
- Weight storage during the reduction operation

The engine computes 16 output values for each 4×4 matrix operation.

**Architecture clarification:** The current implementation is a sequential weight-stationary matrix engine. It is not a fully pipelined two-dimensional systolic wavefront with continuous neighbour-to-neighbour data propagation.

### 2.2 Control and Post-Processing

The design includes:

- Matrix-engine FSM
- Weight-stationary processing element
- FC1 tile accumulator
- FC1 controller
- ReLU
- Fixed-point requantization
- FC1 post-processing
- Sequential FC2 classifier
- Integrated MNIST accelerator top-level controller

The FC2 classifier performs sequential multiply-accumulate operations across 32 hidden activations for each output neuron.

### 2.3 Full Inference Data Flow

1. An external producer supplies FC1 activation and weight tiles.
2. The FC1 controller coordinates matrix operations.
3. The FC1 tile accumulator combines partial results.
4. FC1 post-processing applies bias, ReLU and requantization.
5. The resulting 32 hidden activations are passed to FC2.
6. FC2 computes the 10 output logits.
7. The top-level controller stores the logits for each image.

**Integration boundary:** The current RTL top-level does not accept raw 28×28 MNIST pixels or generate all FC1 tiles internally. Input preprocessing and FC1 tile generation are handled outside the RTL.

## 3. Verification Environment

| Component | Tool |
|---|---|
| RTL simulation | Icarus Verilog 13.0 |
| RTL synthesis | Yosys 0.70+post |
| Reference model | Python |
| Neural-network framework | PyTorch |
| Hardware description | Verilog/SystemVerilog-compatible RTL |

Simulation and verification are performed against generated Python reference vectors.

## 4. Module-Level Verification

The following recorded test results were obtained:

| Module / Test | Passed | Failed |
|---|---:|---:|
| Baseline 4×4 array | 432 | 0 |
| Weight-stationary PE | 5 | 0 |
| Weight-stationary array | 368 | 0 |
| Weight-stationary identity test | 16 | 0 |
| Weight-stationary control | 21 | 0 |
| Matrix-engine FSM | 222 | 0 |
| Matrix-engine FSM control | 16 | 0 |
| ReLU | 8 | 0 |
| Requantization | 15 | 0 |
| Requantization boundary audit | 28 | 0 |
| Accelerator tile top | 368 | 0 |
| FC1 tile accumulator | 17 | 0 |
| FC1 accumulator stress audit | 36 | 0 |
| FC1 accumulator extended audit | 66 | 0 |
| FC1 accumulator overflow audit | 48 | 0 |
| FC1 post-processing | 8 | 0 |
| FC1 post-processing boundary audit | 17 | 0 |
| FC1 MNIST post-processing | 264 | 0 |
| FC1 controller | 131 | 0 |
| FC1 valid/ready protocol audit | 4 | 0 |
| FC1 exhaustive matrix RTL/Python audit | 25088 | 0 |
| FC1 reset recovery | 132 | 0 |
| FC1 repeated inference | 260 | 0 |
| FC2 classifier | 40 | 0 |
| FC2 arithmetic stress audit | 40 | 0 |
| End-to-end integration testbench | 172 | 0 |
| FC2 reset recovery | 13 | 0 |
| FC2 repeated inference | 24 | 0 |
| Top-level functional test | 45 | 0 |
| Top-level reset recovery | 51 | 0 |
| Top-level repeated inference | 90 | 0 |

These are individual testbench results. They should not be interpreted as independent hardware test scenarios or as a single consolidated regression count.

## 5. MNIST Verification

### 5.1 Python Reference Evaluation

The INT8 Python reference model was evaluated on the full MNIST test set.

| Metric | Result |
|---|---:|
| Total images | 10,000 |
| Correct predictions | 9,542 |
| Incorrect predictions | 458 |
| Classification accuracy | 95.42% |

### 5.2 Earlier 100-Image RTL Verification

This earlier RTL verification run used the first 100 MNIST test images. It is reported separately from the later 1,000-image regression.

| Metric | Result |
|---|---:|
| Images tested | 100 |
| Exact RTL/Python logit comparisons | 1,000 / 1,000 |
| RTL predictions matching Python | 100 / 100 |
| Correct classifications | 98 / 100 |
| Classification accuracy | 98.00% |
| Testbench checks | 1,125 PASS |
| Failed checks | 0 |

### 5.3 1,000-Image RTL Verification

The RTL accelerator was tested against 1,000 MNIST images over 250 batches.

| Metric | Result |
|---|---:|
| Images tested | 1,000 |
| FC1 tile transactions | 392,000 |
| Exact logit comparisons | 10,000 / 10,000 |
| RTL predictions matching Python | 1,000 / 1,000 |
| Correct classifications | 948 / 1,000 |
| Classification accuracy | 94.80% |
| Testbench checks | 11,250 PASS |
| Failed checks | 0 |

**Interpretation:** The RTL produced logits and predictions matching the Python INT8 reference for all 1,000 images. The 94.80% figure is classification accuracy against the MNIST labels, not hardware equivalence accuracy.

### 5.4 Fresh-Source Regression Rerun — 3 November 2026

The following testbenches were freshly compiled from the current RTL sources using Icarus Verilog and executed successfully.

| Regression | Verified checks | Result |
|---|---:|---|
| FC1 valid/ready protocol | 4 | 4 PASS, 0 FAIL |
| FC1 exhaustive tile verification | 25,088 outputs across 1,568 tiles | 25,088 PASS, 0 FAIL |
| FC2 classifier | 40 logits | 40 PASS, 0 FAIL |
| Integrated MNIST accelerator top | 45 | 45 PASS, 0 FAIL |
| 1,000-image RTL verification | 11,250 | 11,250 PASS, 0 FAIL |

The fresh 1,000-image run processed 1,000 images across 250 batches, with 392,000 FC1 tile transactions, 10,000 exact logit comparisons, and 1,000 prediction comparisons. All RTL logits and predictions matched the Python reference. Classification accuracy was 94.80%.

The Python INT8 evaluation was also rerun: 9,542 of 10,000 test images were classified correctly (95.42%). The evaluation completed without modifying quantization metadata.

## 6. Reset and Repeated-Inference Verification

### 6.1 FC1 Reset Recovery

The FC1 controller was interrupted during an inference, reset, and then used for a fresh inference.

Result:

- Busy cleared after reset.
- Done cleared after reset.
- Hidden outputs cleared.
- A fresh inference completed.
- All 128 hidden outputs matched Python.

**Result: 132 PASS / 0 FAIL.**

### 6.2 FC1 Repeated Inference

Two complete FC1 inference batches were executed without resetting the controller between them.

- 256 hidden-output comparisons.
- Both inference batches completed.
- Controller returned to idle.

**Result: 260 PASS / 0 FAIL.**

### 6.3 FC2 Reset Recovery

FC2 was interrupted during an inference, reset, and used to recompute the image.

- Reset cleared the output state.
- All 10 logits matched Python.

**Result: 13 PASS / 0 FAIL.**

### 6.4 FC2 Repeated Inference

Two consecutive FC2 inferences were performed without an intervening reset.

- 20 logit comparisons.
- Both inferences completed successfully.

**Result: 24 PASS / 0 FAIL.**

### 6.5 Integrated Top-Level Reset Recovery

The integrated MNIST accelerator was reset and tested for a fresh inference.

- All four predictions matched Python.
- Accelerator returned to idle.
- Reset recovery checks passed.

**Result: 51 PASS / 0 FAIL.**

### 6.6 Integrated Top-Level Repeated Inference

Two consecutive four-image inference batches were executed without resetting the accelerator between batches.

- 8 image predictions matched Python.
- 80 logits matched Python exactly.
- Both batches completed.
- The testbench reported 90 PASS / 0 FAIL.

This verifies repeated inference at the integrated top level for the tested sequence.

### 6.7 Final Integrated Top-Level Regression

The three integrated top-level testbenches were rerun as part of the final non-physical verification audit.

| Test | Passed | Failed |
|---|---:|---:|
| Normal top-level inference | 45 | 0 |
| Top-level reset recovery | 51 | 0 |
| Two-batch repeated inference | 90 | 0 |
| **Combined testbench checks** | **186** | **0** |

All three testbenches completed successfully. The normal inference test verified four predictions; the reset test verified recovery and return to idle; and the repeated-inference test verified two consecutive batches.

These are aggregated testbench checks, not 186 independent hardware scenarios.

## 7. Synthesis Results

The integrated top-level design was synthesized using Yosys.

Synthesis flow included:

- Verilog reading
- Hierarchy checking
- Process lowering
- Optimization
- Logic synthesis
- Final consistency checks
- Statistics generation
- Synthesized Verilog output

### Recorded generic synthesis result

| Metric | Result |
|---|---:|
| Top module | mnist_accelerator_top |
| Generic logic cells | 83,389 |
| Yosys consistency-check problems | 0 |
| Synthesis runtime | Approximately 25.46 seconds |

The synthesis report contains generic cell counts, including:

- ANDNOT: 3,015
- AND: 14,267
- DFFE: 3,595
- DFF: 10
- MUX: 2,604
- NAND: 34,308
- NOR: 695
- NOT: 434
- ORNOT: 1,701
- OR: 3,230
- SDFFE: 1,285
- SDFF: 1
- XNOR: 3,329
- XOR: 14,915

Yosys reported memory/array replacement warnings during synthesis. These relate to arrays being represented as lists of registers in the synthesized design.

**Important:** The 83,389 figure is a generic synthesized logic-cell count. It is not a SkyWater 130nm standard-cell area measurement.

No measured ASIC area, power, clock frequency, critical-path delay, or physical-design result is claimed in this report.

## 8. Verification Artifacts

Important project artifacts include:

### RTL



- `src/rtl/relu.v`

- `src/rtl/requantize.v`

- `src/rtl/systolic_array_4x4.v`

- `src/rtl/weight_stationary_pe.v`
- `src/rtl/weight_stationary_array_4x4.v`
- `src/rtl/weight_stationary_array_4x4_fsm.v`
- `src/rtl/fc1_tile_accumulator.v`
- `src/rtl/fc1_postprocess.v`
- `src/rtl/fc1_controller.v`
- `src/rtl/fc2_classifier.v`
- `src/rtl/accelerator_top.v`
- `src/rtl/mnist_accelerator_top.v`

### Testbenches

- `testbench/tb_weight_stationary_pe.v`
- `testbench/tb_weight_stationary_array_4x4.v`
- `testbench/tb_weight_stationary_array_4x4_fsm.v`
- `testbench/tb_fc1_tile_accumulator.v`
- `testbench/tb_fc1_controller.v`
- `testbench/tb_fc2_classifier.v`
- `testbench/tb_mnist_accelerator_top.v`
- `testbench/tb_mnist_accelerator_top_reset.v`
- `testbench/tb_mnist_accelerator_top_multibatch.v`
- `testbench/tb_mnist_1000_images.v`

- `testbench/tb_requantize_boundary.v`

- `testbench/tb_fc1_postprocess_boundary.v`

- `testbench/tb_fc1_tile_accumulator_audit.v`

- `testbench/tb_fc1_tile_accumulator_extended.v`

- `testbench/tb_fc1_tile_accumulator_overflow.v`

- `testbench/tb_fc1_controller_protocol.v`

- `testbench/tb_fc1_all_tiles.v`

- `testbench/tb_fc2_arithmetic_audit.v`

- `testbench/tb_accelerator_top.v`

- `testbench/tb_fc1_controller_repeat.v`

- `testbench/tb_fc1_controller_reset.v`

- `testbench/tb_fc1_mnist_integration.v`

- `testbench/tb_fc1_mnist_postprocess.v`

- `testbench/tb_fc1_postprocess.v`

- `testbench/tb_fc2_classifier_repeat.v`

- `testbench/tb_fc2_classifier_reset.v`

- `testbench/tb_matrix_demo.v`

- `testbench/tb_mnist_100_images.v`

- `testbench/tb_mnist_end_to_end.v`

- `testbench/tb_python_vectors.v`

- `testbench/tb_relu.v`

- `testbench/tb_requantize.v`

- `testbench/tb_systolic_4x4.v`

- `testbench/tb_weight_stationary_control.v`

- `testbench/tb_weight_stationary_fsm_control.v`

### Python reference and vector generation

- `src/python/generate_fc1_expected.py`
- `src/python/generate_fc2_expected.py`
- `src/python/evaluate_int8.py`
- `src/python/generate_100_image_vectors.py` (legacy filename; currently configured to generate 1,000 images)

- `src/python/audit_fc1_accumulator_range.py`

- `src/python/export_int32_biases.py`

- `src/python/generate_test_vectors.py`

- `src/python/golden_model.py`

- `src/python/quantize_model.py`

- `src/python/tiled_inference.py`

- `src/python/train_mnist.py`

- `src/python/verify_fixed_point.py`

### Synthesis evidence

- `synthesis/logs/yosys_final_synthesis.log`
- `synthesis/results/mnist_accelerator_top_final_synth.v`

### Checkpoints

- `documentation/checkpoints/control_verified/`
- `documentation/checkpoints/end_to_end_v1/`
- `documentation/checkpoints/fc1_mnist_integration_v1/`
- `documentation/checkpoints/integration_v1/`
- `documentation/checkpoints/integration_v2/`
- `documentation/checkpoints/mnist_top_v1/`
- `documentation/checkpoints/multibatch_v1/`
- `documentation/checkpoints/python_rtl_v1/`
- `documentation/checkpoints/reset_recovery_v1/`
- `documentation/checkpoints/synthesis_v1/`
- `documentation/checkpoints/fsm_v1/`
- `documentation/checkpoints/post_processing_v1/`
- `documentation/checkpoints/fc1_accumulator_v1/`
- `documentation/checkpoints/fc1_postprocess_v1/`
- `documentation/checkpoints/fc1_controller_v1/`
- `documentation/checkpoints/fc1_protocol_exhaustive_v1/`
- `documentation/checkpoints/fc2_v1/`
- `documentation/checkpoints/mnist_top_verified_v1/`
- `documentation/checkpoints/mnist_top_verified_v2/`
- `documentation/checkpoints/synthesis_final_v2/`
- `documentation/checkpoints/mnist_1000_v1/`
  - `documentation/checkpoints/mnist_1000_v1/mnist_1000_verification.log`
- `documentation/checkpoints/top_multibatch_v1/`

The top-level repeated-inference checkpoint includes its testbench, synthesis log and simulation verification log.

## 9. Known Limitations

The current implementation has the following limitations:

1. The matrix engine is sequential weight-stationary, not a fully pipelined systolic wavefront.
2. FC1 tile generation and raw image preprocessing are external to the RTL top-level.
3. FC2 uses a sequential MAC architecture rather than a systolic FC2 array.
4. The FC1 accumulator uses signed INT32 addition without overflow detection or saturation. The tested reference vectors did not overflow, but this is not a universal mathematical guarantee.
5. FC1 post-processing uses a fixed Q31 multiplier and fixed quantization parameters rather than a fully programmable quantization interface.
6. The current RTL does not implement a complete raw-pixel-to-prediction hardware input pipeline.
7. Generic Yosys synthesis is not equivalent to technology-mapped ASIC implementation.
8. No OpenLane/Sky130 physical-design run has been completed and verified.

## 10. Future Work

Potential next steps:

- Run technology-mapped synthesis using the SkyWater 130nm PDK.
- Complete the OpenLane physical-design flow.
- Measure standard-cell area, timing and power using the appropriate tools.
- Add overflow detection or saturation to the FC1 accumulator if required by the supported numerical range.
- Improve the dataflow toward a fully pipelined systolic implementation.
- Integrate raw image preprocessing and tile generation into a software interface.
- Build a web demonstration that invokes the actual RTL simulation rather than a separate JavaScript-only matrix simulation.
- Expand verification to additional random vectors, boundary conditions and longer repeated-inference sequences.

## 11. Final Conclusion

The project implements an INT8 neural-network inference accelerator with a 4×4 weight-stationary matrix engine, FC1 accumulation and post-processing, a sequential FC2 classifier, and an integrated top-level controller.

The recorded verification demonstrates:

- Module-level functional tests passing.
- FC1 and FC2 outputs matching the Python reference.
- Reset recovery in the individual controllers and integrated top.
- Repeated inference without reset.
- Exact RTL/Python logit and prediction agreement across 1,000 MNIST test images.
- Successful generic Yosys synthesis with zero reported consistency-check problems.

The 1,000-image test achieved 94.80% classification accuracy while all 10,000 RTL logits and all 1,000 predictions matched the Python reference.

The design has passed extensive functional and simulation-based verification. Physical ASIC implementation and PPA measurement remain future work.

---

**Report status:** Functional verification and generic synthesis documented.
**Physical-design status:** Not yet measured.
