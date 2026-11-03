import numpy as np
from fixed_point import requantize_q31


def validate_int8_matrix(matrix, name):
    """Validate that the input is a 4x4 signed INT8 matrix."""
    matrix = np.asarray(matrix, dtype=np.int64)

    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4)")

    if np.any(matrix < -128) or np.any(matrix > 127):
        raise ValueError(f"{name} contains values outside INT8 range")

    return matrix


def matrix_multiply_int32(a, b):
    """Perform 4x4 INT8 matrix multiplication with INT32 accumulation."""
    a = validate_int8_matrix(a, "Activation matrix")
    b = validate_int8_matrix(b, "Weight matrix")

    # INT64 intermediate prevents accidental Python/NumPy overflow.
    result = a @ b

    # Confirm that the result fits in signed INT32.
    if np.any(result < -(2**31)) or np.any(result > (2**31 - 1)):
        raise OverflowError("Matrix result exceeds INT32 range")

    return result.astype(np.int32)


def relu(matrix):
    """Replace negative values with zero."""
    return np.maximum(matrix, 0).astype(np.int32)


def requantize_int8(matrix, shift):
    """
    Match the RTL requantize module for nonnegative ReLU outputs.

    Uses round-to-nearest with ties upward, arithmetic right shift,
    and signed INT8 saturation.
    """
    if not isinstance(shift, (int, np.integer)) or not 0 <= shift <= 31:
        raise ValueError("shift must be an integer from 0 to 31")

    values = np.asarray(matrix, dtype=np.int64)

    if shift == 0:
        rounded = values
    else:
        rounded = (values + (1 << (shift - 1))) >> shift

    return np.clip(rounded, -128, 127).astype(np.int8)


def requantize_int8_q31(matrix, multiplier):
    """Requantize INT32 accumulators using shared Q31 ties-to-even logic."""
    return requantize_q31(matrix, multiplier)


def accelerator_golden_model(a, b, shift=0):
    """Complete reference model matching the current accelerator."""
    accumulated = matrix_multiply_int32(a, b)
    activated = relu(accumulated)
    quantized = requantize_int8(activated, shift)

    return accumulated, activated, quantized


def self_test():
    print("===== PYTHON GOLDEN MODEL TEST =====")

    a = np.array([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7]
    ], dtype=np.int8)

    b = np.eye(4, dtype=np.int8) * 2

    acc, activated, output = accelerator_golden_model(a, b, shift=1)

    expected = a.astype(np.int32)

    if not np.array_equal(acc, expected * 2):
        raise AssertionError("Matrix multiplication test failed")

    if not np.array_equal(activated, expected * 2):
        raise AssertionError("ReLU test failed")

    if not np.array_equal(output, expected):
        raise AssertionError("Requantization test failed")

    q31_test = np.array([1, 3, 5, 7], dtype=np.int32)
    q31_result = requantize_int8_q31(q31_test, 1 << 30)
    q31_expected = np.array([0, 2, 2, 4], dtype=np.int32)

    if not np.array_equal(q31_result, q31_expected):
        raise AssertionError("Q31 ties-to-even test failed")

    print("[PASS] Q31 ties-to-even requantization")

    print("[PASS] 4x4 matrix multiplication")
    print("[PASS] INT32 accumulation")
    print("[PASS] ReLU")
    print("[PASS] Requantization")
    print("[PASS] INT8 output")

    # Test negative values and saturation.
    test_values = np.array([
        [-100, 0, 100, 10000],
        [127, 128, 200, 300],
        [1, 2, 3, 4],
        [10, 20, 30, 40]
    ], dtype=np.int32)

    activated_test = relu(test_values)
    quantized_test = requantize_int8(activated_test, 0)

    assert quantized_test[0, 0] == 0
    assert quantized_test[0, 3] == 127
    assert quantized_test[1, 1] == 127

    print("[PASS] Negative-value handling")
    print("[PASS] INT8 saturation")

    # Deterministic randomized arithmetic tests.
    rng = np.random.default_rng(2026)

    for test_number in range(100):
        random_a = rng.integers(-128, 128, size=(4, 4), dtype=np.int16)
        random_b = rng.integers(-128, 128, size=(4, 4), dtype=np.int16)
        shift = int(rng.integers(0, 8))

        acc, activated, output = accelerator_golden_model(
            random_a, random_b, shift
        )

        reference = random_a.astype(np.int64) @ random_b.astype(np.int64)
        reference = np.maximum(reference, 0)
        reference = np.clip(
            (reference + (1 << (shift - 1))) >> shift
            if shift else reference,
            -128, 127
        ).astype(np.int8)

        assert np.array_equal(output, reference)

    print("[PASS] 100 randomized matrix tests")
    print("\nALL PYTHON GOLDEN MODEL TESTS PASSED")

    print("\nExample output:")
    print(output)


if __name__ == "__main__":
    self_test()
