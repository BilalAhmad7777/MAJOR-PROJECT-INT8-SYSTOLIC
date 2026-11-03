`timescale 1ns / 1ps

module tb_python_vectors;

    localparam NUM_TESTS = 1568;

    reg clk;
    reg rst_n;
    reg load_weights;
    reg start;

    reg signed [127:0] activation_matrix;
    reg signed [127:0] weight_matrix;

    wire signed [511:0] result_matrix;
    wire done;

    reg [127:0] activation_mem [0:NUM_TESTS-1];
    reg [127:0] weight_mem [0:NUM_TESTS-1];
    reg [511:0] expected_mem [0:NUM_TESTS-1];

    integer test_number;
    integer cycle_count;
    integer pass_count;
    integer fail_count;

    weight_stationary_array_4x4_fsm dut (
        .clk(clk),
        .rst_n(rst_n),
        .load_weights(load_weights),
        .start(start),
        .activation_matrix(activation_matrix),
        .weight_matrix(weight_matrix),
        .result_matrix(result_matrix),
        .done(done)
    );

    always #5 clk = ~clk;

    initial begin

        clk = 0;
        rst_n = 0;
        load_weights = 0;
        start = 0;
        activation_matrix = 0;
        weight_matrix = 0;

        pass_count = 0;
        fail_count = 0;

        $display("===== PYTHON TO VERILOG VERIFICATION =====");

        $readmemh(
            "verification/test_vectors/fc1_hardware/activation.hex",
            activation_mem
        );

        $readmemh(
            "verification/test_vectors/fc1_hardware/weights.hex",
            weight_mem
        );

        $readmemh(
            "verification/test_vectors/fc1_hardware/expected_partial_sums.hex",
            expected_mem
        );

        // Reset the accelerator.
        repeat (3) @(negedge clk);
        rst_n = 1;

        for (test_number = 0;
             test_number < NUM_TESTS;
             test_number = test_number + 1) begin

            // Present tile and request loading.
            @(negedge clk);
            activation_matrix = activation_mem[test_number];
            weight_matrix = weight_mem[test_number];
            load_weights = 1;

            // Allow IDLE -> LOAD -> IDLE.
            @(negedge clk);
            load_weights = 0;

            @(negedge clk);

            // Start computation.
            start = 1;

            @(negedge clk);
            start = 0;

            // Wait for completion, with timeout.
            cycle_count = 0;

            while ((done !== 1'b1) && (cycle_count < 10)) begin
                @(negedge clk);
                cycle_count = cycle_count + 1;
            end

            if (done !== 1'b1) begin
                $display(
                    "[FAIL] Transaction %0d timed out",
                    test_number
                );
                fail_count = fail_count + 1;
            end
            else if (result_matrix !== expected_mem[test_number]) begin
                $display(
                    "[FAIL] Transaction %0d output mismatch",
                    test_number
                );
                fail_count = fail_count + 1;
            end
            else begin
                pass_count = pass_count + 1;
            end

            if ((test_number + 1) % 100 == 0) begin
                $display(
                    "[INFO] Completed %0d / %0d transactions",
                    test_number + 1,
                    NUM_TESTS
                );
            end

        end

        $display("\n===== VERIFICATION RESULTS =====");
        $display("Transactions passed: %0d", pass_count);
        $display("Transactions failed: %0d", fail_count);
        $display("Total transactions: %0d", NUM_TESTS);

        $display(
            "Individual INT32 output comparisons: %0d",
            pass_count * 16
        );

        if (fail_count == 0)
            $display("\n[PASS] ALL PYTHON VECTORS MATCH VERILOG");
        else
            $display("\n[FAIL] VERIFICATION ERRORS DETECTED");

        $finish;
    end

endmodule
