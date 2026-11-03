`timescale 1ns / 1ps

module tb_fc1_mnist_integration;

    localparam NUM_GROUPS = 8;
    localparam TILES_PER_GROUP = 196;
    localparam NUM_TESTS = 1568;

    reg clk;
    reg rst_n;

    reg load_weights;
    reg start;

    reg signed [127:0] activation_matrix;
    reg signed [127:0] weight_matrix;

    wire signed [511:0] tile_result_matrix;
    wire matrix_done;

    reg clear_acc;
    reg tile_valid;

    wire signed [511:0] accumulated_matrix;
    wire accumulator_done;

    reg [127:0] activation_mem [0:NUM_TESTS-1];
    reg [127:0] weight_mem [0:NUM_TESTS-1];
    reg [511:0] expected_mem [0:NUM_TESTS-1];

    reg signed [31:0] expected_sum [0:15];

    integer group_number;
    integer tile_number;
    integer test_number;
    integer i;
    integer cycle_count;
    integer pass_count;
    integer fail_count;

    weight_stationary_array_4x4_fsm matrix_engine (
        .clk(clk),
        .rst_n(rst_n),
        .load_weights(load_weights),
        .start(start),
        .activation_matrix(activation_matrix),
        .weight_matrix(weight_matrix),
        .result_matrix(tile_result_matrix),
        .done(matrix_done)
    );

    fc1_tile_accumulator #(
        .TILE_COUNT(TILES_PER_GROUP)
    ) accumulator (
        .clk(clk),
        .rst_n(rst_n),
        .clear_acc(clear_acc),
        .tile_valid(tile_valid),
        .tile_result_matrix(tile_result_matrix),
        .accumulated_matrix(accumulated_matrix),
        .done(accumulator_done)
    );

    always #5 clk = ~clk;

    initial begin
        clk = 0;
        rst_n = 0;

        load_weights = 0;
        start = 0;
        clear_acc = 0;
        tile_valid = 0;

        activation_matrix = 0;
        weight_matrix = 0;

        pass_count = 0;
        fail_count = 0;

        $display("===== MNIST FC1 RTL INTEGRATION =====");

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

        repeat (3) @(negedge clk);
        rst_n = 1;

        for (group_number = 0;
             group_number < NUM_GROUPS;
             group_number = group_number + 1) begin

            $display("\n[INFO] Starting FC1 output group %0d",
                     group_number);

            // Clear the accumulator for this output group.
            @(negedge clk);
            clear_acc = 1;

            @(negedge clk);
            clear_acc = 0;

            for (i = 0; i < 16; i = i + 1)
                expected_sum[i] = 0;

            for (tile_number = 0;
                 tile_number < TILES_PER_GROUP;
                 tile_number = tile_number + 1) begin

                test_number =
                    group_number * TILES_PER_GROUP + tile_number;

                // Calculate expected accumulated result.
                for (i = 0; i < 16; i = i + 1) begin
                    expected_sum[i] = expected_sum[i]
                        + $signed(expected_mem[test_number][i*32 +: 32]);
                end

                // Load the current activation and weight tiles.
                @(negedge clk);
                activation_matrix = activation_mem[test_number];
                weight_matrix = weight_mem[test_number];
                load_weights = 1;

                @(negedge clk);
                load_weights = 0;

                // Start matrix multiplication.
                @(negedge clk);
                start = 1;

                @(negedge clk);
                start = 0;

                // Wait for matrix engine completion.
                cycle_count = 0;

                while ((matrix_done !== 1'b1) &&
                       (cycle_count < 20)) begin
                    @(negedge clk);
                    cycle_count = cycle_count + 1;
                end

                if (matrix_done !== 1'b1) begin
                    $display(
                        "[FAIL] Matrix timeout: group %0d tile %0d",
                        group_number, tile_number
                    );
                    fail_count = fail_count + 1;
                    $finish;
                end

                // Send completed partial sums to accumulator.
                tile_valid = 1;

                @(negedge clk);
                tile_valid = 0;

                if (tile_number == TILES_PER_GROUP - 1) begin
                    if (accumulator_done !== 1'b1) begin
                        $display(
                            "[FAIL] Accumulator done missing: group %0d",
                            group_number
                        );
                        fail_count = fail_count + 1;
                    end else begin
                        pass_count = pass_count + 1;
                    end
                end

                if ((tile_number + 1) % 50 == 0)
                    $display(
                        "[INFO] Group %0d: processed %0d / %0d tiles",
                        group_number,
                        tile_number + 1,
                        TILES_PER_GROUP
                    );
            end

            // Compare the final accumulated matrix with Python.
            for (i = 0; i < 16; i = i + 1) begin
                if ($signed(accumulated_matrix[i*32 +: 32])
                    !== expected_sum[i]) begin

                    $display(
                        "[FAIL] Group %0d output %0d: expected %0d, got %0d",
                        group_number,
                        i,
                        expected_sum[i],
                        $signed(accumulated_matrix[i*32 +: 32])
                    );

                    fail_count = fail_count + 1;
                end else begin
                    pass_count = pass_count + 1;
                end
            end

            $display("[INFO] Group %0d verification completed",
                     group_number);
        end

        $display("\n===== MNIST FC1 INTEGRATION RESULTS =====");
        $display("Checks passed: %0d", pass_count);
        $display("Checks failed: %0d", fail_count);
        $display("Expected output comparisons: 128");

        if (fail_count == 0)
            $display("\n[PASS] ALL MNIST FC1 RTL RESULTS MATCH PYTHON");
        else
            $display("\n[FAIL] MNIST FC1 INTEGRATION FAILED");

        $finish;
    end

endmodule
