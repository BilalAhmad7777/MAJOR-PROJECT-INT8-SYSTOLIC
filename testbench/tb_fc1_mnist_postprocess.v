`timescale 1ns / 1ps

module tb_fc1_mnist_postprocess;

    localparam NUM_GROUPS = 8;
    localparam TILES_PER_GROUP = 196;
    localparam NUM_TESTS = 1568;

    reg clk, rst_n;
    reg load_weights, start;
    reg signed [127:0] activation_matrix, weight_matrix;

    wire signed [511:0] tile_result_matrix;
    wire matrix_done;

    reg clear_acc, tile_valid;
    wire signed [511:0] accumulated_matrix;
    wire accumulator_done;

    reg [127:0] activation_mem [0:NUM_TESTS-1];
    reg [127:0] weight_mem [0:NUM_TESTS-1];

    reg [511:0] expected_partial_mem [0:NUM_TESTS-1];

    reg [31:0] bias_mem [0:31];
    reg [7:0] expected_hidden_mem [0:127];

    wire signed [7:0] actual_hidden [0:15];
    reg signed [31:0] bias_for_lane [0:15];

    integer expected_sum [0:15];

    integer group_number, tile_number;
    integer test_number, i, lane;
    integer cycle_count;
    integer pass_count, fail_count;
    integer sample_number, neuron_number;

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

   genvar g;
   generate
    for (g = 0; g < 16; g = g + 1) begin : POSTPROCESS_LANES
        fc1_postprocess postprocess (
            .accumulator(accumulated_matrix[g*32 +: 32]),
            .bias(bias_for_lane[g]),
            .activation_int8(actual_hidden[g])
        );
    end
endgenerate

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

        $display("===== FC1 MNIST POST-PROCESSING RTL TEST =====");

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
            expected_partial_mem
        );

        $readmemh(
            "verification/test_vectors/fc1_hardware/fc1_bias_int32.hex",
            bias_mem
        );

        $readmemh(
            "verification/test_vectors/fc1_hardware/expected_hidden_int8.hex",
            expected_hidden_mem
        );

        repeat (3) @(negedge clk);
        rst_n = 1;

        for (group_number = 0;
             group_number < NUM_GROUPS;
             group_number = group_number + 1) begin
                 // Select the four neuron biases for this output group.
    for (i = 0; i < 16; i = i + 1)
        bias_for_lane[i] = bias_mem[
            group_number * 4 + (i % 4)
        ];

    #1;

            $display(
                "\n[INFO] Processing FC1 output group %0d",
                group_number
            );

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

                for (i = 0; i < 16; i = i + 1)
                    expected_sum[i] = expected_sum[i]
                        + $signed(
                            expected_partial_mem[test_number][i*32 +: 32]
                        );

                @(negedge clk);
                activation_matrix = activation_mem[test_number];
                weight_matrix = weight_mem[test_number];
                load_weights = 1;

                @(negedge clk);
                load_weights = 0;

                @(negedge clk);
                start = 1;

                @(negedge clk);
                start = 0;

                cycle_count = 0;

                while ((matrix_done !== 1'b1) &&
                       (cycle_count < 20)) begin
                    @(negedge clk);
                    cycle_count = cycle_count + 1;
                end

                if (matrix_done !== 1'b1) begin
                    $display(
                        "[FAIL] Matrix timeout group=%0d tile=%0d",
                        group_number, tile_number
                    );
                    fail_count = fail_count + 1;
                    $finish;
                end

                tile_valid = 1;
                @(negedge clk);
                tile_valid = 0;

                if (tile_number == TILES_PER_GROUP - 1) begin
                    if (accumulator_done !== 1'b1) begin
                        $display(
                            "[FAIL] Accumulator done missing group=%0d",
                            group_number
                        );
                        fail_count = fail_count + 1;
                    end else begin
                        pass_count = pass_count + 1;
                    end
                end
            end

            // Check accumulated INT32 values.
            // Check accumulated INT32 values.
for (i = 0; i < 16; i = i + 1) begin
    if ($signed(accumulated_matrix[i*32 +: 32])
        !== expected_sum[i]) begin

        $display(
            "[FAIL] Accumulator group=%0d lane=%0d expected=%0d got=%0d",
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

if (group_number == 0) begin
    $display("\n[DEBUG] Group 0 accumulator and bias values:");

    for (i = 0; i < 4; i = i + 1) begin
        $display(
            "Lane %0d: accumulator=%0d bias=%0d",
            i,
            $signed(accumulated_matrix[i*32 +: 32]),
            bias_for_lane[i]
        );
    end
end

            // Check post-processed INT8 outputs.
            for (lane = 0; lane < 16; lane = lane + 1) begin

                sample_number = lane / 4;
                neuron_number = group_number * 4 + (lane % 4);

                if (actual_hidden[lane] !==
                    expected_hidden_mem[
                        sample_number * 32 + neuron_number
                    ]) begin

                    $display(
                        "[FAIL] Sample=%0d neuron=%0d expected=%0d got=%0d",
                        sample_number,
                        neuron_number,
                        $signed(expected_hidden_mem[
                            sample_number * 32 + neuron_number
                        ]),
                        actual_hidden[lane]
                    );

                    fail_count = fail_count + 1;
                end else begin
                    pass_count = pass_count + 1;
                end
            end

            $display(
                "[INFO] Group %0d completed",
                group_number
            );
        end

        $display("\n===== FC1 POST-PROCESSING RESULTS =====");
        $display("Passed: %0d", pass_count);
        $display("Failed: %0d", fail_count);
        $display("Expected INT8 comparisons: 128");

        if (fail_count == 0)
            $display("\n[PASS] FC1 MNIST POST-PROCESSING MATCHES PYTHON");
        else
            $display("\n[FAIL] FC1 MNIST POST-PROCESSING MISMATCH");

        $finish;
    end

endmodule
