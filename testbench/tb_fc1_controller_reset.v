
`timescale 1ns / 1ps

module tb_fc1_controller_reset;

    localparam NUM_GROUPS = 8;
    localparam TILES_PER_GROUP = 196;
    localparam NUM_TESTS = 1568;

    reg clk, rst_n, start, tile_valid;
    reg signed [127:0] activation_matrix;
    reg signed [127:0] weight_matrix;
    reg signed [127:0] bias_group;

    wire tile_ready, busy, done;
    wire [1023:0] hidden_output;

    reg [127:0] activation_mem [0:NUM_TESTS-1];
    reg [127:0] weight_mem [0:NUM_TESTS-1];
    reg [31:0] bias_mem [0:31];
    reg [7:0] expected_hidden_mem [0:127];

    integer g, t, i, idx;
    integer pass_count, fail_count, cycle_count;
    integer sample_number, neuron_number;

    fc1_controller dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .tile_valid(tile_valid),
        .activation_matrix(activation_matrix),
        .weight_matrix(weight_matrix),
        .bias_group(bias_group),
        .tile_ready(tile_ready),
        .busy(busy),
        .done(done),
        .hidden_output(hidden_output)
    );

    always #5 clk = ~clk;

    // Send one tile and wait for acceptance.
    task send_tile;
        input integer tile_index;
        begin
            cycle_count = 0;

            while (tile_ready !== 1'b1 && cycle_count < 100) begin
                @(negedge clk);
                cycle_count = cycle_count + 1;
            end

            if (tile_ready !== 1'b1) begin
                $display("[FAIL] tile_ready timeout at tile %0d",
                         tile_index);
                fail_count = fail_count + 1;
                $finish;
            end

            activation_matrix = activation_mem[tile_index];
            weight_matrix = weight_mem[tile_index];
            tile_valid = 1;

            @(negedge clk);
            tile_valid = 0;
        end
    endtask

    // Start a new FC1 batch.
    task start_batch;
        begin
            @(negedge clk);
            start = 1;
            @(negedge clk);
            start = 0;
        end
    endtask

    initial begin
        clk = 0;
        rst_n = 0;
        start = 0;
        tile_valid = 0;
        activation_matrix = 0;
        weight_matrix = 0;
        bias_group = 0;
        pass_count = 0;
        fail_count = 0;
        cycle_count = 0;

        $display("========================================");
        $display(" FC1 RESET AND RECOVERY TEST");
        $display("========================================");

        // Load the same Python-generated vectors.
        $readmemh(
            "verification/test_vectors/fc1_hardware/activation.hex",
            activation_mem
        );

        $readmemh(
            "verification/test_vectors/fc1_hardware/weights.hex",
            weight_mem
        );

        $readmemh(
            "verification/test_vectors/fc1_hardware/fc1_bias_int32_mem.hex",
            bias_mem
        );

        $readmemh(
            "verification/test_vectors/fc1_hardware/expected_hidden_int8.hex",
            expected_hidden_mem
        );

        // Initial reset.
        repeat (3) @(negedge clk);
        rst_n = 1;

        // Start inference and send ten tiles.
        start_batch();

        bias_group = 0;
        for (i = 0; i < 4; i = i + 1)
            bias_group[i*32 +: 32] = bias_mem[i];

        for (t = 0; t < 10; t = t + 1)
            send_tile(t);

        $display("[INFO] Ten tiles accepted. Interrupting inference.");

        // Assert reset between clock edges.
        @(negedge clk);
        rst_n = 0;
        start = 0;
        tile_valid = 0;

        // Hold reset through multiple rising edges.
        repeat (3) @(negedge clk);

        // Check controller status and output clearing.
        if (busy !== 0) begin
            $display("[FAIL] busy not cleared by reset");
            fail_count = fail_count + 1;
        end else begin
            $display("[PASS] busy cleared");
            pass_count = pass_count + 1;
        end

        if (done !== 0) begin
            $display("[FAIL] done not cleared by reset");
            fail_count = fail_count + 1;
        end else begin
            $display("[PASS] done cleared");
            pass_count = pass_count + 1;
        end

        if (hidden_output !== 1024'b0) begin
            $display("[FAIL] hidden_output not cleared");
            fail_count = fail_count + 1;
        end else begin
            $display("[PASS] hidden_output cleared");
            pass_count = pass_count + 1;
        end

        // Release reset and start a fresh inference.
        @(negedge clk);
        rst_n = 1;

        $display("[INFO] Starting fresh inference after reset.");

        start_batch();

        // Send all eight groups and 196 tiles per group.
        for (g = 0; g < NUM_GROUPS; g = g + 1) begin

            bias_group = 0;

            for (i = 0; i < 4; i = i + 1)
                bias_group[i*32 +: 32] = bias_mem[g*4+i];

            for (t = 0; t < TILES_PER_GROUP; t = t + 1) begin
                idx = g*TILES_PER_GROUP+t;
                send_tile(idx);
            end

            $display("[INFO] Group %0d completed input delivery", g);
        end

        // Wait for inference completion.
        cycle_count = 0;

        while (done !== 1'b1 && cycle_count < 500) begin
            @(negedge clk);
            cycle_count = cycle_count + 1;
        end

        if (done !== 1'b1) begin
            $display("[FAIL] Completion timeout after recovery");
            fail_count = fail_count + 1;
        end else begin
            $display("[PASS] Fresh inference completed");
            pass_count = pass_count + 1;
        end

        // Compare all 128 hidden activations.
        for (sample_number = 0; sample_number < 4;
             sample_number = sample_number + 1) begin

            for (neuron_number = 0; neuron_number < 32;
                 neuron_number = neuron_number + 1) begin

                if (hidden_output[
                    (sample_number*32+neuron_number)*8 +: 8
                ] !== expected_hidden_mem[
                    sample_number*32+neuron_number
                ]) begin

                    $display(
                        "[FAIL] Sample=%0d Neuron=%0d Expected=%0d Got=%0d",
                        sample_number,
                        neuron_number,
                        $signed(expected_hidden_mem[
                            sample_number*32+neuron_number
                        ]),
                        $signed(hidden_output[
                            (sample_number*32+neuron_number)*8 +: 8
                        ])
                    );

                    fail_count = fail_count + 1;

                end else begin
                    pass_count = pass_count + 1;
                end
            end
        end

        $display("");
        $display("========================================");
        $display(" FC1 RESET RECOVERY RESULTS");
        $display("========================================");
        $display("Passed: %0d", pass_count);
        $display("Failed: %0d", fail_count);
        $display("Expected hidden comparisons: 128");

        if (fail_count == 0)
            $display("[PASS] FC1 RESET RECOVERY VERIFIED");
        else
            $display("[FAIL] FC1 RESET RECOVERY FAILED");

        $finish;
    end

endmodule
