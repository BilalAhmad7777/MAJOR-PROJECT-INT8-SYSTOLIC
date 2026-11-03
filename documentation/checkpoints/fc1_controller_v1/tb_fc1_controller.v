
`timescale 1ns / 1ps

module tb_fc1_controller;

    localparam NUM_GROUPS = 8;
    localparam TILES_PER_GROUP = 196;
    localparam NUM_TESTS = 1568;

    reg clk;
    reg rst_n;
    reg start;
    reg tile_valid;

    reg signed [127:0] activation_matrix;
    reg signed [127:0] weight_matrix;
    reg signed [127:0] bias_group;

    wire tile_ready;
    wire busy;
    wire done;
    wire [1023:0] hidden_output;

    reg [127:0] activation_mem [0:NUM_TESTS-1];
    reg [127:0] weight_mem [0:NUM_TESTS-1];

    reg [31:0] bias_mem [0:31];
    reg [7:0] expected_hidden_mem [0:127];

    integer group_number;
    integer tile_number;
    integer test_number;
    integer sample_number;
    integer neuron_number;
    integer pass_count;
    integer fail_count;
    integer cycle_count;
    integer i;

    // --------------------------------------------
    // DUT
    // --------------------------------------------

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

    // --------------------------------------------
    // Clock
    // --------------------------------------------

    always #5 clk = ~clk;

    // --------------------------------------------
    // Main test
    // --------------------------------------------

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

        $display("==========================================");
        $display("       FC1 CONTROLLER MNIST TEST");
        $display("==========================================");

        // Load Python-generated test vectors

        $readmemh(
            "verification/test_vectors/fc1_hardware/activation.hex",
            activation_mem
        );

        $readmemh(
            "verification/test_vectors/fc1_hardware/weights.hex",
            weight_mem
        );

        $readmemh(
            "verification/test_vectors/fc1_hardware/fc1_bias_int32.hex",
            bias_mem
        );

        $readmemh(
            "verification/test_vectors/fc1_hardware/expected_hidden_int8.hex",
            expected_hidden_mem
        );

        // Reset DUT

        repeat (3) @(negedge clk);
        rst_n = 1;

        // Start the complete four-image batch

        @(negedge clk);
        start = 1;

        @(negedge clk);
        start = 0;

        // ----------------------------------------
        // Process all 8 output groups
        // ----------------------------------------

        for (group_number = 0;
             group_number < NUM_GROUPS;
             group_number = group_number + 1) begin

            $display(
                "\n[INFO] Sending output group %0d",
                group_number
            );

            // Pack four INT32 biases into bias_group

            bias_group = 0;

            for (i = 0; i < 4; i = i + 1) begin
                bias_group[i*32 +: 32] =
                    bias_mem[group_number*4 + i];
            end

            // Send 196 tiles

            for (tile_number = 0;
                 tile_number < TILES_PER_GROUP;
                 tile_number = tile_number + 1) begin

                test_number =
                    group_number * TILES_PER_GROUP + tile_number;

                // Wait for controller readiness

                cycle_count = 0;

                while (tile_ready !== 1'b1 &&
                       cycle_count < 100) begin
                    @(negedge clk);
                    cycle_count = cycle_count + 1;
                end

                if (tile_ready !== 1'b1) begin
                    $display(
                        "[FAIL] tile_ready timeout group=%0d tile=%0d",
                        group_number,
                        tile_number
                    );
                    fail_count = fail_count + 1;
                    $finish;
                end

                // Present tile data

                activation_matrix = activation_mem[test_number];
                weight_matrix = weight_mem[test_number];

                tile_valid = 1;

                // Controller accepts tile at next positive edge

                @(negedge clk);
                tile_valid = 0;

            end

            $display(
                "[INFO] All 196 tiles sent for group %0d",
                group_number
            );

        end

        // ----------------------------------------
        // Wait for complete FC1 batch
        // ----------------------------------------

        cycle_count = 0;

        while (done !== 1'b1 && cycle_count < 500) begin
            @(negedge clk);
            cycle_count = cycle_count + 1;
        end

        if (done !== 1'b1) begin

            $display("[FAIL] FC1 controller completion timeout");

            $display("busy=%b", busy);

            fail_count = fail_count + 1;

        end else begin

            $display("\n[PASS] FC1 controller asserted done");

            pass_count = pass_count + 1;

        end

        // ----------------------------------------
        // Compare all 128 hidden activations
        // ----------------------------------------

        $display("\n===== HIDDEN OUTPUT COMPARISON =====");

        for (sample_number = 0;
             sample_number < 4;
             sample_number = sample_number + 1) begin

            for (neuron_number = 0;
                 neuron_number < 32;
                 neuron_number = neuron_number + 1) begin

                if (hidden_output[
                    (sample_number*32 + neuron_number)*8 +: 8
                ] !== expected_hidden_mem[
                    sample_number*32 + neuron_number
                ]) begin

                    $display(
                        "[FAIL] Sample=%0d Neuron=%0d Expected=%0d Got=%0d",
                        sample_number,
                        neuron_number,
                        $signed(expected_hidden_mem[
                            sample_number*32 + neuron_number
                        ]),
                        $signed(hidden_output[
                            (sample_number*32 + neuron_number)*8 +: 8
                        ])
                    );

                    fail_count = fail_count + 1;

                end else begin

                    pass_count = pass_count + 1;

                end

            end

        end

        // ----------------------------------------
        // Results
        // ----------------------------------------

        $display("\n==========================================");
        $display("       FC1 CONTROLLER TEST RESULTS");
        $display("==========================================");

        $display("Passed: %0d", pass_count);
        $display("Failed: %0d", fail_count);
        $display("Expected hidden comparisons: 128");

        if (fail_count == 0)
            $display("\n[PASS] FC1 CONTROLLER MATCHES PYTHON");
        else
            $display("\n[FAIL] FC1 CONTROLLER TEST FAILED");

        $finish;

    end

endmodule
