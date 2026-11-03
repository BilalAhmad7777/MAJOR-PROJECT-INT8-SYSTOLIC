`timescale 1ns / 1ps

module tb_mnist_accelerator_top;

    localparam NUM_IMAGES = 4;
    localparam NUM_GROUPS = 8;
    localparam TILES_PER_GROUP = 196;
    localparam NUM_TILES = 1568;
    localparam EXPECTED_CHECKS = 45;

    reg clk;
    reg rst_n;
    reg start;

    reg tile_valid;
    reg signed [127:0] activation_matrix;
    reg signed [127:0] fc1_weight_matrix;
    reg signed [127:0] fc1_bias_group;

    wire tile_ready;
    wire fc1_busy;

    reg [2559:0] fc2_weight_matrix;
    reg [319:0] fc2_bias_vector;

    wire [1279:0] batch_logits;
    wire busy;
    wire done;

    reg [127:0] activation_mem [0:NUM_TILES-1];
    reg [127:0] fc1_weight_mem [0:NUM_TILES-1];
    reg [31:0] fc1_bias_mem [0:31];

    reg [7:0] fc2_weight_mem [0:319];
    reg [31:0] fc2_bias_mem [0:9];
    reg [31:0] expected_logits_mem [0:39];

    integer group_number;
    integer tile_number;
    integer test_number;
    integer image;
    integer i;
    integer neuron;
    integer cycle_count;
    integer passed;
    integer failed;
    integer prediction;
    integer expected_prediction;

    // --------------------------------------------
    // DUT: Complete MNIST accelerator top
    // --------------------------------------------

    mnist_accelerator_top dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),

        .tile_valid(tile_valid),
        .activation_matrix(activation_matrix),
        .fc1_weight_matrix(fc1_weight_matrix),
        .fc1_bias_group(fc1_bias_group),

        .tile_ready(tile_ready),
        .fc1_busy(fc1_busy),

        .fc2_weight_matrix(fc2_weight_matrix),
        .fc2_bias_vector(fc2_bias_vector),

        .batch_logits(batch_logits),
        .busy(busy),
        .done(done)
    );

    // --------------------------------------------
    // Clock
    // --------------------------------------------

    initial clk = 0;
    always #5 clk = ~clk;

    // --------------------------------------------
    // Main test
    // --------------------------------------------

    initial begin

        rst_n = 0;
        start = 0;
        tile_valid = 0;

        activation_matrix = 0;
        fc1_weight_matrix = 0;
        fc1_bias_group = 0;

        fc2_weight_matrix = 0;
        fc2_bias_vector = 0;

        passed = 0;
        failed = 0;

        $display("==========================================");
        $display("    MNIST ACCELERATOR TOP-LEVEL TEST");
        $display("==========================================");

        // ----------------------------------------
        // Load FC1 vectors
        // ----------------------------------------

        $readmemh(
            "verification/test_vectors/fc1_hardware/activation.hex",
            activation_mem
        );

        $readmemh(
            "verification/test_vectors/fc1_hardware/weights.hex",
            fc1_weight_mem
        );

        $readmemh(
           "verification/test_vectors/fc1_hardware/fc1_bias_int32_mem.hex",
            fc1_bias_mem
        );

        // ----------------------------------------
        // Load FC2 vectors
        // ----------------------------------------

        $readmemh(
            "verification/test_vectors/fc2_hardware/fc2_weights_int8.hex",
            fc2_weight_mem
        );

        $readmemh(
            "verification/test_vectors/fc2_hardware/fc2_bias_int32.hex",
            fc2_bias_mem
        );

        $readmemh(
            "verification/test_vectors/fc2_hardware/fc2_expected_logits_int32.hex",
            expected_logits_mem
        );

        // Pack FC2 weights
        for (i = 0; i < 320; i = i + 1) begin
            fc2_weight_matrix[i*8 +: 8] = fc2_weight_mem[i];
        end

        // Pack FC2 biases
        for (i = 0; i < 10; i = i + 1) begin
            fc2_bias_vector[i*32 +: 32] = fc2_bias_mem[i];
        end

        // ----------------------------------------
        // Reset
        // ----------------------------------------

        repeat (3) @(negedge clk);
        rst_n = 1;

        // ----------------------------------------
        // Start complete accelerator
        // ----------------------------------------

        @(negedge clk);
        start = 1;

        @(negedge clk);
        start = 0;

        $display("\n[TOP] Accelerator started");

        // ----------------------------------------
        // Stream all 1568 FC1 tiles
        // ----------------------------------------

        for (
            group_number = 0;
            group_number < NUM_GROUPS;
            group_number = group_number + 1
        ) begin

            $display(
                "\n[FC1] Sending group %0d",
                group_number
            );

            // Supply four biases for this output group
            fc1_bias_group = 0;

            for (i = 0; i < 4; i = i + 1) begin
                fc1_bias_group[i*32 +: 32] =
                    fc1_bias_mem[group_number*4+i];
            end

            // Send 196 tiles
            for (
                tile_number = 0;
                tile_number < TILES_PER_GROUP;
                tile_number = tile_number + 1
            ) begin

                test_number =
                    group_number*TILES_PER_GROUP + tile_number;

                // Wait until the DUT can accept a tile
                cycle_count = 0;

                while (
                    tile_ready !== 1'b1 &&
                    cycle_count < 100
                ) begin
                    @(negedge clk);
                    cycle_count = cycle_count + 1;
                end

                if (tile_ready !== 1'b1) begin

                    $display(
                        "[FAIL] Tile timeout: group=%0d tile=%0d",
                        group_number,
                        tile_number
                    );

                    failed = failed + 1;
                    $finish;

                end

                // Present tile data
                activation_matrix = activation_mem[test_number];
                fc1_weight_matrix = fc1_weight_mem[test_number];
                tile_valid = 1;

                // DUT accepts tile at next positive edge
                @(negedge clk);
                tile_valid = 0;

            end

            $display(
                "[FC1] Group %0d tiles submitted",
                group_number
            );

        end

        // ----------------------------------------
        // Wait for complete top-level inference
        // ----------------------------------------

        $display("\n[TOP] All 1568 FC1 tiles submitted");
        $display("[TOP] Waiting for FC1 and FC2 completion...");

        cycle_count = 0;

        while (
            done !== 1'b1 &&
            cycle_count < 100000
        ) begin
            @(negedge clk);
            cycle_count = cycle_count + 1;
        end

        if (done !== 1'b1) begin

            $display("[FAIL] Top-level completion timeout");
            $display("busy     = %b", busy);
            $display("fc1_busy = %b", fc1_busy);

            failed = failed + 1;
            $finish;

        end

        passed = passed + 1;

        $display("\n[PASS] Top-level accelerator asserted done");

        // ----------------------------------------
        // Compare all 40 INT32 logits
        // ----------------------------------------

        $display("\n===== FINAL LOGIT COMPARISON =====");

        for (
            image = 0;
            image < NUM_IMAGES;
            image = image + 1
        ) begin

            $display("\n===== IMAGE %0d =====", image);

            for (
                neuron = 0;
                neuron < 10;
                neuron = neuron + 1
            ) begin

                if (
                    $signed(
                        batch_logits[image*320 + neuron*32 +: 32]
                    ) !==
                    $signed(
                        expected_logits_mem[image*10+neuron]
                    )
                ) begin

                    $display(
                        "[FAIL] Image %0d Neuron %0d RTL=%0d Python=%0d",
                        image,
                        neuron,
                        $signed(
                            batch_logits[image*320 + neuron*32 +: 32]
                        ),
                        $signed(
                            expected_logits_mem[image*10+neuron]
                        )
                    );

                    failed = failed + 1;

                end
                else begin

                    passed = passed + 1;

                end

            end

            // ------------------------------------
            // Determine predicted digit
            // ------------------------------------

            prediction = 0;

            for (
                neuron = 1;
                neuron < 10;
                neuron = neuron + 1
            ) begin

                if (
                    $signed(
                        batch_logits[image*320 + neuron*32 +: 32]
                    ) >
                    $signed(
                        batch_logits[image*320 + prediction*32 +: 32]
                    )
                ) begin

                    prediction = neuron;

                end

            end

            case (image)

                0: expected_prediction = 7;
                1: expected_prediction = 2;
                2: expected_prediction = 1;
                3: expected_prediction = 0;

                default: expected_prediction = -1;

            endcase

            if (prediction == expected_prediction) begin

                $display(
                    "[PASS] Predicted digit: %0d",
                    prediction
                );

                passed = passed + 1;

            end
            else begin

                $display(
                    "[FAIL] Expected digit %0d, got %0d",
                    expected_prediction,
                    prediction
                );

                failed = failed + 1;

            end

        end

        // ----------------------------------------
        // Final results
        // ----------------------------------------

        $display("\n==========================================");
        $display("       TOP-LEVEL TEST RESULTS");
        $display("==========================================");

        $display("Passed: %0d", passed);
        $display("Failed: %0d", failed);
        $display("Expected checks: %0d", EXPECTED_CHECKS);

        if (
            passed == EXPECTED_CHECKS &&
            failed == 0
        ) begin

            $display("\n[PASS] MNIST ACCELERATOR TOP VERIFIED");

        end
        else begin

            $display("\n[FAIL] TOP-LEVEL VERIFICATION FAILED");

        end

        $finish;

    end

    // --------------------------------------------
    // Global timeout
    // --------------------------------------------

    initial begin

        #100000000;

        $display("[FAIL] Global simulation timeout");
        $finish;

    end

endmodule
