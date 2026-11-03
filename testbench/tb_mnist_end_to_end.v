
`timescale 1ns / 1ps

module tb_mnist_end_to_end;

    localparam NUM_IMAGES = 4;
    localparam NUM_GROUPS = 8;
    localparam TILES_PER_GROUP = 196;
    localparam NUM_TILES = 1568;

    reg clk, rst_n;

    // FC1 interface
    reg fc1_start;
    reg tile_valid;
    reg signed [127:0] activation_matrix;
    reg signed [127:0] fc1_weight_matrix;
    reg signed [127:0] bias_group;

    wire tile_ready;
    wire fc1_busy;
    wire fc1_done;
    wire [1023:0] hidden_output;

    // FC2 interface
    reg fc2_start;
    reg [255:0] fc2_hidden_input;
    reg [2559:0] fc2_weights;
    reg [319:0] fc2_biases;

    wire [319:0] logits;
    wire fc2_done;

    // FC1 memories
    reg [127:0] activation_mem [0:NUM_TILES-1];
    reg [127:0] fc1_weight_mem [0:NUM_TILES-1];
    reg [31:0] fc1_bias_mem [0:31];
    reg [7:0] expected_hidden_mem [0:127];

    // FC2 memories
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
    // FC1 controller
    // --------------------------------------------

    fc1_controller fc1 (
        .clk(clk),
        .rst_n(rst_n),
        .start(fc1_start),
        .tile_valid(tile_valid),
        .activation_matrix(activation_matrix),
        .weight_matrix(fc1_weight_matrix),
        .bias_group(bias_group),
        .tile_ready(tile_ready),
        .busy(fc1_busy),
        .done(fc1_done),
        .hidden_output(hidden_output)
    );

    // --------------------------------------------
    // FC2 classifier
    // --------------------------------------------

    fc2_classifier fc2 (
        .clk(clk),
        .rst_n(rst_n),
        .start(fc2_start),
        .hidden_activations(fc2_hidden_input),
        .weight_matrix(fc2_weights),
        .bias_vector(fc2_biases),
        .logits(logits),
        .done(fc2_done)
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

        fc1_start = 0;
        fc2_start = 0;
        tile_valid = 0;

        activation_matrix = 0;
        fc1_weight_matrix = 0;
        bias_group = 0;

        fc2_hidden_input = 0;
        fc2_weights = 0;
        fc2_biases = 0;

        passed = 0;
        failed = 0;

        $display("==========================================");
        $display("       END-TO-END MNIST INFERENCE");
        $display("==========================================");

        // ----------------------------------------
        // Load FC1 test vectors
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
            "verification/test_vectors/fc1_hardware/fc1_bias_int32.hex",
            fc1_bias_mem
        );

        $readmemh(
            "verification/test_vectors/fc1_hardware/expected_hidden_int8.hex",
            expected_hidden_mem
        );

        // ----------------------------------------
        // Load FC2 test vectors
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
            fc2_weights[i*8 +: 8] = fc2_weight_mem[i];
        end

        // Pack FC2 biases
        for (i = 0; i < 10; i = i + 1) begin
            fc2_biases[i*32 +: 32] = fc2_bias_mem[i];
        end

        // ----------------------------------------
        // Reset
        // ----------------------------------------

        repeat (3) @(negedge clk);

        rst_n = 1;

        // ----------------------------------------
        // Start FC1
        // ----------------------------------------

        @(negedge clk);
        fc1_start = 1;

        @(negedge clk);
        fc1_start = 0;

        // ----------------------------------------
        // Send all 1568 FC1 tiles
        // ----------------------------------------

        for (
            group_number = 0;
            group_number < NUM_GROUPS;
            group_number = group_number + 1
        ) begin

            $display(
                "\n[FC1] Sending output group %0d",
                group_number
            );

            // Supply four biases for this group
            bias_group = 0;

            for (i = 0; i < 4; i = i + 1) begin
                bias_group[i*32 +: 32] =
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

                // Wait for tile_ready
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
                        "[FAIL] FC1 tile timeout: group=%0d tile=%0d",
                        group_number,
                        tile_number
                    );

                    failed = failed + 1;
                    $finish;

                end

                // Present activation and weight tile
                activation_matrix = activation_mem[test_number];
                fc1_weight_matrix = fc1_weight_mem[test_number];

                tile_valid = 1;

                // Controller accepts at next positive edge
                @(negedge clk);

                tile_valid = 0;

            end

            $display(
                "[FC1] Group %0d completed",
                group_number
            );

        end

        // ----------------------------------------
        // Wait for FC1 completion
        // ----------------------------------------

        cycle_count = 0;

        while (
            fc1_done !== 1'b1 &&
            cycle_count < 500
        ) begin
            @(negedge clk);
            cycle_count = cycle_count + 1;
        end

        if (fc1_done !== 1'b1) begin

            $display("[FAIL] FC1 completion timeout");

            $display("FC1 busy = %b", fc1_busy);

            failed = failed + 1;
            $finish;

        end

        $display("\n[PASS] FC1 completed all four images");

        // ----------------------------------------
        // Verify FC1 hidden activations
        // ----------------------------------------

        $display("\n===== FC1 HIDDEN OUTPUT CHECK =====");

        for (i = 0; i < 128; i = i + 1) begin

            if (
                hidden_output[i*8 +: 8] !==
                expected_hidden_mem[i]
            ) begin

                $display(
                    "[FAIL] Hidden index %0d Expected=%0d Got=%0d",
                    i,
                    $signed(expected_hidden_mem[i]),
                    $signed(hidden_output[i*8 +: 8])
                );

                failed = failed + 1;

            end
            else begin

                passed = passed + 1;

            end

        end

        $display("[INFO] FC1 hidden checks completed");

        // ----------------------------------------
        // FC2: classify all four images
        // ----------------------------------------

        for (
            image = 0;
            image < NUM_IMAGES;
            image = image + 1
        ) begin

            $display(
                "\n===== FC2 IMAGE %0d =====",
                image
            );

            // Select the correct 32 hidden activations
            for (i = 0; i < 32; i = i + 1) begin

                fc2_hidden_input[i*8 +: 8] =
                    hidden_output[
                        (image*32+i)*8 +: 8
                    ];

            end

            // Start FC2
            @(negedge clk);

            fc2_start = 1;

            @(negedge clk);

            fc2_start = 0;

            // Wait for FC2 completion
            cycle_count = 0;

            while (
                fc2_done !== 1'b1 &&
                cycle_count < 500
            ) begin

                @(negedge clk);
                cycle_count = cycle_count + 1;

            end

            if (fc2_done !== 1'b1) begin

                $display(
                    "[FAIL] FC2 timeout on image %0d",
                    image
                );

                failed = failed + 1;
                $finish;

            end

            $display("[PASS] FC2 completed image %0d", image);

            // ------------------------------------
            // Compare all ten logits
            // ------------------------------------

            for (
                neuron = 0;
                neuron < 10;
                neuron = neuron + 1
            ) begin

                if (
                    $signed(logits[neuron*32 +: 32]) !==
                    $signed(
                        expected_logits_mem[image*10+neuron]
                    )
                ) begin

                    $display(
                        "[FAIL] Image %0d Neuron %0d RTL=%0d Python=%0d",
                        image,
                        neuron,
                        $signed(logits[neuron*32 +: 32]),
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
            // Find predicted digit
            // ------------------------------------

            prediction = 0;

            for (
                neuron = 1;
                neuron < 10;
                neuron = neuron + 1
            ) begin

                if (
                    $signed(logits[neuron*32 +: 32]) >
                    $signed(logits[prediction*32 +: 32])
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
                    "[PASS] Image %0d predicted digit %0d",
                    image,
                    prediction
                );

                passed = passed + 1;

            end
            else begin

                $display(
                    "[FAIL] Image %0d Expected digit %0d Got %0d",
                    image,
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
        $display("       END-TO-END RESULTS");
        $display("==========================================");

        $display("Passed: %0d", passed);
        $display("Failed: %0d", failed);
        $display("Expected checks: 172");

        if (passed == 172 && failed == 0) begin

            $display(
                "\n[PASS] FOUR-IMAGE MNIST INFERENCE VERIFIED"
            );

        end
        else begin

            $display(
                "\n[FAIL] END-TO-END TEST FAILED"
            );

        end

        $finish;

    end

    // --------------------------------------------
    // Global simulation timeout
    // --------------------------------------------

    initial begin

        #100000000;

        $display("[FAIL] Simulation timeout");

        $finish;

    end

endmodule
