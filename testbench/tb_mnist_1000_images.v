
`timescale 1ns / 1ps

module tb_mnist_1000_images;

   localparam NUM_IMAGES = 1000;
localparam BATCH_SIZE = 4;
localparam NUM_BATCHES = 250;

localparam NUM_GROUPS = 8;
localparam TILES_PER_GROUP = 196;
localparam TILES_PER_BATCH = 1568;
localparam TOTAL_TILES = 392000;

localparam TOTAL_LOGITS = 10000;
localparam EXPECTED_CHECKS = 11250;

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

    // ------------------------------------------------
    // Memories
    // ------------------------------------------------

    reg [127:0] activation_mem [0:TOTAL_TILES-1];
    reg [127:0] fc1_weight_mem [0:TOTAL_TILES-1];

    reg [31:0] fc1_bias_mem [0:31];

    reg [7:0] fc2_weight_mem [0:319];
    reg [31:0] fc2_bias_mem [0:9];

    reg [31:0] expected_logits_mem [0:TOTAL_LOGITS-1];
    reg [31:0] labels_mem [0:NUM_IMAGES-1];
    reg [31:0] python_predictions_mem [0:NUM_IMAGES-1];

    // ------------------------------------------------
    // Counters
    // ------------------------------------------------

    integer batch_number;
    integer group_number;
    integer tile_number;
    integer test_number;

    integer image;
    integer global_image;
    integer neuron;

    integer cycle_count;
    integer passed;
    integer failed;

    integer prediction;
    integer expected_prediction;

    integer i;
    integer global_tile;
    integer correct_classifications;

           integer labels_file;
integer predictions_file;
integer scan_result;

    // ------------------------------------------------
    // DUT
    // ------------------------------------------------

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

    // ------------------------------------------------
    // Clock
    // ------------------------------------------------

    initial clk = 0;
    always #5 clk = ~clk;

    // ------------------------------------------------
    // Main test
    // ------------------------------------------------

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

        $display("================================================");
        $display("       1000-IMAGE MNIST RTL VERIFICATION");
        $display("================================================");

        // --------------------------------------------
        // Load FC1 vectors
        // --------------------------------------------

        $display("\n[LOAD] Loading FC1 vectors...");

        $readmemh(
            "verification/test_vectors/mnist_1000/activation.hex",
            activation_mem
        );

        $readmemh(
            "verification/test_vectors/mnist_1000/weights.hex",
            fc1_weight_mem
        );

        $readmemh(
            "verification/test_vectors/mnist_1000/fc1_bias_int32.hex",
            fc1_bias_mem
        );

        // --------------------------------------------
        // Load FC2 vectors
        // --------------------------------------------

        $display("[LOAD] Loading FC2 parameters...");

        $readmemh(
            "verification/test_vectors/mnist_1000/fc2_weights_int8.hex",
            fc2_weight_mem
        );

        $readmemh(
            "verification/test_vectors/mnist_1000/fc2_bias_int32.hex",
            fc2_bias_mem
        );

        $readmemh(
            "verification/test_vectors/mnist_1000/expected_logits_int32.hex",
            expected_logits_mem
        );
        // Load decimal labels and Python predictions using fscanf
labels_file = $fopen(
    "verification/test_vectors/mnist_1000/labels.txt", "r"
);

predictions_file = $fopen(
    "verification/test_vectors/mnist_1000/predictions_python.txt", "r"
);

if (labels_file == 0 || predictions_file == 0) begin
    $display("[FAIL] Could not open labels or predictions file.");
    $finish;
end

for (i = 0; i < NUM_IMAGES; i = i + 1) begin
    scan_result = $fscanf(labels_file, "%d", labels_mem[i]);

    if (scan_result != 1) begin
        $display("[FAIL] Could not read label %0d", i);
        $finish;
    end

    scan_result = $fscanf(
        predictions_file, "%d", python_predictions_mem[i]
    );

    if (scan_result != 1) begin
        $display("[FAIL] Could not read Python prediction %0d", i);
        $finish;
    end
end

$fclose(labels_file);
$fclose(predictions_file);

correct_classifications = 0;



        // --------------------------------------------
        // Pack FC2 weights
        // --------------------------------------------

        for (i = 0; i < 320; i = i + 1) begin
            fc2_weight_matrix[i*8 +: 8] = fc2_weight_mem[i];
        end

        // --------------------------------------------
        // Pack FC2 biases
        // --------------------------------------------

        for (i = 0; i < 10; i = i + 1) begin
            fc2_bias_vector[i*32 +: 32] = fc2_bias_mem[i];
        end

        // --------------------------------------------
        // Reset
        // --------------------------------------------

        repeat (3) @(negedge clk);
        rst_n = 1;

        $display("\n[PASS] Reset released.");
        $display("[INFO] Starting 25 inference batches.");

        // ============================================
        // PROCESS 25 BATCHES
        // ============================================

        for (
            batch_number = 0;
            batch_number < NUM_BATCHES;
            batch_number = batch_number + 1
        ) begin

            $display("\n");
            $display("==============================================");
            $display("          BATCH %0d / %0d",
                     batch_number + 1, NUM_BATCHES);
            $display("==============================================");

            // ----------------------------------------
            // Start accelerator
            // ----------------------------------------

            @(negedge clk);
            start = 1;

            @(negedge clk);
            start = 0;

            $display("[TOP] Inference started.");

            // ----------------------------------------
            // Stream all 1568 FC1 tiles
            // ----------------------------------------

            for (
                group_number = 0;
                group_number < NUM_GROUPS;
                group_number = group_number + 1
            ) begin

                // Supply four biases for this group
                fc1_bias_group = 0;

                for (i = 0; i < 4; i = i + 1) begin
                    fc1_bias_group[i*32 +: 32] =
                        fc1_bias_mem[group_number*4+i];
                end

                for (
                    tile_number = 0;
                    tile_number < TILES_PER_GROUP;
                    tile_number = tile_number + 1
                ) begin

                    global_tile =
                        batch_number * TILES_PER_BATCH
                        + group_number * TILES_PER_GROUP
                        + tile_number;

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
                            "[FAIL] Tile timeout: batch=%0d group=%0d tile=%0d",
                            batch_number,
                            group_number,
                            tile_number
                        );

                        failed = failed + 1;
                        $finish;

                    end

                    // Present the current tile
                    activation_matrix = activation_mem[global_tile];
                    fc1_weight_matrix = fc1_weight_mem[global_tile];
                    tile_valid = 1;

                    @(negedge clk);
                    tile_valid = 0;

                end

            end

            $display("[FC1] All 1568 tiles submitted.");

            // ----------------------------------------
            // Wait for complete inference
            // ----------------------------------------

            $display("[TOP] Waiting for FC1 and FC2...");

            cycle_count = 0;

            while (
                done !== 1'b1 &&
                cycle_count < 100000
            ) begin
                @(negedge clk);
                cycle_count = cycle_count + 1;
            end

            if (done !== 1'b1) begin

                $display("[FAIL] Batch %0d completion timeout.",
                         batch_number + 1);

                $display("busy     = %b", busy);
                $display("fc1_busy = %b", fc1_busy);

                failed = failed + 1;
                $finish;

            end

            // One check per completed batch
            passed = passed + 1;

            $display(
                "[PASS] Batch %0d completed in %0d cycles.",
                batch_number + 1,
                cycle_count
            );

            // ========================================
            // COMPARE 40 LOGITS
            // ========================================

            for (
                image = 0;
                image < BATCH_SIZE;
                image = image + 1
            ) begin

                global_image =
                    batch_number * BATCH_SIZE + image;

                for (
                    neuron = 0;
                    neuron < 10;
                    neuron = neuron + 1
                ) begin

                    if (
                        $signed(
                            batch_logits[
                                image*320 + neuron*32 +: 32
                            ]
                        ) !==
                        $signed(
                            expected_logits_mem[
                                global_image*10 + neuron
                            ]
                        )
                    ) begin

                        $display(
                            "[FAIL] Image %0d Neuron %0d RTL=%0d Python=%0d",
                            global_image,
                            neuron,
                            $signed(
                                batch_logits[
                                    image*320 + neuron*32 +: 32
                                ]
                            ),
                            $signed(
                                expected_logits_mem[
                                    global_image*10 + neuron
                                ]
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
                            batch_logits[
                                image*320 + neuron*32 +: 32
                            ]
                        ) >
                        $signed(
                            batch_logits[
                                image*320 + prediction*32 +: 32
                            ]
                        )
                    ) begin

                        prediction = neuron;

                    end

                end

                expected_prediction =
    python_predictions_mem[global_image];
    if (prediction == labels_mem[global_image]) begin
    correct_classifications = correct_classifications + 1;
end

                if (prediction == expected_prediction) begin

                    $display(
                        "[PASS] Image %0d | Expected=%0d | Predicted=%0d",
                        global_image,
                        expected_prediction,
                        prediction
                    );

                    passed = passed + 1;

                end
                else begin

                    $display(
                        "[FAIL] Image %0d | Expected=%0d | Predicted=%0d",
                        global_image,
                        expected_prediction,
                        prediction
                    );

                    failed = failed + 1;

                end

            end

            // ----------------------------------------
            // Ensure accelerator returned to idle
            // ----------------------------------------

            @(negedge clk);

            if (busy !== 1'b0) begin
                $display(
                    "[FAIL] Batch %0d: accelerator not idle.",
                    batch_number + 1
                );
                failed = failed + 1;
            end

        end

        // ============================================
        // FINAL RESULTS
        // ============================================

        $display("\n");
        $display("================================================");
        $display("          1000-IMAGE VERIFICATION RESULTS");
        $display("================================================");

        $display("Images processed       : %0d", NUM_IMAGES);
        $display("Batches completed      : %0d", NUM_BATCHES);
        $display("FC1 tile transactions  : %0d", TOTAL_TILES);
        $display("Logit comparisons      : %0d", TOTAL_LOGITS);
        $display("Prediction comparisons : %0d", NUM_IMAGES);
      $display(
    "Classification accuracy : %0d / %0d (%0d.%0d%%)",
    correct_classifications,
    NUM_IMAGES,
    (correct_classifications * 100) / NUM_IMAGES,
    (correct_classifications * 1000 / NUM_IMAGES) % 10
);

        $display("-----------------------------------------------");
        $display("Passed checks          : %0d", passed);
        $display("Failed checks          : %0d", failed);
        $display("Expected checks        : %0d", EXPECTED_CHECKS);
        $display("-----------------------------------------------");

        if (
            passed == EXPECTED_CHECKS &&
            failed == 0
        ) begin

            $display("\n[PASS] ALL 1000 IMAGES VERIFIED!");
            $display("[PASS] RTL LOGITS MATCH PYTHON REFERENCE!");
            $display("[PASS] ALL RTL PREDICTIONS MATCH PYTHON!");
$display(
    "[INFO] Classification accuracy: %0d.%0d%%",
    (correct_classifications * 100) / NUM_IMAGES,
    (correct_classifications * 1000 / NUM_IMAGES) % 10
);

        end
        else begin

            $display("\n[FAIL] 1000-IMAGE VERIFICATION FAILED.");
            $display("Please inspect the mismatch messages above.");

        end

        $display("================================================");

        $finish;

    end

    // ------------------------------------------------
    // Global simulation timeout
    // ------------------------------------------------

    initial begin

        #100000000;

        $display("[FAIL] Global simulation timeout.");
        $finish;

    end

endmodule
