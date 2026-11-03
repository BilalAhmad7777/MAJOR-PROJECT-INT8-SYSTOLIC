
`timescale 1ns / 1ps

module tb_fc2_classifier_repeat;

    reg clk, rst_n, start;
    reg [255:0] hidden_activations;
    reg [2559:0] weight_matrix;
    reg [319:0] bias_vector;

    wire [319:0] logits;
    wire done;

    reg [7:0] hidden_mem [0:127];
    reg [7:0] weight_mem [0:319];
    reg [31:0] bias_mem [0:9];
    reg [31:0] expected_mem [0:39];

    integer i, image, passed, failed;
    integer cycle_count;

    fc2_classifier dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .hidden_activations(hidden_activations),
        .weight_matrix(weight_matrix),
        .bias_vector(bias_vector),
        .logits(logits),
        .done(done)
    );

    always #5 clk = ~clk;

    task run_image;
        input integer image_index;
        begin
            // Load the selected image's hidden activations.
            for (i = 0; i < 32; i = i + 1)
                hidden_activations[i*8 +: 8] =
                    hidden_mem[image_index*32+i];

            // Start inference.
            @(negedge clk);
            start = 1;

            @(negedge clk);
            start = 0;

            // Wait for completion.
            cycle_count = 0;

            while (done !== 1'b1 && cycle_count < 500) begin
                @(negedge clk);
                cycle_count = cycle_count + 1;
            end

            if (done !== 1'b1) begin
                $display("[FAIL] Image %0d completion timeout",
                         image_index);
                failed = failed + 1;
            end else begin
                $display("[PASS] Image %0d inference completed",
                         image_index);
                passed = passed + 1;
            end

            // Compare all ten logits.
            for (i = 0; i < 10; i = i + 1) begin

                if ($signed(logits[i*32 +: 32]) !==
                    $signed(expected_mem[image_index*10+i])) begin

                    $display(
                        "[FAIL] Image=%0d Neuron=%0d RTL=%0d Python=%0d",
                        image_index,
                        i,
                        $signed(logits[i*32 +: 32]),
                        $signed(expected_mem[image_index*10+i])
                    );

                    failed = failed + 1;

                end else begin

                    $display(
                        "[PASS] Image=%0d Neuron=%0d RTL=%0d Python=%0d",
                        image_index,
                        i,
                        $signed(logits[i*32 +: 32]),
                        $signed(expected_mem[image_index*10+i])
                    );

                    passed = passed + 1;
                end
            end

            // Check that FC2 returned to IDLE.
            @(negedge clk);

            if (done !== 1'b0) begin
                $display("[FAIL] Image %0d done did not clear",
                         image_index);
                failed = failed + 1;
            end else begin
                $display("[PASS] Image %0d done cleared",
                         image_index);
                passed = passed + 1;
            end
        end
    endtask

    initial begin
        clk = 0;
        rst_n = 0;
        start = 0;
        hidden_activations = 0;
        weight_matrix = 0;
        bias_vector = 0;
        passed = 0;
        failed = 0;
        cycle_count = 0;

        $display("========================================");
        $display(" FC2 REPEATED INFERENCE TEST");
        $display("========================================");

        // Load Python-generated vectors.
        $readmemh(
            "verification/test_vectors/fc2_hardware/fc2_hidden_int8.hex",
            hidden_mem
        );

        $readmemh(
            "verification/test_vectors/fc2_hardware/fc2_weights_int8.hex",
            weight_mem
        );

        $readmemh(
            "verification/test_vectors/fc2_hardware/fc2_bias_int32.hex",
            bias_mem
        );

        $readmemh(
            "verification/test_vectors/fc2_hardware/fc2_expected_logits_int32.hex",
            expected_mem
        );

        // Weights and biases are shared by both images.
        for (i = 0; i < 320; i = i + 1)
            weight_matrix[i*8 +: 8] = weight_mem[i];

        for (i = 0; i < 10; i = i + 1)
            bias_vector[i*32 +: 32] = bias_mem[i];

        // Initial reset only.
        repeat (3) @(negedge clk);
        rst_n = 1;

        // First inference.
        $display("");
        $display("[INFO] Starting image 0");
        run_image(0);

        // Second inference without reset.
        $display("");
        $display("[INFO] Starting image 1 WITHOUT RESET");
        run_image(1);

        $display("");
        $display("========================================");
        $display(" FC2 REPEATED INFERENCE RESULTS");
        $display("========================================");
        $display("Passed: %0d", passed);
        $display("Failed: %0d", failed);
        $display("Logit comparisons: 20");
        $display("Inference batches: 2");

        if (failed == 0)
            $display("[PASS] FC2 REPEATED INFERENCE VERIFIED");
        else
            $display("[FAIL] FC2 REPEATED INFERENCE FAILED");

        $finish;
    end

    initial begin
        #100000;
        $display("[FAIL] Simulation timeout");
        $finish;
    end

endmodule
