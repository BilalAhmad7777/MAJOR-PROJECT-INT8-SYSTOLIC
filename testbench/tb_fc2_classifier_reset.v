
`timescale 1ns / 1ps

module tb_fc2_classifier_reset;

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

    integer i;
    integer passed, failed;
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
        $display(" FC2 RESET AND RECOVERY TEST");
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

        // Prepare image 0.
        for (i = 0; i < 32; i = i + 1)
            hidden_activations[i*8 +: 8] = hidden_mem[i];

        for (i = 0; i < 320; i = i + 1)
            weight_matrix[i*8 +: 8] = weight_mem[i];

        for (i = 0; i < 10; i = i + 1)
            bias_vector[i*32 +: 32] = bias_mem[i];

        // Initial reset.
        repeat (3) @(negedge clk);
        rst_n = 1;

        // Start first inference.
        @(negedge clk);
        start = 1;
        @(negedge clk);
        start = 0;

        // Allow ten MAC cycles to execute.
        repeat (10) @(negedge clk);

        $display("[INFO] FC2 computation interrupted after 10 MAC cycles.");

        // Assert reset.
        rst_n = 0;

        repeat (3) @(negedge clk);

        // Verify done is cleared.
        if (done !== 1'b0) begin
            $display("[FAIL] done was not cleared");
            failed = failed + 1;
        end else begin
            $display("[PASS] done cleared");
            passed = passed + 1;
        end

        // Verify all logits are cleared.
        if (logits !== 320'b0) begin
            $display("[FAIL] logits were not cleared");
            failed = failed + 1;
        end else begin
            $display("[PASS] all logits cleared");
            passed = passed + 1;
        end

        // Release reset.
        @(negedge clk);
        rst_n = 1;

        // Start fresh inference.
        @(negedge clk);
        start = 1;
        @(negedge clk);
        start = 0;

        $display("[INFO] Fresh FC2 inference started.");

        // Wait for completion.
        cycle_count = 0;

        while (done !== 1'b1 && cycle_count < 500) begin
            @(negedge clk);
            cycle_count = cycle_count + 1;
        end

        if (done !== 1'b1) begin
            $display("[FAIL] FC2 completion timeout");
            failed = failed + 1;
        end else begin
            $display("[PASS] FC2 inference completed after reset");
            passed = passed + 1;
        end

        // Compare all ten logits with Python.
        for (i = 0; i < 10; i = i + 1) begin
            if ($signed(logits[i*32 +: 32]) !==
                $signed(expected_mem[i])) begin

                $display(
                    "[FAIL] Neuron %0d | RTL=%0d | Python=%0d",
                    i,
                    $signed(logits[i*32 +: 32]),
                    $signed(expected_mem[i])
                );

                failed = failed + 1;

            end else begin

                $display(
                    "[PASS] Neuron %0d | RTL=%0d | Python=%0d",
                    i,
                    $signed(logits[i*32 +: 32]),
                    $signed(expected_mem[i])
                );

                passed = passed + 1;
            end
        end

        $display("");
        $display("========================================");
        $display(" FC2 RESET RECOVERY RESULTS");
        $display("========================================");
        $display("Passed: %0d", passed);
        $display("Failed: %0d", failed);
        $display("Logit comparisons: 10");

        if (failed == 0)
            $display("[PASS] FC2 RESET RECOVERY VERIFIED");
        else
            $display("[FAIL] FC2 RESET RECOVERY FAILED");

        $finish;
    end

    initial begin
        #100000;
        $display("[FAIL] Simulation timeout");
        $finish;
    end

endmodule
