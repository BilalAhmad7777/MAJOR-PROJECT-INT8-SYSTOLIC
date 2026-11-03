`timescale 1ns / 1ps

module tb_weight_stationary_load_start;

    reg clk = 0;
    reg rst_n = 0;
    reg load_weights = 0;
    reg start = 0;

    reg signed [127:0] activation_matrix = 0;
    reg signed [127:0] weight_matrix = 0;

    wire signed [511:0] result_matrix;
    wire done;

    integer i, j;
    integer pass_count = 0;
    integer fail_count = 0;
    integer timeout_count = 0;

    weight_stationary_array_4x4 dut (
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
        $display("\n===== IMMEDIATE LOAD-START TIMING TEST =====");

        // Activation matrix with known values; identity weights.
        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                activation_matrix[(i*4+j)*8 +: 8] = i*4+j+1;
                weight_matrix[(i*4+j)*8 +: 8] = (i == j) ? 1 : 0;
            end
        end

        repeat (2) @(negedge clk);
        rst_n = 1;

        // Assert LOAD and START together.
        @(negedge clk);
        load_weights = 1;
        start = 1;

        // Release LOAD, but hold START for one more cycle.
        @(negedge clk);
        load_weights = 0;
        start = 1;

        @(negedge clk);
        start = 0;

        // Wait for completion, with a timeout.
        while (done !== 1'b1 && timeout_count < 30) begin
            @(negedge clk);
            timeout_count = timeout_count + 1;
        end

        if (done !== 1'b1) begin
            $display("[FAIL] Computation did not complete");
            fail_count = fail_count + 1;
        end else begin
            $display("[PASS] Delayed start was accepted");

            for (i = 0; i < 4; i = i + 1) begin
                for (j = 0; j < 4; j = j + 1) begin
                    if ($signed(result_matrix[(i*4+j)*32 +: 32])
                        === (i*4+j+1)) begin
                        pass_count = pass_count + 1;
                    end else begin
                        $display("[FAIL] C[%0d][%0d]: expected %0d, got %0d",
                            i, j, i*4+j+1,
                            $signed(result_matrix[(i*4+j)*32 +: 32]));
                        fail_count = fail_count + 1;
                    end
                end
            end
        end

        $display("--------------------------------");
        $display("RESULT: %0d PASS / %0d FAIL",
            pass_count, fail_count);
        $display("--------------------------------");

        $finish;
    end

endmodule
