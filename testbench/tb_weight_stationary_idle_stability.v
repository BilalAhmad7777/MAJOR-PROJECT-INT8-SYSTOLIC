`timescale 1ns / 1ps

module tb_weight_stationary_idle_stability;

    reg clk = 0;
    reg rst_n = 0;
    reg load_weights = 0;
    reg start = 0;

    reg signed [127:0] activation_matrix = 0;
    reg signed [127:0] weight_matrix = 0;

    wire signed [511:0] result_matrix;
    wire done;

    reg [511:0] saved_result;
    integer i, j;
    integer pass_count = 0;
    integer fail_count = 0;
    integer timeout_count;

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
        $display("\n===== IDLE RESULT STABILITY TEST =====");

        // Identity weights; activations 1 through 16.
        for (i = 0; i < 4; i = i + 1)
            for (j = 0; j < 4; j = j + 1) begin
                activation_matrix[(i*4+j)*8 +: 8] = i*4+j+1;
                weight_matrix[(i*4+j)*8 +: 8] =
                    (i == j) ? 1 : 0;
            end

        repeat (2) @(negedge clk);
        rst_n = 1;

        @(negedge clk);
        load_weights = 1;
        @(negedge clk);
        load_weights = 0;

        @(negedge clk);
        start = 1;
        @(negedge clk);
        start = 0;

        timeout_count = 0;
        while (done !== 1'b1 && timeout_count < 30) begin
            @(negedge clk);
            timeout_count = timeout_count + 1;
        end

        if (done !== 1'b1) begin
            $display("[FAIL] Computation timeout");
            fail_count = fail_count + 1;
        end else begin
            saved_result = result_matrix;
            $display("[INFO] Result captured after completion");

            // Check output stability over 10 idle clock cycles.
            repeat (10) begin
                @(negedge clk);
                if (result_matrix === saved_result) begin
                    pass_count = pass_count + 1;
                end else begin
                    $display("[FAIL] Result changed during idle");
                    fail_count = fail_count + 1;
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
