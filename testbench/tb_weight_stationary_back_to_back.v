`timescale 1ns / 1ps

module tb_weight_stationary_back_to_back;

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

    task check_result;
        input integer base;
        input [8*20-1:0] label;
        begin
            for (i = 0; i < 4; i = i + 1) begin
                for (j = 0; j < 4; j = j + 1) begin
                    if ($signed(result_matrix[(i*4+j)*32 +: 32])
                        === base + i*4+j) begin
                        pass_count = pass_count + 1;
                    end else begin
                        $display("[FAIL] %0s C[%0d][%0d] expected %0d got %0d",
                            label, i, j, base+i*4+j,
                            $signed(result_matrix[(i*4+j)*32 +: 32]));
                        fail_count = fail_count + 1;
                    end
                end
            end
        end
    endtask

    task wait_done;
        begin
            timeout_count = 0;
            while (done !== 1'b1 && timeout_count < 30) begin
                @(negedge clk);
                timeout_count = timeout_count + 1;
            end
            if (done !== 1'b1) begin
                $display("[FAIL] Computation timeout");
                fail_count = fail_count + 1;
            end
        end
    endtask

    initial begin
        $display("\n===== BACK-TO-BACK TRANSACTION TEST =====");

        // Load identity weights exactly once.
        for (i = 0; i < 4; i = i + 1)
            for (j = 0; j < 4; j = j + 1)
                weight_matrix[(i*4+j)*8 +: 8] =
                    (i == j) ? 1 : 0;

        // First activation matrix: values 1 through 16.
        for (i = 0; i < 4; i = i + 1)
            for (j = 0; j < 4; j = j + 1)
                activation_matrix[(i*4+j)*8 +: 8] = i*4+j+1;

        repeat (2) @(negedge clk);
        rst_n = 1;

        @(negedge clk);
        load_weights = 1;
        @(negedge clk);
        load_weights = 0;

        // First transaction.
        @(negedge clk);
        start = 1;
        @(negedge clk);
        start = 0;

        wait_done;
        if (done === 1'b1)
            check_result(1, "Transaction 1");

        // Wait until controller has returned to IDLE.
        @(negedge clk);
        @(negedge clk);

        // Change activations only. Do not reload weights.
        for (i = 0; i < 4; i = i + 1)
            for (j = 0; j < 4; j = j + 1)
                activation_matrix[(i*4+j)*8 +: 8] = i*4+j+21;

        // Second transaction using the previously loaded weights.
        @(negedge clk);
        start = 1;
        @(negedge clk);
        start = 0;

        wait_done;
        if (done === 1'b1)
            check_result(21, "Transaction 2");

        $display("--------------------------------");
        $display("RESULT: %0d PASS / %0d FAIL",
            pass_count, fail_count);
        $display("--------------------------------");

        $finish;
    end

endmodule
