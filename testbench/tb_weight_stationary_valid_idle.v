`timescale 1ns / 1ps

module tb_weight_stationary_valid_idle;

    reg clk = 0;
    reg rst_n = 0;
    reg load_weights = 0;
    reg start = 0;
    reg signed [127:0] activation_matrix = 0;
    reg signed [127:0] weight_matrix = 0;

    wire signed [511:0] result_matrix;
    wire done;

    integer i, j;
    integer high_count = 0;
    integer low_count = 0;

    weight_stationary_array_4x4_fsm dut (
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
        $display("\n===== ARRAY VALID SIGNAL IDLE TEST =====");

        repeat (3) @(negedge clk);
        rst_n = 1;

        // Allow all PE valid registers to settle.
        repeat (6) @(negedge clk);

        if (dut.state !== 0)
            $display("[FAIL] FSM is not in IDLE");

        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                if (dut.pe_valid_wire[i][j] === 1'b1)
                    high_count = high_count + 1;
                else
                    low_count = low_count + 1;
            end
        end

        $display("PE valid inputs HIGH during IDLE: %0d", high_count);
        $display("PE valid inputs LOW during IDLE:  %0d", low_count);

        if (high_count == 0 && low_count == 16)
            $display("[PASS] All 16 PE inputs are invalid during IDLE");
        else
            $display("[FAIL] PE valid inputs are not all LOW during IDLE");

        $finish;
    end

endmodule
