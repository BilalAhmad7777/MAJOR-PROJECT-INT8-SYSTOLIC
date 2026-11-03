`timescale 1ns / 1ps

module tb_weight_stationary_valid_chain;

    reg clk = 0;
    reg rst_n = 0;
    reg load_weights = 0;
    reg start = 0;
    reg signed [127:0] activation_matrix = 0;
    reg signed [127:0] weight_matrix = 0;

    wire signed [511:0] result_matrix;
    wire done;

    integer i, j;
    integer checks = 0;
    integer failures = 0;
    integer cycles = 0;

    reg previous_valid [0:3][0:3];

    // Testbench reference valid deskew registers.
    reg c0_v0 = 0, c0_v1 = 0, c0_v2 = 0;
    reg c1_v0 = 0, c1_v1 = 0;
    reg c2_v0 = 0;
    reg c3_v0 = 0;

    reg [3:0] aligned_valid;

    always @(posedge clk) begin
        if (!rst_n) begin
            c0_v0 <= 0; c0_v1 <= 0; c0_v2 <= 0;
            c1_v0 <= 0; c1_v1 <= 0;
            c2_v0 <= 0; c3_v0 <= 0;
        end else begin
            c0_v0 <= dut.pe_valid_wire[3][1];
            c0_v1 <= c0_v0;
            c0_v2 <= c0_v1;

            c1_v0 <= dut.pe_valid_wire[3][2];
            c1_v1 <= c1_v0;

            c2_v0 <= dut.pe_valid_wire[3][3];
            c3_v0 <= dut.pe_valid_wire[3][4];
        end
    end

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

    // Check every PE's registered valid_out against
    // the previous cycle's valid_in.
    always @(negedge clk) begin
        if (rst_n) begin
            cycles = cycles + 1;

            if (dut.state == 3 && dut.cycle_cnt >= 7 &&
                dut.cycle_cnt <= 10)
                $display(
                    "[CAPTURE] row=%0d reference_valid={%b,%b,%b,%b}",
                    dut.cycle_cnt - 7,
                    dut.pe_valid_wire[3][4],
                    c2_v0, c1_v1, c0_v2
                );

            for (integer r = 0; r < 4; r = r + 1) begin
                for (integer c = 0; c < 4; c = c + 1) begin
                    if (cycles > 1) begin
                        checks = checks + 1;

                        if (dut.pe_valid_wire[r][c+1] !==
                            previous_valid[r][c]) begin
                            failures = failures + 1;
                            $display(
                                "[FAIL] row=%0d col=%0d cycle=%0d",
                                r, c, cycles
                            );
                        end
                    end

                    previous_valid[r][c] =
                        dut.pe_valid_wire[r][c];
                end
            end
        end
    end

    initial begin
        $display("===== SYSTOLIC VALID CHAIN TEST =====");

        for (i = 0; i < 4; i = i + 1)
            for (j = 0; j < 4; j = j + 1)
                previous_valid[i][j] = 0;

        repeat (3) @(negedge clk);
        rst_n = 1;

        activation_matrix = {
            8'sd16, 8'sd15, 8'sd14, 8'sd13,
            8'sd12, 8'sd11, 8'sd10, 8'sd9,
            8'sd8,  8'sd7,  8'sd6,  8'sd5,
            8'sd4,  8'sd3,  8'sd2,  8'sd1
        };

        weight_matrix = {
            8'sd1, 8'sd0, 8'sd0, 8'sd0,
            8'sd0, 8'sd1, 8'sd0, 8'sd0,
            8'sd0, 8'sd0, 8'sd1, 8'sd0,
            8'sd0, 8'sd0, 8'sd0, 8'sd1
        };

        @(negedge clk);
        load_weights = 1;
        @(negedge clk);
        load_weights = 0;

        wait (dut.state == 0);
        @(negedge clk);
        start = 1;
        @(negedge clk);
        start = 0;

        wait (done == 1);
        repeat (8) @(negedge clk);

        $display("Cycles monitored: %0d", cycles);
        $display("Valid propagation checks: %0d", checks);
        $display("Failures: %0d", failures);

        if (failures == 0)
            $display("[PASS] All PE valid chains propagate correctly");
        else
            $display("[FAIL] Valid propagation errors detected");

        $finish;
    end

    initial begin
        #2000;
        $display("[FAIL] Test timeout");
        $finish;
    end

endmodule
