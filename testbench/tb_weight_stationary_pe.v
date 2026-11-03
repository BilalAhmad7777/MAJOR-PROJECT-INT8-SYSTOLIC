`timescale 1ns / 1ps

module tb_weight_stationary_pe;

    reg clk;
    reg rst_n;

    reg load_weight;
    reg signed [7:0] weight_in;

    reg valid_in;
    reg signed [7:0] activation_in;

    reg signed [31:0] partial_sum_in;

    wire signed [7:0] activation_out;
    wire signed [31:0] partial_sum_out;
    wire valid_out;

    integer pass_count = 0;
    integer fail_count = 0;

    weight_stationary_pe dut (
        .clk(clk),
        .rst_n(rst_n),
        .load_weight(load_weight),
        .weight_in(weight_in),
        .valid_in(valid_in),
        .activation_in(activation_in),
        .partial_sum_in(partial_sum_in),
        .activation_out(activation_out),
        .partial_sum_out(partial_sum_out),
        .valid_out(valid_out)
    );

    always #5 clk = ~clk;

    task check_result;
        input signed [31:0] expected;
        input [8*40-1:0] test_name;
        begin
            if (partial_sum_out === expected && valid_out === 1'b1) begin
                $display("[PASS] %0s | Output = %0d",
                         test_name, partial_sum_out);
                pass_count = pass_count + 1;
            end
            else begin
                $display("[FAIL] %0s | Expected = %0d, Got = %0d",
                         test_name, expected, partial_sum_out);
                fail_count = fail_count + 1;
            end
        end
    endtask

    initial begin

        clk = 0;
        rst_n = 0;
        load_weight = 0;
        weight_in = 0;
        valid_in = 0;
        activation_in = 0;
        partial_sum_in = 0;

        // Reset
        repeat (2) @(posedge clk);
        @(negedge clk);
        rst_n = 1;

        // Test 1: 5 * 3 + 0 = 15
        load_weight = 1;
        weight_in = 3;
        @(negedge clk);
        load_weight = 0;
        activation_in = 5;
        partial_sum_in = 0;
        valid_in = 1;
        @(posedge clk);
        #1;
        check_result(15, "Positive multiplication");

        // Test 2: -4 * 5 + 10 = -10
        @(negedge clk);
        activation_in = -4;
        partial_sum_in = 10;
        @(posedge clk);
        #1;
        check_result(-2, "Negative activation");

        // Test 3: -128 * -128 + 0 = 16384
        @(negedge clk);
        load_weight = 1;
        weight_in = -128;
        valid_in = 0;
        @(negedge clk);
        load_weight = 0;
        activation_in = -128;
        partial_sum_in = 0;
        valid_in = 1;
        @(posedge clk);
        #1;
        check_result(16384, "INT8 boundary values");

        // Test 4: activation forwarding
        if (activation_out === -128) begin
            $display("[PASS] Activation forwarding");
            pass_count = pass_count + 1;
        end
        else begin
            $display("[FAIL] Activation forwarding");
            fail_count = fail_count + 1;
        end

        // Test 5: invalid input
        @(negedge clk);
        valid_in = 0;
        @(posedge clk);
        #1;
        if (valid_out === 0) begin
            $display("[PASS] Invalid cycle");
            pass_count = pass_count + 1;
        end
        else begin
            $display("[FAIL] Invalid cycle");
            fail_count = fail_count + 1;
        end

        $display("--------------------------------");
        $display("RESULT: %0d PASS / %0d FAIL",
                 pass_count, fail_count);
        $display("--------------------------------");

        $finish;
    end

endmodule
