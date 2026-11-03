`timescale 1ns / 1ps

module tb_fc1_postprocess_boundary;

    reg signed [31:0] accumulator;
    reg signed [31:0] bias;
    wire signed [7:0] activation_int8;

    integer pass_count = 0;
    integer fail_count = 0;

    fc1_postprocess dut (
        .accumulator(accumulator),
        .bias(bias),
        .activation_int8(activation_int8)
    );

    task check;
        input signed [31:0] acc;
        input signed [31:0] b;
        input signed [7:0] expected;
        begin
            accumulator = acc;
            bias = b;
            #2;
            if (activation_int8 === expected) begin
                $display("[PASS] acc=%0d bias=%0d output=%0d",
                         acc, b, activation_int8);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] acc=%0d bias=%0d expected=%0d got=%0d",
                         acc, b, expected, activation_int8);
                fail_count = fail_count + 1;
            end
        end
    endtask

    initial begin
        $display("===== FC1 POSTPROCESS BOUNDARY AUDIT =====");

        check(0, 0, 0);
        check(-2147483648, 0, 0);
        check(0, -2147483648, 0);
        check(-2147483648, -2147483648, 0);

        check(2147483647, 0, 127);
        check(0, 2147483647, 127);
        check(2147483647, 2147483647, 127);

        check(2147483647, -2147483648, 0);
        check(-2147483648, 2147483647, 0);
        check(2147483647, -2147483647, 0);

        check(100000, -100000, 0);
        check(100000, -99999, 0);
        check(100000, -90000, 3);
        check(100000, 0, 30);

        check(10000, 5000, 4);
        check(0, 500000, 127);
        check(500000, -500000, 0);

        $display("-----------------------------------------");
        $display("BOUNDARY AUDIT: %0d PASS / %0d FAIL",
                 pass_count, fail_count);
        $finish;
    end

endmodule
