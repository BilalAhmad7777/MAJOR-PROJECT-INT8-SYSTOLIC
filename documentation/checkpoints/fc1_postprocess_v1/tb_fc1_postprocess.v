`timescale 1ns / 1ps

module tb_fc1_postprocess;

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
            #1;

            if (activation_int8 !== expected) begin
                $display(
                    "[FAIL] acc=%0d bias=%0d expected=%0d got=%0d",
                    acc, b, expected, activation_int8
                );
                fail_count = fail_count + 1;
            end else begin
                $display(
                    "[PASS] acc=%0d bias=%0d output=%0d",
                    acc, b, activation_int8
                );
                pass_count = pass_count + 1;
            end
        end
    endtask

    initial begin
        $display("===== FC1 POST-PROCESSING TEST =====");

        check(0, 0, 0);
        check(-1000, 0, 0);
        check(100000, 0, 30);
        check(100000, -100000, 0);
        check(1000000, 0, 127);
        check(0, 500000, 127);
        check(500000, -500000, 0);
        check(10000, 5000, 4);

        $display("\n===== RESULTS =====");
        $display("Passed: %0d", pass_count);
        $display("Failed: %0d", fail_count);

        if (fail_count == 0)
            $display("[PASS] FC1 POST-PROCESSING VERIFIED");
        else
            $display("[FAIL] POST-PROCESSING ERRORS");

        $finish;
    end

endmodule
