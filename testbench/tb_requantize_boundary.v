`timescale 1ns / 1ps

module tb_requantize_boundary;

    reg signed [31:0] data_in;
    reg [5:0] shift;
    wire signed [7:0] data_out;

    integer pass_count = 0;
    integer fail_count = 0;

    requantize dut (
        .data_in(data_in),
        .shift(shift),
        .data_out(data_out)
    );

    task check;
        input signed [31:0] value;
        input [5:0] shift_amount;
        input signed [7:0] expected;
        begin
            data_in = value;
            shift = shift_amount;
            #10;
            if (data_out === expected) begin
                $display("[PASS] input=%0d shift=%0d output=%0d",
                         value, shift_amount, data_out);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] input=%0d shift=%0d expected=%0d got=%0d",
                         value, shift_amount, expected, data_out);
                fail_count = fail_count + 1;
            end
        end
    endtask

    initial begin
        $display("===== REQUANTIZATION BOUNDARY AUDIT =====");

        // Zero and small shifts
        check(0, 0, 0);
        check(127, 0, 127);
        check(-128, 0, -128);
        check(3, 1, 2);
        check(-3, 1, -2);

        // Positive and negative exact rounding ties
        check(1, 1, 1);
        check(-1, 1, -1);
        check(5, 1, 3);
        check(-5, 1, -3);
        check(6, 2, 2);
        check(-6, 2, -2);

        // Values immediately around INT8 saturation
        check(2031, 4, 127);
        check(2032, 4, 127);
        check(-2040, 4, -128);
        check(-2048, 4, -128);

        // INT32 extremes
        check(2147483647, 0, 127);
        check(-2147483647, 0, -128);
        check(-2147483648, 0, -128);

        // Large shift values
        check(2147483647, 30, 2);
        check(-2147483648, 30, -2);
        check(2147483647, 31, 1);
        check(-2147483648, 31, -1);
        check(2147483647, 32, 0);
        check(-2147483648, 32, -1);
        check(2147483647, 33, 0);
        check(-2147483648, 33, 0);
        check(2147483647, 63, 0);
        check(-2147483648, 63, 0);

        $display("-----------------------------------------");
        $display("BOUNDARY AUDIT: %0d PASS / %0d FAIL",
                 pass_count, fail_count);
        $finish;
    end

endmodule
