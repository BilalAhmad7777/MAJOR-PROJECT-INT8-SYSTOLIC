`timescale 1ns / 1ps

module tb_requantize;

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
                $display("[PASS] Input=%0d Shift=%0d Output=%0d",
                         value, shift_amount, data_out);
                pass_count = pass_count + 1;
            end
            else begin
                $display("[FAIL] Input=%0d Shift=%0d Expected=%0d Got=%0d",
                         value, shift_amount, expected, data_out);
                fail_count = fail_count + 1;
            end
        end
    endtask

    initial begin
        $display("\n===== REQUANTIZATION VERIFICATION =====");

        // Basic scaling
        check(1600, 4, 100);
        check(800, 4, 50);
        check(-800, 4, -50);

        // Zero shift
        check(50, 0, 50);
        check(-50, 0, -50);

        // Positive and negative rounding
        check(24, 4, 2);
        check(40, 4, 3);
        check(-24, 4, -2);
        check(-40, 4, -3);

        // Positive saturation
        check(10000, 0, 127);
        check(2147483647, 0, 127);

        // Negative saturation
        check(-10000, 0, -128);
        check(-2147483647, 0, -128);

        // INT8 boundaries
        check(2032, 4, 127);
        check(-2048, 4, -128);

        $display("\n-------------------------------");
        $display("REQUANTIZATION RESULTS");
        $display("PASS: %0d", pass_count);
        $display("FAIL: %0d", fail_count);
        $display("-------------------------------");

        $finish;
    end

endmodule
