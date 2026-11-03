`timescale 1ns / 1ps

module tb_relu;

    reg signed [31:0] data_in;
    wire signed [31:0] data_out;

    integer pass_count = 0;
    integer fail_count = 0;

    relu #(
        .DATA_WIDTH(32)
    ) dut (
        .data_in(data_in),
        .data_out(data_out)
    );

    task check;
        input signed [31:0] test_input;
        input signed [31:0] expected;
        begin
            data_in = test_input;
            #10;

            if (data_out === expected) begin
                $display("[PASS] Input: %0d | Output: %0d",
                         data_in, data_out);
                pass_count = pass_count + 1;
            end
            else begin
                $display("[FAIL] Input: %0d | Expected: %0d | Got: %0d",
                         data_in, expected, data_out);
                fail_count = fail_count + 1;
            end
        end
    endtask

    initial begin
        $display("\n===== ReLU VERIFICATION =====");

        check(100, 100);
        check(25, 25);
        check(0, 0);
        check(-10, 0);
        check(-500, 0);
        check(2147483647, 2147483647);
        check(-2147483647, 0);
        check(-2147483648, 0);

        $display("\n-------------------------------");
        $display("ReLU RESULTS");
        $display("PASS: %0d", pass_count);
        $display("FAIL: %0d", fail_count);
        $display("-------------------------------");

        $finish;
    end

endmodule
