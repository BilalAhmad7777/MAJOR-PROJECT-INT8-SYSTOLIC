`timescale 1ns / 1ps

module tb_accelerator_top;

    reg clk = 0;
    reg rst_n = 0;

    reg load_weights = 0;
    reg start = 0;

    reg signed [127:0] activation_matrix = 0;
    reg signed [127:0] weight_matrix = 0;

    reg [5:0] quant_shift = 0;

    wire signed [127:0] output_matrix;
    wire done;

    integer A [0:3][0:3];
    integer B [0:3][0:3];
    integer golden [0:3][0:3];

    integer i, j, k;
    integer sum;
    integer expected;
    integer rounded;
    integer pass_count = 0;
    integer fail_count = 0;

    accelerator_top dut (
        .clk(clk),
        .rst_n(rst_n),
        .load_weights(load_weights),
        .start(start),
        .activation_matrix(activation_matrix),
        .weight_matrix(weight_matrix),
        .quant_shift(quant_shift),
        .output_matrix(output_matrix),
        .done(done)
    );

    always #5 clk = ~clk;

    task calculate_golden;
        begin
            for (i = 0; i < 4; i = i + 1) begin
                for (j = 0; j < 4; j = j + 1) begin
                    sum = 0;

                    for (k = 0; k < 4; k = k + 1)
                        sum = sum + A[i][k] * B[k][j];

                    // ReLU
                    if (sum < 0)
                        sum = 0;

                    // Match RTL rounding and scaling
                    rounded = sum;

                    if (quant_shift != 0)
                        rounded = rounded +
                                  (1 << (quant_shift - 1));

                    expected = rounded >>> quant_shift;

                    // INT8 saturation
                    if (expected > 127)
                        expected = 127;
                    else if (expected < -128)
                        expected = -128;

                    golden[i][j] = expected;
                end
            end
        end
    endtask

    task run_test;
        input integer test_mode;
        begin
            // Prepare matrices
            for (i = 0; i < 4; i = i + 1) begin
                for (j = 0; j < 4; j = j + 1) begin

                    case (test_mode)

                        1: begin
                            A[i][j] = (i + 1) * (j + 2);
                            B[i][j] = (i == j) ? 2 : (i + j + 1);
                        end

                        2: begin
                            A[i][j] = (i - j) * 5 - 3;
                            B[i][j] = (i + j) * 3 - 7;
                        end

                        3: begin
                            A[i][j] = (i + j) % 2 ? -128 : 127;
                            B[i][j] = (i + j) % 2 ? 127 : -128;
                        end

                    endcase

                    activation_matrix[(i*4+j)*8 +: 8] = A[i][j];
                    weight_matrix[(i*4+j)*8 +: 8] = B[i][j];

                end
            end

            // Calculate expected output
            calculate_golden();

            // Load matrices
            @(negedge clk);
            load_weights = 1;

            @(negedge clk);
            load_weights = 0;

            @(negedge clk);

            // Start computation
            start = 1;

            @(negedge clk);
            start = 0;

            // Wait for completion
            wait(done == 1);
            #1;

            // Check all 16 outputs
            for (i = 0; i < 4; i = i + 1) begin
                for (j = 0; j < 4; j = j + 1) begin

                    if ($signed(output_matrix[(i*4+j)*8 +: 8])
                        === golden[i][j]) begin

                        pass_count = pass_count + 1;

                    end
                    else begin

                        $display(
                            "[FAIL] Test %0d Output[%0d][%0d] Expected=%0d Got=%0d",
                            test_mode, i, j, golden[i][j],
                            $signed(output_matrix[(i*4+j)*8 +: 8])
                        );

                        fail_count = fail_count + 1;

                    end
                end
            end

            $display("[INFO] Integration test %0d completed",
                     test_mode);

            // Wait for controller to return to IDLE
            @(negedge clk);
            @(negedge clk);

        end
    endtask

    initial begin
        $display("\n===== ACCELERATOR INTEGRATION TEST =====");

        repeat (2) @(negedge clk);
        rst_n = 1;

        quant_shift = 0;
        run_test(1);

        quant_shift = 2;
        run_test(2);

        quant_shift = 4;
        run_test(3);

        $display("\n--------------------------------");
        $display("INTEGRATION RESULTS");
        $display("PASS: %0d", pass_count);
        $display("FAIL: %0d", fail_count);
        $display("--------------------------------");

        $finish;
    end

endmodule
