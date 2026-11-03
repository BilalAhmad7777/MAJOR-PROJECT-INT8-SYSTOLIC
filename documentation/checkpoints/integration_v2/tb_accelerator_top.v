
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

    integer random_seed;
    integer test_counter;
    integer wait_cycles;

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


    // ==========================================
    // CALCULATE SOFTWARE GOLDEN OUTPUT
    // ==========================================

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

                    // Match RTL rounding
                    rounded = sum;

                    if (quant_shift != 0)
                        rounded = rounded +
                                  (1 << (quant_shift - 1));

                    // Scaling
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


    // ==========================================
    // RUN ONE MATRIX TEST
    // ==========================================

    task run_test;
        input integer test_mode;

        begin

            // Generate input matrices
            for (i = 0; i < 4; i = i + 1) begin
                for (j = 0; j < 4; j = j + 1) begin

                    case (test_mode)

                        // Test 1: Positive values
                        1: begin
                            A[i][j] = (i + 1) * (j + 2);
                            B[i][j] = (i == j) ? 2 : (i + j + 1);
                        end

                        // Test 2: Mixed positive and negative values
                        2: begin
                            A[i][j] = (i - j) * 5 - 3;
                            B[i][j] = (i + j) * 3 - 7;
                        end

                        // Test 3: INT8 boundary values
                        3: begin
                            A[i][j] = ((i + j) % 2) ? -128 : 127;
                            B[i][j] = ((i + j) % 2) ? 127 : -128;
                        end

                        // Test 4: Random signed INT8 values
                        4: begin

                            A[i][j] = $random(random_seed) % 256;

                            if (A[i][j] > 127)
                                A[i][j] = A[i][j] - 256;
                            else if (A[i][j] < -128)
                                A[i][j] = A[i][j] + 256;

                            B[i][j] = $random(random_seed) % 256;

                            if (B[i][j] > 127)
                                B[i][j] = B[i][j] - 256;
                            else if (B[i][j] < -128)
                                B[i][j] = B[i][j] + 256;

                        end

                        default: begin
                            A[i][j] = 0;
                            B[i][j] = 0;
                        end

                    endcase

                    // Pack matrices into input buses
                    activation_matrix[(i*4+j)*8 +: 8] = A[i][j];
                    weight_matrix[(i*4+j)*8 +: 8] = B[i][j];

                end
            end


            // Calculate expected results
            calculate_golden();


            // LOAD MATRICES
            @(negedge clk);
            load_weights = 1;

            @(negedge clk);
            load_weights = 0;

            // Allow FSM to complete LOAD state
            @(negedge clk);


            // START COMPUTATION
            start = 1;

            @(negedge clk);
            start = 0;


            // WAIT FOR COMPLETION WITH TIMEOUT
            wait_cycles = 0;

            while ((done !== 1'b1) && (wait_cycles < 20)) begin
                @(negedge clk);
                wait_cycles = wait_cycles + 1;
            end

            if (done !== 1'b1) begin

                $display(
                    "[ERROR] TIMEOUT: Test mode %0d did not complete",
                    test_mode
                );

                fail_count = fail_count + 1;

                $display("Simulation stopped due to timeout.");
                $finish;

            end


            // Allow output signals to settle
            #1;


            // CHECK ALL 16 OUTPUTS
            for (i = 0; i < 4; i = i + 1) begin
                for (j = 0; j < 4; j = j + 1) begin

                    if ($signed(output_matrix[(i*4+j)*8 +: 8])
                        === golden[i][j]) begin

                        pass_count = pass_count + 1;

                    end
                    else begin

                        $display(
                            "[FAIL] Mode=%0d Output[%0d][%0d] Expected=%0d Got=%0d",
                            test_mode,
                            i,
                            j,
                            golden[i][j],
                            $signed(output_matrix[(i*4+j)*8 +: 8])
                        );

                        fail_count = fail_count + 1;

                    end

                end
            end


            $display(
                "[INFO] Test mode %0d completed",
                test_mode
            );


            // Wait for controller to return to IDLE
            @(negedge clk);
            @(negedge clk);

        end
    endtask


    // ==========================================
    // MAIN TEST SEQUENCE
    // ==========================================

    initial begin

        $display("\n===== ACCELERATOR RANDOMIZED INTEGRATION TEST =====");

        random_seed = 32'h13579BDF;

        // Reset
        repeat (2) @(negedge clk);
        rst_n = 1;


        // TEST 1: POSITIVE MATRIX
        $display("\n[TEST 1] Positive matrix");

        quant_shift = 0;
        run_test(1);


        // TEST 2: MIXED SIGN MATRIX
        $display("\n[TEST 2] Mixed-sign matrix");

        quant_shift = 2;
        run_test(2);


        // TEST 3: INT8 BOUNDARY MATRIX
        $display("\n[TEST 3] INT8 boundary matrix");

        quant_shift = 4;
        run_test(3);


        // TESTS 4-23: RANDOMIZED MATRICES
        $display("\n[TEST 4-23] Randomized matrices");

        quant_shift = 2;

        for (test_counter = 0;
             test_counter < 20;
             test_counter = test_counter + 1) begin

            $display(
                "\n[INFO] Random matrix %0d / 20",
                test_counter + 1
            );

            run_test(4);

        end


        // FINAL RESULTS
        $display("\n================================");
        $display("ACCELERATOR VERIFICATION RESULTS");
        $display("PASS: %0d", pass_count);
        $display("FAIL: %0d", fail_count);
        $display("================================");

        if (fail_count == 0)
            $display("OVERALL RESULT: ALL TESTS PASSED");
        else
            $display("OVERALL RESULT: VERIFICATION FAILED");

        $finish;

    end

endmodule
