
`timescale 1ns / 1ps

module tb_weight_stationary_array_4x4_fsm;

    reg clk = 0;
    reg rst_n = 0;
    reg load_weights = 0;
    reg start = 0;

    reg signed [127:0] activation_matrix = 0;
    reg signed [127:0] weight_matrix = 0;

    wire signed [511:0] result_matrix;
    wire done;

    integer A [0:3][0:3];
    integer B [0:3][0:3];
    integer GOLD [0:3][0:3];

    integer i, j, k;
    integer test_number;
    integer random_seed = 32'h13579BDF;

    integer pass_count = 0;
    integer fail_count = 0;

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

    // Run one complete matrix multiplication test
    task run_test;
        input integer mode;
        begin

            // PASS 1: Prepare input matrices
            for (i = 0; i < 4; i = i + 1) begin
                for (j = 0; j < 4; j = j + 1) begin

                    case (mode)

                        // Positive values
                        1: begin
                            A[i][j] = (i+1)*(j+2);
                            B[i][j] = (i == j) ? 2 : (i+j+1);
                        end

                        // Mixed positive and negative values
                        2: begin
                            A[i][j] = (i-j)*5-3;
                            B[i][j] = (i+j)*3-7;
                        end

                        // INT8 boundary values
                        3: begin
                            A[i][j] = ((i+j)%2 == 0) ? -128 : 127;
                            B[i][j] = ((i+j)%2 == 0) ? 127 : -128;
                        end

                        // Deterministic random values
                        4: begin
                            A[i][j] = $random(random_seed) % 128;
                            B[i][j] = $random(random_seed) % 128;
                        end

                        default: begin
                            A[i][j] = 0;
                            B[i][j] = 0;
                        end

                    endcase

                    activation_matrix[(i*4+j)*8 +: 8] = A[i][j];
                    weight_matrix[(i*4+j)*8 +: 8] = B[i][j];

                end
            end

            // PASS 2: Calculate golden outputs
            for (i = 0; i < 4; i = i + 1) begin
                for (j = 0; j < 4; j = j + 1) begin

                    GOLD[i][j] = 0;

                    for (k = 0; k < 4; k = k + 1) begin
                        GOLD[i][j] = GOLD[i][j] +
                                     A[i][k] * B[k][j];
                    end

                end
            end

            // LOAD MATRICES
            @(negedge clk);
            load_weights = 1;

            @(negedge clk);
            load_weights = 0;

            // START COMPUTATION
            @(negedge clk);
            start = 1;

            @(negedge clk);
            start = 0;

            // WAIT FOR COMPLETION
            wait(done == 1);
            #1;

            // VERIFY ALL 16 OUTPUTS
            for (i = 0; i < 4; i = i + 1) begin
                for (j = 0; j < 4; j = j + 1) begin

                    if ($signed(result_matrix[(i*4+j)*32 +: 32])
                        === GOLD[i][j]) begin

                        pass_count = pass_count + 1;

                    end
                    else begin

                        fail_count = fail_count + 1;

                        $display(
                            "[FAIL] Test %0d C[%0d][%0d] expected %0d got %0d",
                            mode, i, j, GOLD[i][j],
                            $signed(result_matrix[(i*4+j)*32 +: 32])
                        );

                    end

                end
            end

            $display("[INFO] Matrix test mode %0d completed", mode);

            // Verify done returns LOW after next rising edge
            @(posedge clk);
            @(negedge clk);

            if (done == 0) begin
                pass_count = pass_count + 1;
            end
            else begin
                $display("[FAIL] Done did not return LOW");
                fail_count = fail_count + 1;
            end

        end
    endtask

    initial begin

        $display("\n===== COMPREHENSIVE FSM TEST =====");

        // RESET
        repeat (2) @(negedge clk);
        rst_n = 1;

        // TEST 1: Start before load must be ignored
        @(negedge clk);
        start = 1;

        @(negedge clk);
        start = 0;

        if (done == 0 && dut.state == 0) begin
            $display("[PASS] Start before load ignored");
            pass_count = pass_count + 1;
        end
        else begin
            $display("[FAIL] Start before load protection");
            fail_count = fail_count + 1;
        end

        // TEST 2: Positive matrix
        run_test(1);

        // TEST 3: Mixed positive and negative matrix
        run_test(2);

        // TEST 4: INT8 boundary matrix
        run_test(3);

        // TEST 5: Ten deterministic random matrices
        for (test_number = 0; test_number < 10;
             test_number = test_number + 1) begin
            run_test(4);
        end

        $display("\n--------------------------------");
        $display("FSM VERIFICATION RESULTS");
        $display("PASS: %0d", pass_count);
        $display("FAIL: %0d", fail_count);
        $display("--------------------------------");

        $finish;

    end

endmodule
