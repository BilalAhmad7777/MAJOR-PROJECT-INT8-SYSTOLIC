
`timescale 1ns / 1ps

module tb_weight_stationary_array_4x4;

    reg clk, rst_n, load_weights, start;
    reg signed [127:0] activation_matrix, weight_matrix;
    wire signed [511:0] result_matrix;
    wire done;

    reg signed [7:0] A [0:3][0:3];
    reg signed [7:0] B [0:3][0:3];

    integer i, j, k;
    integer expected;
    integer pass_count, fail_count;
    integer test_number;
    integer random_seed;

    weight_stationary_array_4x4 dut (
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

    task run_test;
        input integer test_id;
        begin
            // Build test matrices
            for (i=0; i<4; i=i+1) begin
                for (j=0; j<4; j=j+1) begin
                    case (test_id)
                        1: begin
                            A[i][j] = (i+1)*(j+2);
                            B[i][j] = (i==j) ? 2 : (i+j+1);
                        end

                        2: begin
                            A[i][j] = (i-j)*5 - 3;
                            B[i][j] = (i+j)*3 - 7;
                        end

                        3: begin
                            A[i][j] = (i+j)%2 ? 127 : -128;
                            B[i][j] = (i+j)%2 ? -128 : 127;

                        end
                         4: begin
                            A[i][j] = $random(random_seed);
                            B[i][j] = $random(random_seed);
                         end
                        default: begin
                            A[i][j] = 0;
                            B[i][j] = 0;
                        end
                    endcase
                end
            end

            // Pack matrices
            for (i=0; i<4; i=i+1) begin
                for (j=0; j<4; j=j+1) begin
                    activation_matrix[(i*4+j)*8 +: 8] = A[i][j];
                    weight_matrix[(i*4+j)*8 +: 8] = B[i][j];
                end
            end

            // Load weights and activations
            @(negedge clk);
            load_weights = 1;
            @(negedge clk);
            load_weights = 0;

            // Start computation
            start = 1;
            @(negedge clk);
            start = 0;

            wait(done == 1);
            #1;

            // Compare against golden matrix multiplication
            for (i=0; i<4; i=i+1) begin
                for (j=0; j<4; j=j+1) begin
                    expected = 0;

                    for (k=0; k<4; k=k+1)
                        expected = expected + A[i][k]*B[k][j];

                    if ($signed(result_matrix[(i*4+j)*32 +: 32])
                        === expected) begin
                        $display("[PASS] Test %0d C[%0d][%0d] = %0d",
                            test_id, i, j, expected);
                        pass_count = pass_count + 1;
                    end
                    else begin
                        $display("[FAIL] Test %0d C[%0d][%0d] Expected=%0d Got=%0d",
                            test_id, i, j, expected,
                            $signed(result_matrix[(i*4+j)*32 +: 32]));
                        fail_count = fail_count + 1;
                    end
                end
            end

            $display("Test %0d completed.", test_id);
        end
    endtask

    initial begin
        clk = 0;
        rst_n = 0;
        load_weights = 0;
        start = 0;
        activation_matrix = 0;
        weight_matrix = 0;
        pass_count = 0;
        fail_count = 0;
        random_seed = 32'h13579BDF;

        repeat(2) @(posedge clk);
        @(negedge clk);
        rst_n = 1;

        run_test(1);
        run_test(2);
        run_test(3);
        for (test_number = 0; test_number < 20; test_number = test_number + 1)
             run_test(4);

        $display("--------------------------------");
        $display("FINAL RESULT: %0d PASS / %0d FAIL",
            pass_count, fail_count);
        $display("--------------------------------");

        $finish;
    end

endmodule
