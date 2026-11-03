
`timescale 1ns/1ps

module tb_matrix_demo;

    reg clk = 0;
    reg rst_n = 0;
    reg load_weights = 0;
    reg start = 0;

    reg signed [127:0] activation_matrix;
    reg signed [127:0] weight_matrix;

    wire signed [511:0] result_matrix;
    wire done;

    integer i, j;
    integer passed = 0;
    integer failed = 0;
    integer result_value;

    always #5 clk = ~clk;

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

    // Store matrices in row-major order.
    reg signed [7:0] A [0:15];
    reg signed [7:0] B [0:15];
    integer expected [0:15];

    initial begin

        $display("\n====================================");
        $display(" LIVE 4x4 INT8 MATRIX MULTIPLICATION");
        $display("====================================");

        // Matrix A
        A[0]=1; A[1]=2; A[2]=3; A[3]=4;
        A[4]=5; A[5]=6; A[6]=7; A[7]=8;
        A[8]=1; A[9]=1; A[10]=1; A[11]=1;
        A[12]=2; A[13]=2; A[14]=2; A[15]=2;

        // Matrix B = Identity
        B[0]=1; B[1]=0; B[2]=0; B[3]=0;
        B[4]=0; B[5]=1; B[6]=0; B[7]=0;
        B[8]=0; B[9]=0; B[10]=1; B[11]=0;
        B[12]=0; B[13]=0; B[14]=0; B[15]=1;

        // Calculate software reference result.
        for (i=0; i<4; i=i+1) begin
            for (j=0; j<4; j=j+1) begin
                expected[i*4+j] = 0;
                for (integer k=0; k<4; k=k+1)
                    expected[i*4+j] =
                        expected[i*4+j] +
                        A[i*4+k] * B[k*4+j];
            end
        end

        // Pack matrices into DUT input buses.
        activation_matrix = 0;
        weight_matrix = 0;

        for (i=0; i<16; i=i+1) begin
            activation_matrix[i*8 +: 8] = A[i];
            weight_matrix[i*8 +: 8] = B[i];
        end

        repeat(2) @(negedge clk);
        rst_n = 1;

       // Load weights.
@(negedge clk);
load_weights = 1;

@(negedge clk);
load_weights = 0;

// Allow the FSM to return to IDLE.
@(negedge clk);

// Start computation.
start = 1;
@(negedge clk);
start = 0;

$display("[DEMO] Weights loaded. Computation started.");

        // Wait for completion.
        wait(done == 1);
        #1;

        $display("\nMATRIX A:");
        for (i=0; i<4; i=i+1) begin
            for (j=0; j<4; j=j+1)
                $write("%6d", A[i*4+j]);
            $display("");
        end

        $display("\nMATRIX B:");
        for (i=0; i<4; i=i+1) begin
            for (j=0; j<4; j=j+1)
                $write("%6d", B[i*4+j]);
            $display("");
        end

        $display("\nVERILOG OUTPUT:");
        for (i=0; i<4; i=i+1) begin
            for (j=0; j<4; j=j+1) begin
                result_value = $signed(result_matrix[(i*4+j)*32 +: 32]);
                $write("%6d", result_value);

                if (result_value == expected[i*4+j])
                    passed = passed + 1;
                else begin
                    failed = failed + 1;
                    $display("\nMismatch at row %0d col %0d", i, j);
                end
            end
            $display("");
        end

        $display("\nEXPECTED OUTPUT:");
        for (i=0; i<4; i=i+1) begin
            for (j=0; j<4; j=j+1)
                $write("%6d", expected[i*4+j]);
            $display("");
        end

        $display("\nPASSED: %0d", passed);
        $display("FAILED: %0d", failed);

        if (failed == 0)
            $display("\nMATRIX MULTIPLICATION VERIFIED!");

        $finish;
    end

    initial begin
        #1000;
        $display("ERROR: Simulation timeout");
        $finish;
    end

endmodule
