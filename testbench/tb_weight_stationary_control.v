
`timescale 1ns / 1ps

module tb_weight_stationary_control;

    reg clk;
    reg rst_n;
    reg load_weights;
    reg start;

    reg signed [127:0] activation_matrix;
    reg signed [127:0] weight_matrix;

    wire signed [511:0] result_matrix;
    wire done;

    integer i, j;
    integer pass_count, fail_count;

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

    initial begin
        clk = 0;
        rst_n = 0;
        load_weights = 0;
        start = 0;
        activation_matrix = 0;
        weight_matrix = 0;
        pass_count = 0;
        fail_count = 0;

        // TEST 1: RESET
        repeat (2) @(posedge clk);
        #1;

        if (done == 0 && result_matrix == 0) begin
            $display("[PASS] Reset clears outputs");
            pass_count = pass_count + 1;
        end
        else begin
            $display("[FAIL] Reset behavior");
            fail_count = fail_count + 1;
        end

        @(negedge clk);
        rst_n = 1;

        // Prepare original activation matrix and identity weights
        for (i=0; i<4; i=i+1) begin
            for (j=0; j<4; j=j+1) begin
                activation_matrix[(i*4+j)*8 +: 8] =
                    (i*4+j+1);

                if (i == j)
                    weight_matrix[(i*4+j)*8 +: 8] = 1;
                else
                    weight_matrix[(i*4+j)*8 +: 8] = 0;
            end
        end

        // Load original matrices
        @(negedge clk);
        load_weights = 1;

        @(negedge clk);
        load_weights = 0;

        // TEST 2: START AND COMPLETE
        start = 1;

        @(negedge clk);
        start = 0;

        // TEST 3: ATTEMPT START AND LOAD WHILE BUSY
        // These requests must be ignored.
        @(negedge clk);
        start = 1;
        load_weights = 1;

        // Change input matrices to different values.
        // They must not replace the already loaded matrices.
        activation_matrix = {16{8'd9}};
        weight_matrix = {16{8'd9}};

        @(negedge clk);
        start = 0;
        load_weights = 0;

        wait(done == 1);
        #1;

        // Check original result: A multiplied by identity = A
        for (i=0; i<4; i=i+1) begin
            for (j=0; j<4; j=j+1) begin
                if ($signed(result_matrix[(i*4+j)*32 +: 32])
                    == (i*4+j+1)) begin
                    pass_count = pass_count + 1;
                end
                else begin
                    $display("[FAIL] Busy protection C[%0d][%0d]",
                        i, j);
                    fail_count = fail_count + 1;
                end
            end
        end

        if (fail_count == 0) begin
            $display("[PASS] Busy engine ignores new start/load requests");
            pass_count = pass_count + 1;
        end

        // TEST 4: DONE MUST BE A ONE-CYCLE PULSE
        @(posedge clk);
        #1;

        if (done == 0) begin
            $display("[PASS] Done returns LOW after completion");
            pass_count = pass_count + 1;
        end
        else begin
            $display("[FAIL] Done pulse duration");
            fail_count = fail_count + 1;
        end

        // TEST 5: RESET DURING COMPUTATION
        @(negedge clk);
        start = 1;

        @(negedge clk);
        start = 0;

        // Engine is now processing. Assert reset.
        rst_n = 0;

        @(posedge clk);
        #1;

        if (done == 0 && result_matrix == 0) begin
            $display("[PASS] Reset aborts computation and clears outputs");
            pass_count = pass_count + 1;
        end
        else begin
            $display("[FAIL] Reset during computation");
            fail_count = fail_count + 1;
        end

        // Ensure no delayed completion occurs
        repeat (6) begin
            @(posedge clk);
            #1;
            if (done != 0) begin
                $display("[FAIL] Unexpected done after reset");
                fail_count = fail_count + 1;
            end
        end

        if (done == 0) begin
            $display("[PASS] No completion after reset");
            pass_count = pass_count + 1;
        end

        $display("--------------------------------");
        $display("CONTROL TESTS: %0d PASS / %0d FAIL",
            pass_count, fail_count);
        $display("--------------------------------");

        $finish;
    end

endmodule
