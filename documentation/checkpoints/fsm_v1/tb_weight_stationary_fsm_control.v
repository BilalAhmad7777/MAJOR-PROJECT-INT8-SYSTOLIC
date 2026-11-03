
`timescale 1ns / 1ps

module tb_weight_stationary_fsm_control;

    reg clk = 0;
    reg rst_n = 0;
    reg load_weights = 0;
    reg start = 0;

    reg signed [127:0] activation_matrix = 0;
    reg signed [127:0] weight_matrix = 0;

    wire signed [511:0] result_matrix;
    wire done;

    integer i, j;
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

    task report;
        input condition;
        input [8*100-1:0] message;
        begin
            if (condition) begin
                $display("[PASS] %0s", message);
                pass_count = pass_count + 1;
            end
            else begin
                $display("[FAIL] %0s", message);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task load_identity_test_matrix;
    begin
        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin

                activation_matrix[(i*4+j)*8 +: 8] =
                    (i+1)*(j+2);

                weight_matrix[(i*4+j)*8 +: 8] =
                    (i == j) ? 1 : 0;

            end
        end

        @(negedge clk);
        load_weights = 1;

        @(negedge clk);
        load_weights = 0;

        // Allow the FSM to execute LOAD and return to IDLE
        @(negedge clk);

       end
     endtask
    task start_computation;
        begin
            @(negedge clk);
            start = 1;

            @(negedge clk);
            start = 0;
        end
    endtask

    task check_identity_result;
        reg all_correct;
        begin
            all_correct = 1;

            for (i = 0; i < 4; i = i + 1) begin
                for (j = 0; j < 4; j = j + 1) begin

                    if ($signed(result_matrix[(i*4+j)*32 +: 32])
                        !== (i+1)*(j+2))
                        all_correct = 0;

                end
            end

            report(all_correct, "Original computation result is correct");
        end
    endtask

    initial begin

        $display("\n===== FSM CONTROL ROBUSTNESS TEST =====");

        // INITIAL RESET
        repeat (2) @(negedge clk);
        rst_n = 1;

        // LOAD ORIGINAL MATRIX
        load_identity_test_matrix();

        report(dut.state == 0, "Controller returns to IDLE after loading");

        // TEST 1: START COMPUTATION
        start_computation();

        @(negedge clk);
        report(dut.state == 2, "Controller enters COMPUTE state");

        // TEST 2: SEND NEW COMMANDS WHILE COMPUTING
        start = 1;
        load_weights = 1;

        // Change external input data to zeros.
        // If incorrectly loaded, the result would change.
        activation_matrix = 0;
        weight_matrix = 0;

        repeat (2) @(negedge clk);

        start = 0;
        load_weights = 0;

        report(dut.state == 2,
               "Controller remains in COMPUTE despite new commands");

        // WAIT FOR ORIGINAL COMPUTATION TO FINISH
        wait(done == 1);
        #1;

        check_identity_result();

        // TEST 3: DONE PULSE
        @(posedge clk);
        @(negedge clk);

        report(done == 0, "Done returns LOW after one cycle");

        report(dut.state == 0, "Controller returns to IDLE");

        // TEST 4: RESET DURING COMPUTATION
        start_computation();

        @(negedge clk);
        report(dut.state == 2, "Second computation started");

        // Assert reset while computation is active
        rst_n = 0;

        @(posedge clk);
        #1;

        report(dut.state == 0, "Reset returns controller to IDLE");
        report(done == 0, "Reset clears done signal");

        report(result_matrix == 0,
               "Reset clears result matrix");

        // Release reset
        @(negedge clk);
        rst_n = 1;

        // Ensure no delayed completion appears
        repeat (6) begin
            @(negedge clk);
            report(done == 0, "No delayed completion after reset");
        end

        $display("\n--------------------------------");
        $display("CONTROL VERIFICATION RESULTS");
        $display("PASS: %0d", pass_count);
        $display("FAIL: %0d", fail_count);
        $display("--------------------------------");

        $finish;

    end

endmodule
