`timescale 1ns/1ps

module tb_weight_stationary_utilization;

    reg clk = 0;
    reg rst_n = 0;
    reg load_weights = 0;
    reg start = 0;

    reg signed [127:0] activation_matrix = 0;
    reg signed [127:0] weight_matrix = 0;

    wire signed [511:0] result_matrix;
    wire done;

    integer i, j;
    integer active_count;
    integer total_macs = 0;
    integer peak_macs = 0;
    integer active_cycles = 0;
    integer pe_count [0:3][0:3];

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

    // Sample PE valid inputs before the next active clock edge.
    // Each valid input represents one MAC operation at that edge.
    always @(negedge clk) begin
        if (rst_n && (dut.state == 3'd2 || dut.state == 3'd3)) begin
            active_count = 0;

            for (i = 0; i < 4; i = i + 1) begin
                for (j = 0; j < 4; j = j + 1) begin
                    if (dut.pe_valid_wire[i][j]) begin
                        active_count = active_count + 1;
                        pe_count[i][j] = pe_count[i][j] + 1;
                    end
                end
            end

            if (active_count > 0) begin
                total_macs = total_macs + active_count;
                active_cycles = active_cycles + 1;

                if (active_count > peak_macs)
                    peak_macs = active_count;
            end

            $display(
                "Cycle=%0t | Active PEs=%0d",
                $time, active_count
            );
        end
    end

    initial begin

        for (i = 0; i < 4; i = i + 1)
            for (j = 0; j < 4; j = j + 1)
                pe_count[i][j] = 0;

        // Identity activation and weight matrices.
        activation_matrix = {
            8'd0,8'd0,8'd0,8'd1,
            8'd0,8'd0,8'd1,8'd0,
            8'd0,8'd1,8'd0,8'd0,
            8'd1,8'd0,8'd0,8'd0
        };

        weight_matrix = {
            8'd0,8'd0,8'd0,8'd1,
            8'd0,8'd0,8'd1,8'd0,
            8'd0,8'd1,8'd0,8'd0,
            8'd1,8'd0,8'd0,8'd0
        };

        repeat (3) @(negedge clk);
        rst_n = 1;

        @(negedge clk);
        load_weights = 1;

        @(negedge clk);
        load_weights = 0;

        repeat (2) @(negedge clk);

        start = 1;
        @(negedge clk);
        start = 0;

        wait(done == 1);
        @(negedge clk);

        $display("\n================================");
        $display("PE UTILIZATION RESULTS");
        $display("Total MAC operations: %0d", total_macs);
        $display("Peak simultaneous MACs: %0d", peak_macs);
        $display("Active cycles: %0d", active_cycles);
        $display("================================");

        $display("\nPER-PE MAC COUNT:");

        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1)
                $write("%0d ", pe_count[i][j]);
            $display("");
        end

        if (total_macs == 64)
            $display("\n[PASS] Expected 64 MAC operations");
        else
            $display("\n[FAIL] Expected 64 MAC operations");

        $finish;
    end

    initial begin
        #2000;
        $display("[TIMEOUT] Simulation did not finish");
        $finish;
    end

endmodule
