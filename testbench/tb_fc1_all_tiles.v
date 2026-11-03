`timescale 1ns / 1ps

module tb_fc1_all_tiles;

    localparam NUM_TILES = 1568;

    reg clk = 0;
    reg rst_n = 0;
    reg load_weights = 0;
    reg start = 0;

    reg signed [127:0] activation_matrix = 0;
    reg signed [127:0] weight_matrix = 0;

    wire signed [511:0] result_matrix;
    wire done;

    reg [127:0] activation_mem [0:NUM_TILES-1];
    reg [127:0] weight_mem [0:NUM_TILES-1];
    reg [511:0] expected_mem [0:NUM_TILES-1];

    integer tile;
    integer lane;
    integer pass_count = 0;
    integer fail_count = 0;
    integer total_outputs = 0;

    reg signed [31:0] expected_value;
    reg signed [31:0] actual_value;

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

    initial begin

        $display("\n========================================");
        $display(" FC1 EXHAUSTIVE RTL TILE VERIFICATION");
        $display("========================================");

        // Load generated Python vectors
        $readmemh(
            "verification/test_vectors/fc1_hardware/activation.hex",
            activation_mem
        );

        $readmemh(
            "verification/test_vectors/fc1_hardware/weights.hex",
            weight_mem
        );

        $readmemh(
            "verification/test_vectors/fc1_hardware/expected_partial_sums.hex",
            expected_mem
        );

        // Reset
        repeat (3) @(negedge clk);
        rst_n = 1;

        // Process all 1568 tiles
        for (tile = 0; tile < NUM_TILES; tile = tile + 1) begin

            // Present input matrices
            @(negedge clk);
            activation_matrix = activation_mem[tile];
            weight_matrix = weight_mem[tile];
            load_weights = 1;

            // Load weights and activations
            @(negedge clk);
            load_weights = 0;

            // Start computation
            @(negedge clk);
            start = 1;

            @(negedge clk);
            start = 0;

            // Wait for actual RTL completion
            wait(done == 1);
            #1;

            // Compare all 16 INT32 outputs
            for (lane = 0; lane < 16; lane = lane + 1) begin

                expected_value =
                    $signed(expected_mem[tile][lane*32 +: 32]);

                actual_value =
                    $signed(result_matrix[lane*32 +: 32]);

                total_outputs = total_outputs + 1;

                if (actual_value === expected_value) begin
                    pass_count = pass_count + 1;
                end
                else begin
                    fail_count = fail_count + 1;

                    if (fail_count <= 20) begin
                        $display(
                            "[FAIL] Tile=%0d Lane=%0d Expected=%0d Actual=%0d",
                            tile,
                            lane,
                            expected_value,
                            actual_value
                        );
                    end
                end

            end

            // Wait for FSM to return to IDLE
            @(posedge clk);
            @(negedge clk);

            if ((tile + 1) % 196 == 0) begin
                $display(
                    "[PROGRESS] Completed %0d / %0d tiles",
                    tile + 1,
                    NUM_TILES
                );
            end

        end

        // Final results
        $display("\n========================================");
        $display(" EXHAUSTIVE VERIFICATION RESULTS");
        $display("========================================");

        $display("Tiles processed: %0d", NUM_TILES);
        $display("Outputs checked: %0d", total_outputs);
        $display("PASS: %0d", pass_count);
        $display("FAIL: %0d", fail_count);

        if (fail_count == 0 && total_outputs == 25088) begin
            $display("\n[PASS] ALL FC1 RTL TILE OUTPUTS MATCH PYTHON!");
        end
        else begin
            $display("\n[FAIL] FC1 RTL VERIFICATION FAILED!");
        end

        $display("========================================\n");

        $finish;

    end

    initial begin
        #100000000;
        $display("[FAIL] SIMULATION TIMEOUT");
        $finish;
    end

endmodule
