`timescale 1ns / 1ps

module tb_fc1_tile_accumulator;

    localparam TILE_COUNT = 196;

    reg clk;
    reg rst_n;
    reg clear_acc;
    reg tile_valid;
    reg signed [511:0] tile_result_matrix;

    wire signed [511:0] accumulated_matrix;
    wire done;

    reg signed [31:0] expected [0:15];

    integer t, i;
    integer pass_count, fail_count;
    integer timeout_count;

    fc1_tile_accumulator #(
        .TILE_COUNT(TILE_COUNT)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .clear_acc(clear_acc),
        .tile_valid(tile_valid),
        .tile_result_matrix(tile_result_matrix),
        .accumulated_matrix(accumulated_matrix),
        .done(done)
    );

    always #5 clk = ~clk;

    initial begin
        clk = 0;
        rst_n = 0;
        clear_acc = 0;
        tile_valid = 0;
        tile_result_matrix = 0;
        pass_count = 0;
        fail_count = 0;
        timeout_count = 0;

        for (i = 0; i < 16; i = i + 1)
            expected[i] = 0;

        $display("===== FC1 TILE ACCUMULATOR TEST =====");

        repeat (3) @(negedge clk);
        rst_n = 1;

        // Clear accumulator before starting.
        @(negedge clk);
        clear_acc = 1;

        @(negedge clk);
        clear_acc = 0;

        // Send 196 partial-sum matrices.
        for (t = 0; t < TILE_COUNT; t = t + 1) begin
            @(negedge clk);

            for (i = 0; i < 16; i = i + 1) begin
                tile_result_matrix[i*32 +: 32] =
                    (t + 1) * (i + 1);

                expected[i] = expected[i]
                    + (t + 1) * (i + 1);
            end

            tile_valid = 1;

            @(negedge clk);
            tile_valid = 0;

            if ((t + 1) % 50 == 0)
                $display("[INFO] Processed %0d / %0d tiles",
                         t + 1, TILE_COUNT);
        end

        // The final tile should have asserted done.
        if (done !== 1'b1) begin
            $display("[FAIL] Done was not asserted after final tile");
            fail_count = fail_count + 1;
        end else begin
            $display("[PASS] Done asserted after all 196 tiles");
            pass_count = pass_count + 1;
        end

        // Check all 16 accumulated INT32 values.
        for (i = 0; i < 16; i = i + 1) begin
            if ($signed(accumulated_matrix[i*32 +: 32])
                !== expected[i]) begin

                $display("[FAIL] Output %0d: expected %0d, got %0d",
                    i, expected[i],
                    $signed(accumulated_matrix[i*32 +: 32]));

                fail_count = fail_count + 1;
            end else begin
                pass_count = pass_count + 1;
            end
        end

        $display("\n===== TEST RESULTS =====");
        $display("Checks passed: %0d", pass_count);
        $display("Checks failed: %0d", fail_count);

        if (fail_count == 0)
            $display("[PASS] FC1 ACCUMULATOR VERIFIED");
        else
            $display("[FAIL] ACCUMULATOR VERIFICATION FAILED");

        $finish;
    end

endmodule
