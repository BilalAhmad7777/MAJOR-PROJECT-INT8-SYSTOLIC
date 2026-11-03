`timescale 1ns / 1ps

module tb_fc1_tile_accumulator_audit;

    localparam TILE_COUNT = 196;

    reg clk = 0;
    reg rst_n = 0;
    reg clear_acc = 0;
    reg tile_valid = 0;
    reg signed [511:0] tile_result_matrix = 0;

    wire signed [511:0] accumulated_matrix;
    wire done;

    integer i;
    integer t;
    integer pass_count = 0;
    integer fail_count = 0;

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

    task check_zero;
        begin
            for (i = 0; i < 16; i = i + 1) begin
                if ($signed(accumulated_matrix[i*32 +: 32]) !== 0) begin
                    $display("[FAIL] Lane %0d not zero", i);
                    fail_count = fail_count + 1;
                end
            end
            if (fail_count == 0)
                pass_count = pass_count + 1;
        end
    endtask

    initial begin
        $display("===== FC1 ACCUMULATOR STRESS AUDIT =====");

        // Reset
        repeat (2) @(negedge clk);
        rst_n = 1;

        // Test 1: Mixed signed accumulation
        @(negedge clk);
        for (i = 0; i < 16; i = i + 1)
            tile_result_matrix[i*32 +: 32] =
                (i % 2 == 0) ? 100 : -100;
        tile_valid = 1;

        @(negedge clk);
        tile_valid = 0;

        for (i = 0; i < 16; i = i + 1) begin
            if ($signed(accumulated_matrix[i*32 +: 32]) !==
                ((i % 2 == 0) ? 100 : -100)) begin
                $display("[FAIL] Signed lane %0d", i);
                fail_count = fail_count + 1;
            end else begin
                pass_count = pass_count + 1;
            end
        end

        // Test 2: clear_acc
        clear_acc = 1;
        @(negedge clk);
        clear_acc = 0;

        for (i = 0; i < 16; i = i + 1) begin
            if ($signed(accumulated_matrix[i*32 +: 32]) !== 0) begin
                $display("[FAIL] Clear failed at lane %0d", i);
                fail_count = fail_count + 1;
            end else begin
                pass_count = pass_count + 1;
            end
        end

        // Test 3: done must remain low before final tile
        for (t = 0; t < TILE_COUNT - 1; t = t + 1) begin
            @(negedge clk);
            tile_result_matrix = 0;
            tile_valid = 1;
            @(negedge clk);
            tile_valid = 0;

            if (done !== 0) begin
                $display("[FAIL] Early done at tile %0d", t + 1);
                fail_count = fail_count + 1;
            end
        end

        // Test 4: done on final tile
        @(negedge clk);
        tile_valid = 1;
        @(negedge clk);
        tile_valid = 0;

        if (done === 1) begin
            $display("[PASS] Done asserted on tile 196");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Done missing on tile 196");
            fail_count = fail_count + 1;
        end

        @(negedge clk);
        if (done === 0) begin
            $display("[PASS] Done cleared after one cycle");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Done did not clear");
            fail_count = fail_count + 1;
        end

        // Test 5: asynchronous reset
        @(negedge clk);
        rst_n = 0;
        #1;

        if (done === 0) begin
            $display("[PASS] Reset clears done");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Reset did not clear done");
            fail_count = fail_count + 1;
        end

        for (i = 0; i < 16; i = i + 1) begin
            if ($signed(accumulated_matrix[i*32 +: 32]) !== 0) begin
                $display("[FAIL] Reset failed at lane %0d", i);
                fail_count = fail_count + 1;
            end
        end

        if (fail_count == 0)
            pass_count = pass_count + 1;

        $display("\n===== AUDIT RESULTS =====");
        $display("PASS: %0d", pass_count);
        $display("FAIL: %0d", fail_count);

        if (fail_count == 0)
            $display("[PASS] FC1 ACCUMULATOR STRESS AUDIT");
        else
            $display("[FAIL] FC1 ACCUMULATOR STRESS AUDIT");

        $finish;
    end

endmodule