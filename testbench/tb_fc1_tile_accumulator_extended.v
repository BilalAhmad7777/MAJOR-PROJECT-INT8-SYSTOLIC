`timescale 1ns / 1ps

module tb_fc1_tile_accumulator_extended;

    localparam TILE_COUNT = 196;

    reg clk = 0;
    reg rst_n = 0;
    reg clear_acc = 0;
    reg tile_valid = 0;
    reg signed [511:0] tile_result_matrix = 0;

    wire signed [511:0] accumulated_matrix;
    wire done;

    integer i;
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

    task check_lane;
        input integer lane;
        input integer expected;
        begin
            if ($signed(accumulated_matrix[lane*32 +: 32]) === expected) begin
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] Lane %0d expected=%0d got=%0d",
                    lane, expected,
                    $signed(accumulated_matrix[lane*32 +: 32]));
                fail_count = fail_count + 1;
            end
        end
    endtask

    task reset_accumulator;
        begin
            @(negedge clk);
            clear_acc = 1;
            tile_valid = 0;
            @(negedge clk);
            clear_acc = 0;
        end
    endtask

    initial begin
        $display("===== FC1 ACCUMULATOR EXTENDED AUDIT =====");

        repeat (2) @(negedge clk);
        rst_n = 1;

        // TEST 1: Repeated mixed-sign accumulation.
        for (i = 0; i < 16; i = i + 1)
            tile_result_matrix[i*32 +: 32] = (i % 2 == 0) ? 100 : -100;

        repeat (3) begin
            @(negedge clk);
            tile_valid = 1;
            @(negedge clk);
            tile_valid = 0;
        end

        for (i = 0; i < 16; i = i + 1)
            check_lane(i, (i % 2 == 0) ? 300 : -300);

        $display("[INFO] Repeated signed accumulation checked.");

        // TEST 2: clear_acc and tile_valid simultaneously.
        @(negedge clk);
        clear_acc = 1;
        tile_valid = 1;
        tile_result_matrix = {16{32'd123}};
        @(negedge clk);
        clear_acc = 0;
        tile_valid = 0;

        for (i = 0; i < 16; i = i + 1)
            check_lane(i, 0);

        $display("[INFO] Clear priority checked.");

        // TEST 3: Fill all 196 tiles with value 1 per lane.
        tile_result_matrix = {16{32'd1}};

        for (i = 0; i < TILE_COUNT; i = i + 1) begin
            @(negedge clk);
            tile_valid = 1;
            @(negedge clk);
            tile_valid = 0;
        end

        for (i = 0; i < 16; i = i + 1)
            check_lane(i, TILE_COUNT);

                if (done === 1'b1) begin
            $display("[PASS] Done asserted on final tile");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Done missing on final tile");
            fail_count = fail_count + 1;
        end

        @(negedge clk);
        if (done === 1'b0) begin
            $display("[PASS] Done cleared after one cycle");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Done did not clear");
            fail_count = fail_count + 1;
        end

        // TEST 4: Additional valid tile after completion must be ignored.
        @(negedge clk);
        tile_result_matrix = {16{32'd999}};
        tile_valid = 1;
        @(negedge clk);
        tile_valid = 0;

        for (i = 0; i < 16; i = i + 1)
            check_lane(i, TILE_COUNT);

        $display("[INFO] Post-completion protection checked.");

        $display("-----------------------------------------");
        $display("EXTENDED AUDIT: %0d PASS / %0d FAIL",
                 pass_count, fail_count);

        $finish;
    end

endmodule
