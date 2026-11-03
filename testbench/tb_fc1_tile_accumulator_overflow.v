`timescale 1ns / 1ps

module tb_fc1_tile_accumulator_overflow;

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
        .TILE_COUNT(2)
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

    task send_tile;
        input signed [31:0] value;
        begin
            @(negedge clk);
            for (i = 0; i < 16; i = i + 1)
                tile_result_matrix[i*32 +: 32] = value;
            tile_valid = 1;
            @(negedge clk);
            tile_valid = 0;
        end
    endtask

    task check_lane;
        input integer lane;
        input signed [31:0] expected;
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

    initial begin
        $display("===== FC1 ACCUMULATOR OVERFLOW AUDIT =====");

        repeat (2) @(negedge clk);
        rst_n = 1;

        // Test 1: Positive INT32 overflow.
        send_tile(32'sh7FFFFFFF);
        send_tile(32'sd1);

        for (i = 0; i < 16; i = i + 1)
            check_lane(i, 32'sh80000000);

        $display("[INFO] Positive overflow behaviour checked.");

        // Test 2: Clear and negative overflow.
        @(negedge clk);
        clear_acc = 1;
        @(negedge clk);
        clear_acc = 0;

        send_tile(-32'sh80000000);
        send_tile(-32'sd1);

        for (i = 0; i < 16; i = i + 1)
            check_lane(i, 32'sh7FFFFFFF);

        $display("[INFO] Negative overflow behaviour checked.");

        // Test 3: Clear and normal boundary accumulation.
        @(negedge clk);
        clear_acc = 1;
        @(negedge clk);
        clear_acc = 0;

        send_tile(32'sh7FFFFFFE);
        send_tile(32'sd1);

        for (i = 0; i < 16; i = i + 1)
            check_lane(i, 32'sh7FFFFFFF);

        $display("[INFO] Normal upper boundary checked.");

        $display("-----------------------------------------");
        $display("OVERFLOW AUDIT: %0d PASS / %0d FAIL",
                 pass_count, fail_count);

        $finish;
    end

endmodule
