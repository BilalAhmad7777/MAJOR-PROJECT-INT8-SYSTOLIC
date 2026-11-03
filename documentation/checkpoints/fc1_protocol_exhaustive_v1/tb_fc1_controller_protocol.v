`timescale 1ns / 1ps

module tb_fc1_controller_protocol;

    reg clk = 0;
    reg rst_n = 0;
    reg start = 0;
    reg tile_valid = 0;

    reg signed [127:0] activation_matrix = 0;
    reg signed [127:0] weight_matrix = 0;
    reg signed [127:0] bias_group = 0;

    wire tile_ready;
    wire busy;
    wire done;
    wire [1023:0] hidden_output;

    integer pass_count = 0;
    integer fail_count = 0;

    reg handshake_seen = 0;

    localparam [127:0] TEST_ACT =
        128'h123456789ABCDEF0123456789ABCDEF0;

    localparam [127:0] TEST_WEIGHT =
        128'h0FEDCBA9876543210FEDCBA987654321;

    fc1_controller dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .tile_valid(tile_valid),
        .activation_matrix(activation_matrix),
        .weight_matrix(weight_matrix),
        .bias_group(bias_group),
        .tile_ready(tile_ready),
        .busy(busy),
        .done(done),
        .hidden_output(hidden_output)
    );

    always #5 clk = ~clk;

    initial begin

        $display("===== FC1 VALID/READY PROTOCOL TEST =====");

        // Reset
        repeat (2) @(negedge clk);
        rst_n = 1;

        // Start FC1
        @(negedge clk);
        start = 1;

        @(negedge clk);
        start = 0;

        // Wait for WAIT_TILE
        wait (dut.state == 4'd2);
        @(negedge clk);

        if (tile_ready !== 1'b0) begin
            $display("[FAIL] tile_ready should initially be LOW");
            fail_count = fail_count + 1;
        end else begin
            $display("[PASS] Initial tile_ready is LOW");
            pass_count = pass_count + 1;
        end

        // Present valid while ready is LOW.
        activation_matrix = TEST_ACT;
        weight_matrix = TEST_WEIGHT;
        tile_valid = 1;

        // First clock: controller must NOT accept the tile.
        @(posedge clk);
        #1;

        if (tile_ready === 1'b1 &&
            dut.activation_reg !== TEST_ACT &&
            dut.weight_reg !== TEST_WEIGHT) begin

            $display("[PASS] Tile not accepted before handshake");
            pass_count = pass_count + 1;

        end else begin

            $display("[FAIL] Incorrect early tile acceptance");
            fail_count = fail_count + 1;

        end

        // Keep valid and data stable until ready is HIGH.
        while (!handshake_seen) begin

            @(posedge clk);

            if (tile_valid && tile_ready) begin

                handshake_seen = 1;
                #1;

                if (dut.activation_reg === TEST_ACT &&
                    dut.weight_reg === TEST_WEIGHT) begin

                    $display("[PASS] Tile captured on valid/ready handshake");
                    pass_count = pass_count + 1;

                end else begin

                    $display("[FAIL] Captured tile data mismatch");
                    fail_count = fail_count + 1;

                end

            end

        end

        // Deassert valid after handshake.
        @(negedge clk);
        tile_valid = 0;
        // Change external inputs after the handshake.
activation_matrix = ~TEST_ACT;
weight_matrix = ~TEST_WEIGHT;

#1;

if (dut.activation_reg === TEST_ACT &&
    dut.weight_reg === TEST_WEIGHT) begin

    $display("[PASS] Captured tile remains stable after input changes");
    pass_count = pass_count + 1;

end else begin

    $display("[FAIL] Captured tile changed unexpectedly");
    fail_count = fail_count + 1;

end

        $display("-----------------------------------------");
        $display("PROTOCOL TEST: %0d PASS / %0d FAIL",
                 pass_count, fail_count);

        if (fail_count == 0)
            $display("[PASS] VALID/READY PROTOCOL VERIFIED");
        else
            $display("[FAIL] VALID/READY PROTOCOL FAILED");

        $finish;

    end

    initial begin
        #100000;
        $display("[FAIL] Protocol test timed out");
        $finish;
    end

endmodule
