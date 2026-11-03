`timescale 1ns / 1ps
// Stand-alone test of the streaming array: 12 tiles x 8 vectors, random + extreme data,
// two bursts separated by an idle gap (like the top-level controller does).
module tb_systolic_array_ws_4x4;

    localparam T = 12;
    localparam B = 8;
    localparam N_ROWS = T*B;

    reg         clk = 1'b0;
    reg         rst_n = 1'b0;
    reg         stream_valid = 1'b0;
    reg         stream_first = 1'b0;
    reg  [31:0] act_word = 32'd0;
    reg  [127:0] weight_tile = 128'd0;
    wire [127:0] row_data;
    wire         row_valid;
    wire [15:0]  pe_active;

    always #5 clk = ~clk;

    systolic_array_ws_4x4 dut (
        .clk(clk), .rst_n(rst_n),
        .stream_valid(stream_valid), .stream_first(stream_first),
        .act_word(act_word), .weight_tile(weight_tile),
        .row_data(row_data), .row_valid(row_valid), .pe_active(pe_active)
    );

    reg signed [7:0]  Wm   [0:T*16-1];        // Wm[t*16 + k*4 + c]
    reg signed [7:0]  Am   [0:N_ROWS*4-1];    // Am[(t*B+i)*4 + k]
    reg signed [31:0] expv [0:N_ROWS*4-1];
    integer pres_cycle [0:N_ROWS-1];

    integer t, i, k, c, s, r, acc_i;
    integer errors, checks, out_idx, cycle, lat0, lat_bad, total_active, peak_active, q, mk, mc;

    task present;
        input integer sidx;
        integer tt, ii, kk, cc;
        begin
            tt = sidx / B;
            ii = sidx % B;
            for (kk = 0; kk < 4; kk = kk + 1)
                act_word[8*kk +: 8] = Am[(tt*B + ii)*4 + kk];
            for (kk = 0; kk < 4; kk = kk + 1)
                for (cc = 0; cc < 4; cc = cc + 1)
                    weight_tile[(kk*4 + cc)*8 +: 8] = Wm[tt*16 + kk*4 + cc];
            stream_valid = 1'b1;
            stream_first = (ii == 0);
            pres_cycle[sidx] = cycle;
        end
    endtask

    // ---- monitor: sample on every rising edge (sees pre-edge register values) ----
    always @(posedge clk) begin
        cycle = cycle + 1;
        if (rst_n) begin
            q = 0;
            for (mk = 0; mk < 16; mk = mk + 1) q = q + pe_active[mk];
            total_active = total_active + q;
            if (q > peak_active) peak_active = q;

            if (row_valid) begin
                if (out_idx < N_ROWS) begin
                    for (mc = 0; mc < 4; mc = mc + 1) begin
                        checks = checks + 1;
                        if (row_data[mc*32 +: 32] !== expv[out_idx*4 + mc]) begin
                            errors = errors + 1;
                            if (errors <= 10)
                                $display("[FAIL] row %0d col %0d : got %0d expected %0d",
                                         out_idx, mc, $signed(row_data[mc*32 +: 32]), expv[out_idx*4 + mc]);
                        end
                    end
                    if (out_idx == 0) lat0 = cycle - pres_cycle[0];
                    else if ((cycle - pres_cycle[out_idx]) != lat0) lat_bad = lat_bad + 1;
                end else begin
                    errors = errors + 1;
                    $display("[FAIL] unexpected extra row");
                end
                out_idx = out_idx + 1;
            end
        end
    end

    initial begin
        errors = 0; checks = 0; out_idx = 0; cycle = 0; lat0 = 0; lat_bad = 0;
        total_active = 0; peak_active = 0;

        // ---- data: random, plus extreme corner tiles ----
        for (s = 0; s < T*16; s = s + 1)      Wm[s] = $random;
        for (s = 0; s < N_ROWS*4; s = s + 1)  Am[s] = $random;
        for (k = 0; k < 16; k = k + 1) Wm[0*16 + k] = -8'sd128;      // tile 0: all -128
        for (k = 0; k < 16; k = k + 1) Wm[1*16 + k] =  8'sd127;      // tile 1: all +127
        for (s = 0; s < B*4; s = s + 1) Am[0*B*4 + s] = -8'sd128;    // tile 0 acts: -128
        for (s = 0; s < B*4; s = s + 1) Am[1*B*4 + s] = -8'sd128;    // tile 1 acts: -128

        for (r = 0; r < N_ROWS; r = r + 1) begin
            t = r / B;
            for (c = 0; c < 4; c = c + 1) begin
                acc_i = 0;
                for (k = 0; k < 4; k = k + 1)
                    acc_i = acc_i + Am[r*4 + k] * Wm[t*16 + k*4 + c];
                expv[r*4 + c] = acc_i;
            end
        end

        $display("==========================================");
        $display("  STREAMING 4x4 ARRAY TEST (%0d tiles x %0d vectors)", T, B);
        $display("==========================================");

        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        // burst 0: tiles 0..5
        for (s = 0; s < 6*B; s = s + 1) begin
            present(s);
            @(negedge clk);
        end
        stream_valid = 1'b0; stream_first = 1'b0;          // weight bus keeps the last tile
        repeat (25) @(negedge clk);

        // burst 1: tiles 6..11
        for (s = 6*B; s < N_ROWS; s = s + 1) begin
            present(s);
            @(negedge clk);
        end
        stream_valid = 1'b0; stream_first = 1'b0;
        repeat (30) @(negedge clk);

        if (out_idx != N_ROWS) begin
            errors = errors + 1;
            $display("[FAIL] expected %0d rows, got %0d", N_ROWS, out_idx);
        end
        if (lat_bad != 0) begin
            errors = errors + 1;
            $display("[FAIL] row latency not constant (%0d rows differ)", lat_bad);
        end

        $display("INFO: input-to-output latency = %0d clock edges (constant)", lat0);
        $display("INFO: PE multiply-accumulates counted = %0d (expected %0d)", total_active, N_ROWS*16);
        $display("INFO: peak simultaneous active PEs   = %0d", peak_active);
        if (total_active != N_ROWS*16) errors = errors + 1;
        if (peak_active != 16)         errors = errors + 1;

        $display("------------------------------------------");
        $display("ARRAY CHECKS: %0d   FAIL: %0d", checks, errors);
        if (errors == 0) $display("RESULT: PASS");
        else             $display("RESULT: FAIL");
        $display("------------------------------------------");
        $finish;
    end

    initial begin
        #1000000;
        $display("TIMEOUT");
        $finish;
    end

endmodule
