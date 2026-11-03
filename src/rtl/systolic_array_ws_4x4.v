`timescale 1ns / 1ps
// -----------------------------------------------------------------------------
// systolic_array_ws_4x4 : streaming 4x4 weight-stationary systolic array.
//
//   C[i][c] = sum_k A[i][k] * W[k][c]
//
//  Inputs (one vector per clock while stream_valid = 1):
//    act_word    : A[i][0..3], byte k at bits [8k+:8]      (reduction index k)
//    weight_tile : W[k][c],    byte (k*4+c) at [8(k*4+c)+:8]
//    stream_first: 1 on the first vector of a tile -> PEs capture weight_tile
//
//  Rules for the caller:
//    * weight_tile must stay constant for the whole tile AND for 6 more cycles
//      (the wavefront needs up to 6 cycles to reach PE[3][3]).  With
//      >= 7 vectors per tile this is automatic as long as the tile index is not
//      advanced early; after the last tile keep it unchanged until row_valid
//      has delivered every row.
//
//  Output: one 4 x INT32 row per clock, row_data[32c +: 32] = C[i][c],
//          row_valid = 1 exactly 7 cycles after the matching vector entered.
//          Rows leave in the same order as the vectors entered.
//
//  Steady state: 16 MACs per clock (all PEs busy), no bubbles between tiles.
// -----------------------------------------------------------------------------
module systolic_array_ws_4x4 (
    input  wire         clk,
    input  wire         rst_n,

    input  wire         stream_valid,
    input  wire         stream_first,
    input  wire [31:0]  act_word,
    input  wire [127:0] weight_tile,

    output wire [127:0] row_data,
    output wire         row_valid,
    output wire [15:0]  pe_active          // valid MAC in each PE this cycle
);

    // ---------------- input skew (row k delayed by k cycles) ----------------
    wire signed [7:0] a_raw [0:3];
    assign a_raw[0] = act_word[ 7: 0];
    assign a_raw[1] = act_word[15: 8];
    assign a_raw[2] = act_word[23:16];
    assign a_raw[3] = act_word[31:24];

    reg signed [7:0] a1_d0;
    reg signed [7:0] a2_d0, a2_d1;
    reg signed [7:0] a3_d0, a3_d1, a3_d2;

    reg v1_d0;
    reg v2_d0, v2_d1;
    reg v3_d0, v3_d1, v3_d2;

    reg f1_d0;
    reg f2_d0, f2_d1;
    reg f3_d0, f3_d1, f3_d2;

    always @(posedge clk) begin
        if (!rst_n) begin
            a1_d0 <= 8'sd0;
            a2_d0 <= 8'sd0; a2_d1 <= 8'sd0;
            a3_d0 <= 8'sd0; a3_d1 <= 8'sd0; a3_d2 <= 8'sd0;
            v1_d0 <= 1'b0;
            v2_d0 <= 1'b0; v2_d1 <= 1'b0;
            v3_d0 <= 1'b0; v3_d1 <= 1'b0; v3_d2 <= 1'b0;
            f1_d0 <= 1'b0;
            f2_d0 <= 1'b0; f2_d1 <= 1'b0;
            f3_d0 <= 1'b0; f3_d1 <= 1'b0; f3_d2 <= 1'b0;
        end else begin
            a1_d0 <= a_raw[1];
            a2_d0 <= a_raw[2]; a2_d1 <= a2_d0;
            a3_d0 <= a_raw[3]; a3_d1 <= a3_d0; a3_d2 <= a3_d1;

            v1_d0 <= stream_valid;
            v2_d0 <= stream_valid; v2_d1 <= v2_d0;
            v3_d0 <= stream_valid; v3_d1 <= v3_d0; v3_d2 <= v3_d1;

            f1_d0 <= stream_first;
            f2_d0 <= stream_first; f2_d1 <= f2_d0;
            f3_d0 <= stream_first; f3_d1 <= f3_d0; f3_d2 <= f3_d1;
        end
    end

    wire signed [7:0] a_skewed [0:3];
    assign a_skewed[0] = a_raw[0];
    assign a_skewed[1] = a1_d0;
    assign a_skewed[2] = a2_d1;
    assign a_skewed[3] = a3_d2;

    wire v_skewed [0:3];
    assign v_skewed[0] = stream_valid;
    assign v_skewed[1] = v1_d0;
    assign v_skewed[2] = v2_d1;
    assign v_skewed[3] = v3_d2;

    wire f_skewed [0:3];
    assign f_skewed[0] = stream_first;
    assign f_skewed[1] = f1_d0;
    assign f_skewed[2] = f2_d1;
    assign f_skewed[3] = f3_d2;

    // ---------------- 4x4 PE mesh ----------------
    // PE[k][c]: activations flow along the row (c -> c+1),
    //           partial sums flow down the column (k -> k+1).
    wire signed [7:0]  pe_act   [0:3][0:4];
    wire               pe_valid [0:3][0:4];
    wire               pe_first [0:3][0:4];
    wire signed [31:0] pe_psum  [0:4][0:3];

    genvar k, c;
    generate
        for (k = 0; k < 4; k = k + 1) begin : ROW_IN
            assign pe_act  [k][0] = a_skewed[k];
            assign pe_valid[k][0] = v_skewed[k];
            assign pe_first[k][0] = f_skewed[k];
        end
        for (c = 0; c < 4; c = c + 1) begin : COL_IN
            assign pe_psum[0][c] = 32'sd0;
        end

        for (k = 0; k < 4; k = k + 1) begin : PE_ROW
            for (c = 0; c < 4; c = c + 1) begin : PE_COL
                ws_pe_stream #(.DATA_WIDTH(8), .ACC_WIDTH(32)) pe (
                    .clk            (clk),
                    .rst_n          (rst_n),
                    .valid_in       (pe_valid[k][c]),
                    .first_in       (pe_first[k][c]),
                    .weight_bus     ($signed(weight_tile[(k*4 + c)*8 +: 8])),
                    .activation_in  (pe_act[k][c]),
                    .partial_sum_in (pe_psum[k][c]),
                    .activation_out (pe_act[k][c+1]),
                    .partial_sum_out(pe_psum[k+1][c]),
                    .valid_out      (pe_valid[k][c+1]),
                    .first_out      (pe_first[k][c+1])
                );
                assign pe_active[k*4 + c] = pe_valid[k][c];
            end
        end
    endgenerate

    // ---------------- output deskew (column c delayed by 3-c cycles) ----------------
    reg signed [31:0] dc0_d0, dc0_d1, dc0_d2;
    reg signed [31:0] dc1_d0, dc1_d1;
    reg signed [31:0] dc2_d0;

    always @(posedge clk) begin
        if (!rst_n) begin
            dc0_d0 <= 32'sd0; dc0_d1 <= 32'sd0; dc0_d2 <= 32'sd0;
            dc1_d0 <= 32'sd0; dc1_d1 <= 32'sd0;
            dc2_d0 <= 32'sd0;
        end else begin
            dc0_d0 <= pe_psum[4][0]; dc0_d1 <= dc0_d0; dc0_d2 <= dc0_d1;
            dc1_d0 <= pe_psum[4][1]; dc1_d1 <= dc1_d0;
            dc2_d0 <= pe_psum[4][2];
        end
    end

    assign row_data  = { pe_psum[4][3], dc2_d0, dc1_d1, dc0_d2 };
    assign row_valid = pe_valid[3][4];

endmodule
