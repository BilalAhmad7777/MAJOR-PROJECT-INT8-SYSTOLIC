`timescale 1ns / 1ps

module weight_stationary_array_4x4_fsm (
    input  wire                clk,
    input  wire                rst_n,

    input  wire                load_weights,
    input  wire                start,

    input  wire signed [127:0] activation_matrix,
    input  wire signed [127:0] weight_matrix,

    output reg  signed [511:0] result_matrix,
    output reg                 done
);

    // Explicit Synopsis / Presentation FSM states
    localparam IDLE   = 3'd0;
    localparam LOAD   = 3'd1;
    localparam STREAM = 3'd2;
    localparam DRAIN  = 3'd3;
    localparam DONE   = 3'd4;

    reg [2:0] state;
    reg [3:0] cycle_cnt;
    reg       weights_loaded;

    // Input and Weight Buffers
    reg signed [7:0] act_buf [0:3][0:3];
    reg [127:0]      w_buf;
    reg              pe_load_w;

    integer i, j;

    // -------------------------------------------------------------------------
    // 1. Input Skew Logic (delays row k by k clock cycles for systolic wavefront)
    // -------------------------------------------------------------------------
    wire signed [7:0] a_raw [0:3];
    wire              stream_active = (state == STREAM && cycle_cnt < 4'd4);

    assign a_raw[0] = stream_active ? act_buf[cycle_cnt[1:0]][0] : 8'sd0;
    assign a_raw[1] = stream_active ? act_buf[cycle_cnt[1:0]][1] : 8'sd0;
    assign a_raw[2] = stream_active ? act_buf[cycle_cnt[1:0]][2] : 8'sd0;
    assign a_raw[3] = stream_active ? act_buf[cycle_cnt[1:0]][3] : 8'sd0;

    reg signed [7:0] skew_r1_d0;
    reg signed [7:0] skew_r2_d0, skew_r2_d1;
    reg signed [7:0] skew_r3_d0, skew_r3_d1, skew_r3_d2;

    always @(posedge clk) begin
        if (!rst_n) begin
            skew_r1_d0 <= 8'sd0;
            skew_r2_d0 <= 8'sd0; skew_r2_d1 <= 8'sd0;
            skew_r3_d0 <= 8'sd0; skew_r3_d1 <= 8'sd0; skew_r3_d2 <= 8'sd0;
        end else begin
            skew_r1_d0 <= a_raw[1];
            skew_r2_d0 <= a_raw[2]; skew_r2_d1 <= skew_r2_d0;
            skew_r3_d0 <= a_raw[3]; skew_r3_d1 <= skew_r3_d0; skew_r3_d2 <= skew_r3_d1;
        end
    end

    wire signed [7:0] a_skewed [0:3];
    assign a_skewed[0] = a_raw[0];
    assign a_skewed[1] = skew_r1_d0;
    assign a_skewed[2] = skew_r2_d1;
    assign a_skewed[3] = skew_r3_d2;

    // Valid pipeline mirrors the activation skew exactly.
    reg valid_r1_d0;
    reg valid_r2_d0, valid_r2_d1;
    reg valid_r3_d0, valid_r3_d1, valid_r3_d2;

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_r1_d0 <= 1'b0;
            valid_r2_d0 <= 1'b0;
            valid_r2_d1 <= 1'b0;
            valid_r3_d0 <= 1'b0;
            valid_r3_d1 <= 1'b0;
            valid_r3_d2 <= 1'b0;
        end else begin
            valid_r1_d0 <= stream_active;
            valid_r2_d0 <= stream_active;
            valid_r2_d1 <= valid_r2_d0;
            valid_r3_d0 <= stream_active;
            valid_r3_d1 <= valid_r3_d0;
            valid_r3_d2 <= valid_r3_d1;
        end
    end

    wire valid_skewed [0:3];
    assign valid_skewed[0] = stream_active;
    assign valid_skewed[1] = valid_r1_d0;
    assign valid_skewed[2] = valid_r2_d1;
    assign valid_skewed[3] = valid_r3_d2;

    // -------------------------------------------------------------------------
    // 2. 4x4 Mesh of `weight_stationary_pe` Instances
    //    PE[k][c] holds W[k][c].
    //    Activations flow horizontally (c -> c+1); Partial sums flow vertically (k -> k+1).
    // -------------------------------------------------------------------------
    wire signed [7:0]  pe_act_wire   [0:3][0:4];
    wire signed [31:0] pe_psum_wire  [0:4][0:3];
    wire               pe_valid_wire [0:3][0:4];

    genvar k, c;
    generate
        for (k = 0; k < 4; k = k + 1) begin : ROW_IN
            assign pe_act_wire[k][0]   = a_skewed[k];
            assign pe_valid_wire[k][0] = valid_skewed[k];
        end
        for (c = 0; c < 4; c = c + 1) begin : COL_IN
            assign pe_psum_wire[0][c] = 32'sd0;
        end

        for (k = 0; k < 4; k = k + 1) begin : PE_ROW
            for (c = 0; c < 4; c = c + 1) begin : PE_COL
                weight_stationary_pe #(
                    .DATA_WIDTH(8),
                    .ACC_WIDTH(32)
                ) pe_inst (
                    .clk             (clk),
                    .rst_n           (rst_n),
                    .load_weight     (pe_load_w),
                    .weight_in       ($signed(w_buf[(k*4 + c)*8 +: 8])),
                    .valid_in        (pe_valid_wire[k][c]),
                    .activation_in   (pe_act_wire[k][c]),
                    .partial_sum_in  (pe_psum_wire[k][c]),
                    .activation_out  (pe_act_wire[k][c+1]),
                    .partial_sum_out (pe_psum_wire[k+1][c]),
                    .valid_out       (pe_valid_wire[k][c+1])
                );
            end
        end
    endgenerate

    // -------------------------------------------------------------------------
    // 3. Output Deskew Logic (aligns all 4 columns of row `i` at cycle i + 7)
    // -------------------------------------------------------------------------
    reg signed [31:0] deskew_c0_d0, deskew_c0_d1, deskew_c0_d2;
    reg signed [31:0] deskew_c1_d0, deskew_c1_d1;
    reg signed [31:0] deskew_c2_d0;

    always @(posedge clk) begin
        if (!rst_n) begin
            deskew_c0_d0 <= 32'sd0; deskew_c0_d1 <= 32'sd0; deskew_c0_d2 <= 32'sd0;
            deskew_c1_d0 <= 32'sd0; deskew_c1_d1 <= 32'sd0;
            deskew_c2_d0 <= 32'sd0;
        end else begin
            deskew_c0_d0 <= pe_psum_wire[4][0];
            deskew_c0_d1 <= deskew_c0_d0;
            deskew_c0_d2 <= deskew_c0_d1;

            deskew_c1_d0 <= pe_psum_wire[4][1];
            deskew_c1_d1 <= deskew_c1_d0;

            deskew_c2_d0 <= pe_psum_wire[4][2];
        end
    end

    wire [127:0] c_row_aligned = {
        pe_psum_wire[4][3],
        deskew_c2_d0,
        deskew_c1_d1,
        deskew_c0_d2
    };

    // -------------------------------------------------------------------------
    // 4. FSM Controller (IDLE -> LOAD -> STREAM -> DRAIN -> DONE)
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            state          <= IDLE;
            cycle_cnt      <= 4'd0;
            weights_loaded <= 1'b0;
            pe_load_w      <= 1'b0;
            result_matrix  <= 512'sd0;
            done           <= 1'b0;
            w_buf          <= 128'd0;

            for (i = 0; i < 4; i = i + 1)
                for (j = 0; j < 4; j = j + 1)
                    act_buf[i][j] <= 8'sd0;
        end else begin
            done      <= 1'b0;
            pe_load_w <= 1'b0;

            case (state)
                IDLE: begin
                    cycle_cnt <= 4'd0;
                    if (load_weights) begin
                        w_buf <= weight_matrix;
                        for (i = 0; i < 4; i = i + 1)
                            for (j = 0; j < 4; j = j + 1)
                                act_buf[i][j] <= $signed(activation_matrix[(i*4 + j)*8 +: 8]);
                        pe_load_w <= 1'b1;
                        state     <= LOAD;
                    end else if (start && weights_loaded) begin
                        cycle_cnt <= 4'd0;
                        state     <= STREAM;
                    end
                end

                LOAD: begin
                    weights_loaded <= 1'b1;
                    state          <= IDLE;
                end

                STREAM: begin
                    if (cycle_cnt == 4'd3) begin
                        cycle_cnt <= 4'd4;
                        state     <= DRAIN;
                    end else begin
                        cycle_cnt <= cycle_cnt + 4'd1;
                    end
                end

                DRAIN: begin
                    if (cycle_cnt >= 4'd7 && cycle_cnt <= 4'd10) begin
                        result_matrix[(cycle_cnt - 4'd7)*128 +: 128] <= c_row_aligned;
                    end

                    if (cycle_cnt == 4'd10) begin
                        state <= DONE;
                    end else begin
                        cycle_cnt <= cycle_cnt + 4'd1;
                    end
                end

                DONE: begin
                    done  <= 1'b1;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule