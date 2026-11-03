`timescale 1ns / 1ps
// -----------------------------------------------------------------------------
// mnist_accel_v2_top : self-contained INT8 MNIST accelerator (784-32-10)
//
//   image buffer  -> streaming 4x4 weight-stationary array -> accumulators
//        ^                         ^                              |
//   img_wr_* port            weight ROMs (FC1/FC2)      bias + ReLU + requant (FC1)
//                                                                 v
//                          INT8 hidden (output) buffer --(FC2 on the SAME array)--> INT32 logits
//                                                                 v
//                                                           argmax -> predicted_class
//
//  * Processes BATCH images at once (BATCH >= 7, default 8): each 4x4 weight tile is
//    held in the PEs while BATCH activation vectors stream through, and tiles follow
//    each other without bubbles -> ~16 MACs per clock.
//  * Weights/biases are ROMs initialised with $readmemh (files from
//    src/python/export_streaming_roms.py).  Run simulation from the project root.
//  * Load images through the img_wr_* word port while busy = 0:
//        word address = image*196 + k_tile, data = {pix[4k+3], pix[4k+2], pix[4k+1], pix[4k]}
//  * start (1 cycle) -> busy -> done (1 cycle).  batch_logits / predicted_class stay valid
//    until the next start.
// -----------------------------------------------------------------------------
module mnist_accel_v2_top #(
    parameter BATCH      = 8,
    parameter FC1_W_FILE = "rom/fc1_weights_tiles.hex",
    parameter FC2_W_FILE = "rom/fc2_weights_tiles.hex",
    parameter FC1_B_FILE = "rom/fc1_bias.hex",
    parameter FC2_B_FILE = "rom/fc2_bias.hex"
)(
    input  wire                              clk,
    input  wire                              rst_n,

    // image buffer write port (ignored while busy)
    input  wire                              img_wr_en,
    input  wire [$clog2(BATCH*196)-1:0]      img_wr_addr,
    input  wire [31:0]                       img_wr_data,

    input  wire                              start,
    output reg                               busy,
    output reg                               done,

    output wire [BATCH*320-1:0]              batch_logits,      // [image][neuron] INT32
    output wire [BATCH*4-1:0]                predicted_class,

    output reg  [31:0]                       perf_cycles,       // busy cycles of last run
    output reg  [31:0]                       perf_mac_ops       // PE multiply-accumulates of last run
);

    // synopsys translate_off
    initial begin
        if (BATCH < 7) begin
            $display("ERROR: mnist_accel_v2_top needs BATCH >= 7 (weight-swap wavefront)");
            $finish;
        end
    end
    // synopsys translate_on

    // ------------------------------------------------------------------ memories
    reg [31:0]  img_mem   [0:BATCH*196-1];      // input (image) buffer
    reg [7:0]   hidden    [0:BATCH*32-1];       // INT8 output buffer of FC1 (input of FC2)
    reg [127:0] fc1_w_rom [0:1567];
    reg [127:0] fc2_w_rom [0:31];            // 24 tiles used
    reg [31:0]  fc1_b_rom [0:31];
    reg [31:0]  fc2_b_rom [0:9];
    reg signed [31:0] acc       [0:BATCH*4-1];  // 32-bit accumulators
    reg signed [31:0] logit_mem [0:BATCH*10-1];
    reg [3:0]         pred_mem  [0:BATCH-1];

    initial begin
        $readmemh(FC1_W_FILE, fc1_w_rom);
        $readmemh(FC2_W_FILE, fc2_w_rom);
        $readmemh(FC1_B_FILE, fc1_b_rom);
        $readmemh(FC2_B_FILE, fc2_b_rom);
    end

    always @(posedge clk) begin
        if (img_wr_en && !busy)
            img_mem[img_wr_addr] <= img_wr_data;
    end

    genvar gi;
    generate
        for (gi = 0; gi < BATCH*10; gi = gi + 1) begin : LOGIT_OUT
            assign batch_logits[gi*32 +: 32] = logit_mem[gi];
        end
        for (gi = 0; gi < BATCH; gi = gi + 1) begin : PRED_OUT
            assign predicted_class[gi*4 +: 4] = pred_mem[gi];
        end
    endgenerate

    // ------------------------------------------------------------------ control state
    localparam S_IDLE   = 3'd0;
    localparam S_PREP   = 3'd1;
    localparam S_STREAM = 3'd2;
    localparam S_DRAIN  = 3'd3;
    localparam S_POST   = 3'd4;
    localparam S_NEXT   = 3'd5;
    localparam S_ARGMAX = 3'd6;
    localparam S_DONE   = 3'd7;

    reg [2:0]  state;
    reg        phase;           // 0 = FC1 (784->32), 1 = FC2 (32->10)
    reg [3:0]  grp;             // group of 4 output neurons
    reg [7:0]  cnt_kt;          // reduction tile (k / 4)
    reg [3:0]  cnt_i;           // image index inside the batch (input side)
    reg [3:0]  out_i;           // image index of the row leaving the array
    reg [11:0] rows_acc;        // rows accumulated in this burst
    reg [3:0]  post_i;
    reg [3:0]  am_i, am_n;
    reg signed [31:0] best_v;
    reg [3:0]  best_n;

    wire [7:0]  kt_last    = phase ? 8'd7 : 8'd195;
    wire [11:0] total_rows = phase ? (8 * BATCH) : (196 * BATCH);

    // ------------------------------------------------------------------ array feed
    wire streaming = (state == S_STREAM);

    wire [15:0] w_idx1 = grp * 196 + cnt_kt;
    wire [15:0] w_idx2 = grp * 8   + cnt_kt;
    wire [127:0] weight_tile = phase ? fc2_w_rom[w_idx2[4:0]] : fc1_w_rom[w_idx1];

    wire [15:0] a_idx = cnt_i * 196 + cnt_kt;
    wire [15:0] h_idx = cnt_i * 32  + cnt_kt * 4;
    wire [31:0] img_word = img_mem[a_idx];
    wire [31:0] hid_word = { hidden[h_idx+3], hidden[h_idx+2], hidden[h_idx+1], hidden[h_idx] };
    wire [31:0] act_word = phase ? hid_word : img_word;

    wire [127:0] row_data;
    wire         row_valid;
    wire [15:0]  pe_active;

    systolic_array_ws_4x4 array (
        .clk          (clk),
        .rst_n        (rst_n),
        .stream_valid (streaming),
        .stream_first (cnt_i == 4'd0),
        .act_word     (act_word),
        .weight_tile  (weight_tile),
        .row_data     (row_data),
        .row_valid    (row_valid),
        .pe_active    (pe_active)
    );

    // ------------------------------------------------------------------ FC1 post-processing (bias+ReLU+requant)
    wire signed [31:0] pp_acc  [0:3];
    wire signed [31:0] pp_bias [0:3];
    wire signed [7:0]  pp_out  [0:3];

    generate
        for (gi = 0; gi < 4; gi = gi + 1) begin : POST
            assign pp_acc[gi]  = acc[post_i*4 + gi];
            assign pp_bias[gi] = fc1_b_rom[grp*4 + gi];
            fc1_postprocess pp (
                .accumulator    (pp_acc[gi]),
                .bias           (pp_bias[gi]),
                .activation_int8(pp_out[gi])
            );
        end
    endgenerate

    // ------------------------------------------------------------------ argmax helpers
    wire signed [31:0] am_cur  = logit_mem[am_i*10 + am_n];
    wire               am_take = (am_n == 4'd0) || (am_cur > best_v);
    wire signed [31:0] am_v    = am_take ? am_cur : best_v;
    wire [3:0]         am_idx  = am_take ? am_n   : best_n;

    // ------------------------------------------------------------------ performance counters helper
    reg [4:0] active_count;
    integer   p;
    always @(*) begin
        active_count = 5'd0;
        for (p = 0; p < 16; p = p + 1)
            active_count = active_count + pe_active[p];
    end

    // ------------------------------------------------------------------ main controller
    integer c, j, n_idx;

    always @(posedge clk) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            phase        <= 1'b0;
            grp          <= 4'd0;
            cnt_kt       <= 8'd0;
            cnt_i        <= 4'd0;
            out_i        <= 4'd0;
            rows_acc     <= 12'd0;
            post_i       <= 4'd0;
            am_i         <= 4'd0;
            am_n         <= 4'd0;
            best_v       <= 32'sd0;
            best_n       <= 4'd0;
            busy         <= 1'b0;
            done         <= 1'b0;
            perf_cycles  <= 32'd0;
            perf_mac_ops <= 32'd0;
        end else begin
            done <= 1'b0;

            // performance counters (counts every busy cycle)
            if (busy) begin
                perf_cycles  <= perf_cycles + 32'd1;
                perf_mac_ops <= perf_mac_ops + active_count;
            end

            // accumulate every row leaving the array (in order: image 0..BATCH-1, repeating)
            if (row_valid) begin
                for (c = 0; c < 4; c = c + 1)
                    acc[out_i*4 + c] <= acc[out_i*4 + c] + $signed(row_data[c*32 +: 32]);
                rows_acc <= rows_acc + 12'd1;
                out_i    <= (out_i == BATCH-1) ? 4'd0 : out_i + 4'd1;
            end

            case (state)
                S_IDLE: begin
                    if (start) begin
                        busy         <= 1'b1;
                        phase        <= 1'b0;
                        grp          <= 4'd0;
                        perf_cycles  <= 32'd0;
                        perf_mac_ops <= 32'd0;
                        state        <= S_PREP;
                    end
                end

                // clear accumulators/counters for one burst (one group of 4 neurons)
                S_PREP: begin
                    for (j = 0; j < BATCH*4; j = j + 1)
                        acc[j] <= 32'sd0;
                    rows_acc <= 12'd0;
                    out_i    <= 4'd0;
                    cnt_i    <= 4'd0;
                    cnt_kt   <= 8'd0;
                    state    <= S_STREAM;
                end

                // one vector per clock; weight tile index (cnt_kt) advances every BATCH clocks
                S_STREAM: begin
                    if (cnt_i == BATCH-1) begin
                        cnt_i <= 4'd0;
                        if (cnt_kt == kt_last)
                            state <= S_DRAIN;          // keep cnt_kt: weight bus must stay put
                        else
                            cnt_kt <= cnt_kt + 8'd1;
                    end else begin
                        cnt_i <= cnt_i + 4'd1;
                    end
                end

                // wait until every row of the burst has been accumulated
                S_DRAIN: begin
                    if (rows_acc == total_rows) begin
                        post_i <= 4'd0;
                        state  <= S_POST;
                    end
                end

                // one image per clock
                S_POST: begin
                    if (!phase) begin
                        for (c = 0; c < 4; c = c + 1)
                            hidden[post_i*32 + grp*4 + c] <= pp_out[c];
                    end else begin
    for (c = 0; c < 4; c = c + 1) begin
        n_idx = grp*4 + c;

        case (n_idx)
            0: logit_mem[post_i*10 + 0] <= acc[post_i*4+c] + $signed(fc2_b_rom[0]);
            1: logit_mem[post_i*10 + 1] <= acc[post_i*4+c] + $signed(fc2_b_rom[1]);
            2: logit_mem[post_i*10 + 2] <= acc[post_i*4+c] + $signed(fc2_b_rom[2]);
            3: logit_mem[post_i*10 + 3] <= acc[post_i*4+c] + $signed(fc2_b_rom[3]);
            4: logit_mem[post_i*10 + 4] <= acc[post_i*4+c] + $signed(fc2_b_rom[4]);
            5: logit_mem[post_i*10 + 5] <= acc[post_i*4+c] + $signed(fc2_b_rom[5]);
            6: logit_mem[post_i*10 + 6] <= acc[post_i*4+c] + $signed(fc2_b_rom[6]);
            7: logit_mem[post_i*10 + 7] <= acc[post_i*4+c] + $signed(fc2_b_rom[7]);
            8: logit_mem[post_i*10 + 8] <= acc[post_i*4+c] + $signed(fc2_b_rom[8]);
            9: logit_mem[post_i*10 + 9] <= acc[post_i*4+c] + $signed(fc2_b_rom[9]);
            default: ;
        endcase
    end
end

                    if (post_i == BATCH-1)
                        state <= S_NEXT;
                    else
                        post_i <= post_i + 4'd1;
                end

                S_NEXT: begin
                    if (!phase) begin
                        if (grp == 4'd7) begin
                            phase <= 1'b1;
                            grp   <= 4'd0;
                        end else begin
                            grp <= grp + 4'd1;
                        end
                        state <= S_PREP;
                    end else begin
                        if (grp == 4'd2) begin
                            am_i  <= 4'd0;
                            am_n  <= 4'd0;
                            state <= S_ARGMAX;
                        end else begin
                            grp   <= grp + 4'd1;
                            state <= S_PREP;
                        end
                    end
                end

                // serial argmax: 10 clocks per image, first maximum wins (same as numpy.argmax)
                S_ARGMAX: begin
                    best_v <= am_v;
                    best_n <= am_idx;
                    if (am_n == 4'd9) begin
                        pred_mem[am_i] <= am_idx;
                        am_n <= 4'd0;
                        if (am_i == BATCH-1)
                            state <= S_DONE;
                        else
                            am_i <= am_i + 4'd1;
                    end else begin
                        am_n <= am_n + 4'd1;
                    end
                end

                S_DONE: begin
                    done  <= 1'b1;
                    busy  <= 1'b0;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
