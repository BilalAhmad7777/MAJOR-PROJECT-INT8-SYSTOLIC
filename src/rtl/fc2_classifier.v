`timescale 1ns / 1ps

module fc2_classifier (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          start,

    input  wire [255:0]  hidden_activations,
    input  wire [2559:0] weight_matrix,
    input  wire [319:0]  bias_vector,

    output reg  [319:0]  logits,
    output reg           done
);

    localparam IDLE       = 3'd0;
    localparam PREP_TILE  = 3'd1;
    localparam WAIT_LOAD  = 3'd2;
    localparam START_MAC  = 3'd3;
    localparam WAIT_MAC   = 3'd4;
    localparam NEXT_GROUP = 3'd5;
    localparam DONE_STATE = 3'd6;

    reg [2:0] state;
    reg [1:0] group_idx; // 0: neurons 0..3, 1: neurons 4..7, 2: neurons 8..9
    reg [2:0] tile_idx;  // 0..7 (8 tiles x 4 activations = 32 hidden activations)

    reg          sa_load;
    reg          sa_start;
    reg  [127:0] sa_act_mat;
    reg  [127:0] sa_w_mat;
    wire [511:0] sa_res_mat;
    wire         sa_done;

    reg signed [31:0] group_acc [0:3];

    integer k, c, n_idx;

    weight_stationary_array_4x4_fsm fc2_systolic_engine (
        .clk               (clk),
        .rst_n             (rst_n),
        .load_weights      (sa_load),
        .start             (sa_start),
        .activation_matrix (sa_act_mat),
        .weight_matrix     (sa_w_mat),
        .result_matrix     (sa_res_mat),
        .done              (sa_done)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= IDLE;
            group_idx  <= 2'd0;
            tile_idx   <= 3'd0;
            sa_load    <= 1'b0;
            sa_start   <= 1'b0;
            sa_act_mat <= 128'd0;
            sa_w_mat   <= 128'd0;
            logits     <= 320'd0;
            done       <= 1'b0;
            for (c = 0; c < 4; c = c + 1)
                group_acc[c] <= 32'sd0;
        end else begin
            sa_load  <= 1'b0;
            sa_start <= 1'b0;
            done     <= 1'b0;

            case (state)
                IDLE: begin
                    if (start) begin
                        group_idx <= 2'd0;
                        tile_idx  <= 3'd0;
                        logits    <= 320'd0;
                        for (c = 0; c < 4; c = c + 1)
                            group_acc[c] <= 32'sd0;
                        state <= PREP_TILE;
                    end
                end

                PREP_TILE: begin
                    sa_act_mat <= {96'd0, hidden_activations[tile_idx*32 +: 32]};

                    for (k = 0; k < 4; k = k + 1) begin
                        for (c = 0; c < 4; c = c + 1) begin
                            n_idx = group_idx * 4 + c;
                            if (n_idx < 10) begin
                                sa_w_mat[(k*4 + c)*8 +: 8] <=
                                    weight_matrix[n_idx*256 + (tile_idx*4 + k)*8 +: 8];
                            end else begin
                                sa_w_mat[(k*4 + c)*8 +: 8] <= 8'd0;
                            end
                        end
                    end

                    sa_load <= 1'b1;
                    state   <= WAIT_LOAD;
                end

                WAIT_LOAD: begin
                    state <= START_MAC;
                end

                START_MAC: begin
                    sa_start <= 1'b1;
                    state    <= WAIT_MAC;
                end

                WAIT_MAC: begin
                    if (sa_done) begin
                        for (c = 0; c < 4; c = c + 1) begin
                            group_acc[c] <= group_acc[c] + $signed(sa_res_mat[c*32 +: 32]);
                        end

                        if (tile_idx == 3'd7) begin
                            state <= NEXT_GROUP;
                        end else begin
                            tile_idx <= tile_idx + 3'd1;
                            state    <= PREP_TILE;
                        end
                    end
                end

                NEXT_GROUP: begin
                    for (c = 0; c < 4; c = c + 1) begin
                        n_idx = group_idx * 4 + c;
                        if (n_idx < 10) begin
                            logits[n_idx*32 +: 32] <=
                                group_acc[c] + $signed(bias_vector[n_idx*32 +: 32]);
                        end
                    end

                    for (c = 0; c < 4; c = c + 1)
                        group_acc[c] <= 32'sd0;

                    tile_idx <= 3'd0;

                    if (group_idx == 2'd2) begin
                        state <= DONE_STATE;
                    end else begin
                        group_idx <= group_idx + 2'd1;
                        state     <= PREP_TILE;
                    end
                end

                DONE_STATE: begin
                    done  <= 1'b1;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule