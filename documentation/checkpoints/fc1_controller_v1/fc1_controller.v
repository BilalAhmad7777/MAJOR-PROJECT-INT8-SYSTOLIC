
`timescale 1ns / 1ps

module fc1_controller (
    input wire clk,
    input wire rst_n,

    // Start one complete FC1 batch (4 images)
    input wire start,

    // External tile input interface
    input wire tile_valid,
    input wire signed [127:0] activation_matrix,
    input wire signed [127:0] weight_matrix,

    // Bias values for the current group of 4 neurons
    input wire signed [127:0] bias_group,

    // Tile request / status
    output reg tile_ready,
    output reg busy,
    output reg done,

    // 4 images x 32 hidden neurons = 128 INT8 values
    output reg [1023:0] hidden_output
);

    localparam IDLE           = 4'd0;
    localparam WAIT_CLEAR     = 4'd1;
    localparam WAIT_TILE      = 4'd2;
    localparam LOAD_TILE      = 4'd3;
    localparam WAIT_LOAD      = 4'd4;
    localparam WAIT_LOAD_DONE = 4'd5;
    localparam START_MAC      = 4'd6;
    localparam WAIT_MAC       = 4'd7;
    localparam ACCUMULATE     = 4'd8;
    localparam NEXT_TILE      = 4'd9;
    localparam WAIT_ACC_DONE  = 4'd10;
    localparam GROUP_ADVANCE  = 4'd11;
    localparam FINISH         = 4'd12;

    reg [3:0] state;

    reg [7:0] tile_count;
    reg [2:0] group_count;

    reg load_weights;
    reg start_mac;

    reg clear_acc;
    reg tile_acc_valid;

    reg signed [127:0] activation_reg;
    reg signed [127:0] weight_reg;

    wire signed [511:0] tile_result;
    wire matrix_done;

    wire signed [511:0] accumulated_matrix;
    wire accumulator_done;

    reg signed [31:0] bias_for_lane [0:15];

    wire signed [7:0] processed_activation [0:15];

    integer i;
    integer sample_number;
    integer neuron_number;

    // ------------------------------------------------
    // 4x4 matrix engine
    // ------------------------------------------------

    weight_stationary_array_4x4_fsm matrix_engine (
        .clk(clk),
        .rst_n(rst_n),
        .load_weights(load_weights),
        .start(start_mac),
        .activation_matrix(activation_reg),
        .weight_matrix(weight_reg),
        .result_matrix(tile_result),
        .done(matrix_done)
    );

    // ------------------------------------------------
    // Accumulates 196 tiles for each group
    // ------------------------------------------------

    fc1_tile_accumulator #(
        .TILE_COUNT(196)
    ) accumulator (
        .clk(clk),
        .rst_n(rst_n),
        .clear_acc(clear_acc),
        .tile_valid(tile_acc_valid),
        .tile_result_matrix(tile_result),
        .accumulated_matrix(accumulated_matrix),
        .done(accumulator_done)
    );

    // ------------------------------------------------
    // Bias + ReLU + requantization
    // ------------------------------------------------

    genvar g;

    generate
        for (g = 0; g < 16; g = g + 1) begin : POSTPROCESS

            fc1_postprocess postprocess (
                .accumulator(accumulated_matrix[g*32 +: 32]),
                .bias(bias_for_lane[g]),
                .activation_int8(processed_activation[g])
            );

        end
    endgenerate

    // ------------------------------------------------
    // FC1 controller
    // ------------------------------------------------

    always @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            state <= IDLE;

            tile_count <= 0;
            group_count <= 0;

            load_weights <= 0;
            start_mac <= 0;

            clear_acc <= 0;
            tile_acc_valid <= 0;

            activation_reg <= 0;
            weight_reg <= 0;

            tile_ready <= 0;
            busy <= 0;
            done <= 0;

            hidden_output <= 0;

            for (i = 0; i < 16; i = i + 1)
                bias_for_lane[i] <= 0;

        end else begin

            // Default: control signals are one-cycle pulses
            load_weights <= 0;
            start_mac <= 0;
            clear_acc <= 0;
            tile_acc_valid <= 0;
            done <= 0;

            case (state)

                // ------------------------------------
                // IDLE
                // ------------------------------------

                IDLE: begin

                    busy <= 0;
                    tile_ready <= 0;

                    if (start) begin

                        busy <= 1;

                        tile_count <= 0;
                        group_count <= 0;

                        clear_acc <= 1;

                        state <= WAIT_CLEAR;

                    end

                end

                // ------------------------------------
                // Wait for accumulator clear
                // ------------------------------------

                WAIT_CLEAR: begin

                    tile_ready <= 0;
                    state <= WAIT_TILE;

                end

                // ------------------------------------
                // Wait for the next input tile
                // ------------------------------------

                WAIT_TILE: begin

                    tile_ready <= 1;

                    if (tile_valid) begin

                        tile_ready <= 0;

                        activation_reg <= activation_matrix;
                        weight_reg <= weight_matrix;

                        // Capture the four biases for this group
                        if (tile_count == 0) begin

                            for (i = 0; i < 16; i = i + 1) begin

                                bias_for_lane[i] <=
                                    $signed(
                                        bias_group[((i % 4) * 32) +: 32]
                                    );

                            end

                        end

                        state <= LOAD_TILE;

                    end

                end

                // ------------------------------------
                // Issue matrix load pulse
                // ------------------------------------

                LOAD_TILE: begin

                    load_weights <= 1;
                    state <= WAIT_LOAD;

                end

                // ------------------------------------
                // Matrix engine accepts load command
                // ------------------------------------

                WAIT_LOAD: begin

                    state <= WAIT_LOAD_DONE;

                end

                // ------------------------------------
                // Matrix engine captures matrix data
                // ------------------------------------

                WAIT_LOAD_DONE: begin

                    state <= START_MAC;

                end

                // ------------------------------------
                // Start matrix computation
                // ------------------------------------

                START_MAC: begin

                    start_mac <= 1;
                    state <= WAIT_MAC;

                end

                // ------------------------------------
                // Wait for matrix engine completion
                // ------------------------------------

                WAIT_MAC: begin

                    if (matrix_done)
                        state <= ACCUMULATE;

                end

                // ------------------------------------
                // Add this tile to the accumulator
                // ------------------------------------

                ACCUMULATE: begin

                    tile_acc_valid <= 1;
                    state <= NEXT_TILE;

                end

                // ------------------------------------
                // Decide whether more tiles are needed
                // ------------------------------------

                NEXT_TILE: begin

                    if (tile_count == 195) begin

                        state <= WAIT_ACC_DONE;

                    end else begin

                        tile_count <= tile_count + 1'b1;
                        state <= WAIT_TILE;

                    end

                end

                // ------------------------------------
                // Wait for final accumulator update
                // ------------------------------------

                WAIT_ACC_DONE: begin

                    if (accumulator_done) begin

                        // Capture 16 processed INT8 values
                        for (i = 0; i < 16; i = i + 1) begin

                            sample_number = i / 4;

                            neuron_number =
                                group_count * 4 + (i % 4);

                            hidden_output[
                                (sample_number * 32 + neuron_number) * 8 +: 8
                            ] <= processed_activation[i];

                        end

                        if (group_count == 7) begin

                            state <= FINISH;

                        end else begin

                            state <= GROUP_ADVANCE;

                        end

                    end

                end

                // ------------------------------------
                // Prepare the next group of neurons
                // ------------------------------------

                GROUP_ADVANCE: begin

                    group_count <= group_count + 1'b1;
                    tile_count <= 0;

                    clear_acc <= 1;

                    state <= WAIT_CLEAR;

                end

                // ------------------------------------
                // Batch complete
                // ------------------------------------

                FINISH: begin

                    busy <= 0;
                    done <= 1;
                    tile_ready <= 0;

                    state <= IDLE;

                end

                default: begin

                    state <= IDLE;
                    tile_ready <= 0;

                end

            endcase

        end

    end

endmodule
