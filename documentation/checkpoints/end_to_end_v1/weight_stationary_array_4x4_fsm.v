
`timescale 1ns / 1ps

module weight_stationary_array_4x4_fsm (
    input wire clk,
    input wire rst_n,

    input wire load_weights,
    input wire start,

    input wire signed [127:0] activation_matrix,
    input wire signed [127:0] weight_matrix,

    output reg signed [511:0] result_matrix,
    output reg done
);

    // FSM states
    localparam IDLE    = 2'b00;
    localparam LOAD    = 2'b01;
    localparam COMPUTE = 2'b10;
    localparam DONE    = 2'b11;

    reg [1:0] state;

    // Weight registers: one stationary weight per output position
    reg signed [7:0] weight_reg [0:3][0:3];

    // Stored activation matrix
    reg signed [7:0] activation_reg [0:3][0:3];

    // INT32 accumulators
    reg signed [31:0] accumulator [0:3][0:3];

    reg [1:0] reduction_index;
    reg weights_loaded;

    integer i, j;

    // Signed INT8 multiplication produces INT16
    wire signed [15:0] product [0:3][0:3];

    genvar r, c;

    generate
        for (r = 0; r < 4; r = r + 1) begin : PRODUCT_ROWS
            for (c = 0; c < 4; c = c + 1) begin : PRODUCT_COLS

                assign product[r][c] =
                    activation_reg[r][reduction_index] *
                    weight_reg[reduction_index][c];

            end
        end
    endgenerate

    always @(posedge clk) begin

        if (!rst_n) begin
            state <= IDLE;
            done <= 0;
            weights_loaded <= 0;
            reduction_index <= 0;
            result_matrix <= 0;

            for (i = 0; i < 4; i = i + 1) begin
                for (j = 0; j < 4; j = j + 1) begin
                    weight_reg[i][j] <= 0;
                    activation_reg[i][j] <= 0;
                    accumulator[i][j] <= 0;
                end
            end
        end

        else begin

            // Default: done is a one-cycle pulse
            done <= 0;

            case (state)

                // WAIT FOR LOAD OR START
                IDLE: begin

                    if (load_weights) begin
                        state <= LOAD;
                    end

                    else if (start && weights_loaded) begin
                        reduction_index <= 0;

                        // Clear previous computation
                        for (i = 0; i < 4; i = i + 1) begin
                            for (j = 0; j < 4; j = j + 1) begin
                                accumulator[i][j] <= 0;
                            end
                        end

                        state <= COMPUTE;
                    end

                end

                // LOAD MATRICES
                LOAD: begin

                    for (i = 0; i < 4; i = i + 1) begin
                        for (j = 0; j < 4; j = j + 1) begin

                            weight_reg[i][j] <=
                                weight_matrix[(i*4+j)*8 +: 8];

                            activation_reg[i][j] <=
                                activation_matrix[(i*4+j)*8 +: 8];

                            accumulator[i][j] <= 0;

                        end
                    end

                    weights_loaded <= 1;
                    state <= IDLE;

                end

                // FOUR REDUCTION CYCLES
                COMPUTE: begin

                    for (i = 0; i < 4; i = i + 1) begin
                        for (j = 0; j < 4; j = j + 1) begin

                            accumulator[i][j] <=
                                accumulator[i][j] + product[i][j];

                        end
                    end

                    if (reduction_index == 3) begin

                        // Include the final product
                        for (i = 0; i < 4; i = i + 1) begin
                            for (j = 0; j < 4; j = j + 1) begin

                                result_matrix[(i*4+j)*32 +: 32] <=
                                    accumulator[i][j] + product[i][j];

                            end
                        end

                        state <= DONE;

                    end

                    else begin
                        reduction_index <= reduction_index + 1;
                    end

                end

                // SIGNAL COMPLETION
                DONE: begin
                    done <= 1;
                    state <= IDLE;
                end

                default: begin
                    state <= IDLE;
                end

            endcase

        end
    end

endmodule
