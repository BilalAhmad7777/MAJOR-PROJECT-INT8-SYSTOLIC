
`timescale 1ns / 1ps

module weight_stationary_array_4x4 #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input wire clk,
    input wire rst_n,

    input wire load_weights,
    input wire start,

    input wire signed [127:0] activation_matrix,
    input wire signed [127:0] weight_matrix,

    output reg signed [511:0] result_matrix,
    output reg done
);

    // Locally stored weights: one weight per PE
    reg signed [7:0] weight_reg [0:3][0:3];

    // Stored activation matrix
    reg signed [7:0] activation_reg [0:3][0:3];

    // INT32 accumulation for each output
    reg signed [31:0] accumulator [0:3][0:3];

    // Control
    reg [1:0] reduction_index;
    reg busy;

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
            done <= 0;
            busy <= 0;
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
            done <= 0;

            // LOAD PHASE
            if (load_weights && !busy) begin
                for (i = 0; i < 4; i = i + 1) begin
                    for (j = 0; j < 4; j = j + 1) begin
                        weight_reg[i][j] <=
                            weight_matrix[(i*4+j)*8 +: 8];

                        activation_reg[i][j] <=
                            activation_matrix[(i*4+j)*8 +: 8];

                        accumulator[i][j] <= 0;
                    end
                end
            end

            // COMPUTE START
            if (start && !busy) begin
                busy <= 1;
                reduction_index <= 0;
            end

            // FOUR REDUCTION STAGES
            else if (busy) begin

                for (i = 0; i < 4; i = i + 1) begin
                    for (j = 0; j < 4; j = j + 1) begin
                        accumulator[i][j] <=
                            accumulator[i][j] + product[i][j];
                    end
                end

                if (reduction_index == 3) begin
                    for (i = 0; i < 4; i = i + 1) begin
                        for (j = 0; j < 4; j = j + 1) begin
                            result_matrix[(i*4+j)*32 +: 32] <=
                                accumulator[i][j] + product[i][j];
                        end
                    end

                    busy <= 0;
                    done <= 1;
                end
                else begin
                    reduction_index <= reduction_index + 1;
                end
            end
        end
    end

endmodule
