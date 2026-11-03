`timescale 1ns / 1ps

module accelerator_top (
    input wire clk,
    input wire rst_n,

    input wire load_weights,
    input wire start,

    input wire signed [127:0] activation_matrix,
    input wire signed [127:0] weight_matrix,

    input wire [5:0] quant_shift,

    output wire signed [127:0] output_matrix,
    output wire done
);

    wire signed [511:0] accumulator_matrix;
    wire signed [511:0] relu_matrix;
    wire matrix_done;

    // Matrix multiplication engine
    weight_stationary_array_4x4_fsm matrix_engine (
        .clk(clk),
        .rst_n(rst_n),
        .load_weights(load_weights),
        .start(start),
        .activation_matrix(activation_matrix),
        .weight_matrix(weight_matrix),
        .result_matrix(accumulator_matrix),
        .done(matrix_done)
    );

    // ReLU + requantization for all 16 outputs
    genvar i;

    generate
        for (i = 0; i < 16; i = i + 1) begin : POST_PROCESSING

            wire signed [31:0] relu_value;

            relu #(
                .DATA_WIDTH(32)
            ) relu_unit (
                .data_in(accumulator_matrix[i*32 +: 32]),
                .data_out(relu_value)
            );

            requantize #(
                .ACC_WIDTH(32),
                .OUT_WIDTH(8)
            ) quant_unit (
                .data_in(relu_value),
                .shift(quant_shift),
                .data_out(output_matrix[i*8 +: 8])
            );

        end
    endgenerate

    assign relu_matrix = accumulator_matrix;
    assign done = matrix_done;

endmodule
