`timescale 1ns / 1ps

module weight_stationary_pe #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input wire clk,
    input wire rst_n,

    // Weight loading
    input wire load_weight,
    input wire signed [DATA_WIDTH-1:0] weight_in,

    // Activation input
    input wire valid_in,
    input wire signed [DATA_WIDTH-1:0] activation_in,

    // Partial sum input
    input wire signed [ACC_WIDTH-1:0] partial_sum_in,

    // Activation forwarded to next PE
    output reg signed [DATA_WIDTH-1:0] activation_out,

    // Partial sum forwarded to next PE
    output reg signed [ACC_WIDTH-1:0] partial_sum_out,

    output reg valid_out
);

    // Locally stored weight: remains stationary during computation
    reg signed [DATA_WIDTH-1:0] weight_reg;

    // Signed INT8 x INT8 produces signed 16-bit product
    wire signed [(2*DATA_WIDTH)-1:0] product;

    assign product = activation_in * weight_reg;

    // Sign-extend product to accumulator width
    wire signed [ACC_WIDTH-1:0] product_extended;

    assign product_extended =
        {{(ACC_WIDTH-(2*DATA_WIDTH)){product[(2*DATA_WIDTH)-1]}},
          product};

    always @(posedge clk) begin
        if (!rst_n) begin
            weight_reg      <= 0;
            activation_out  <= 0;
            partial_sum_out <= 0;
            valid_out       <= 0;
        end
        else begin

            // Load weight only when requested
            if (load_weight)
                weight_reg <= weight_in;

            // Forward activation and accumulated partial sum
            valid_out <= valid_in;

            if (valid_in) begin
                activation_out  <= activation_in;
                partial_sum_out <= partial_sum_in
                                 + product_extended;
            end
        end
    end

endmodule
