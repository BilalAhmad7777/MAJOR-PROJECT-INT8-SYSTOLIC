`timescale 1ns / 1ps
// -----------------------------------------------------------------------------
// ws_pe_stream : weight-stationary INT8 PE for the streaming 4x4 array.
//
//  * The weight is held in weight_reg for as long as activations stream through.
//  * A new weight is captured from weight_bus when a vector arrives with
//    first_in = 1 (first vector of a tile).  That vector is already multiplied
//    with the NEW weight (bypass), so tiles follow each other with no bubble.
//  * valid_in / first_in travel with the activation (registered, 1 cycle per PE),
//    so every PE swaps its weight at exactly the right cycle of the wavefront.
// -----------------------------------------------------------------------------
module ws_pe_stream #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input  wire                          clk,
    input  wire                          rst_n,

    input  wire                          valid_in,
    input  wire                          first_in,
    input  wire signed [DATA_WIDTH-1:0]  weight_bus,
    input  wire signed [DATA_WIDTH-1:0]  activation_in,
    input  wire signed [ACC_WIDTH-1:0]   partial_sum_in,

    output reg  signed [DATA_WIDTH-1:0]  activation_out,
    output reg  signed [ACC_WIDTH-1:0]   partial_sum_out,
    output reg                           valid_out,
    output reg                           first_out
);

    reg signed [DATA_WIDTH-1:0] weight_reg;

    wire signed [DATA_WIDTH-1:0] weight_eff = first_in ? weight_bus : weight_reg;

    wire signed [(2*DATA_WIDTH)-1:0] product = activation_in * weight_eff;

    wire signed [ACC_WIDTH-1:0] product_extended =
        {{(ACC_WIDTH-(2*DATA_WIDTH)){product[(2*DATA_WIDTH)-1]}}, product};

    always @(posedge clk) begin
        if (!rst_n) begin
            weight_reg      <= 0;
            activation_out  <= 0;
            partial_sum_out <= 0;
            valid_out       <= 1'b0;
            first_out       <= 1'b0;
        end else begin
            valid_out <= valid_in;
            first_out <= valid_in & first_in;

            if (valid_in) begin
                if (first_in)
                    weight_reg <= weight_bus;

                activation_out  <= activation_in;
                partial_sum_out <= partial_sum_in + product_extended;
            end
        end
    end

endmodule
