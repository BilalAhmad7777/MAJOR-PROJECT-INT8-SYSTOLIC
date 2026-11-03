`timescale 1ns / 1ps

module requantize #(
    parameter ACC_WIDTH = 32,
    parameter OUT_WIDTH = 8
)(
    input wire signed [ACC_WIDTH-1:0] data_in,
    input wire [5:0] shift,
    output reg signed [OUT_WIDTH-1:0] data_out
);

    reg signed [ACC_WIDTH:0] rounded_value;
    reg signed [ACC_WIDTH:0] shifted_value;
    reg signed [ACC_WIDTH:0] rounding_bias;

    localparam signed [OUT_WIDTH-1:0] MAX_VALUE =
        (1 <<< (OUT_WIDTH-1)) - 1;

    localparam signed [OUT_WIDTH-1:0] MIN_VALUE =
        -(1 <<< (OUT_WIDTH-1));

    always @(*) begin

        rounded_value = {data_in[ACC_WIDTH-1], data_in};
        rounding_bias = 0;
        shifted_value = 0;
        data_out = 0;

        if (shift == 0) begin
            shifted_value = rounded_value;
        end
        else if (shift > ACC_WIDTH) begin
            shifted_value = 0;
        end
        else begin

            rounding_bias =
                {{ACC_WIDTH{1'b0}}, 1'b1} << (shift - 1);

            if (data_in >= 0)
                rounded_value =
                    rounded_value + rounding_bias;
            else
                rounded_value =
                    rounded_value + rounding_bias - 1;

            shifted_value = rounded_value >>> shift;

        end

        if (shifted_value > MAX_VALUE)
            data_out = MAX_VALUE;
        else if (shifted_value < MIN_VALUE)
            data_out = MIN_VALUE;
        else
            data_out = shifted_value[OUT_WIDTH-1:0];

    end

endmodule