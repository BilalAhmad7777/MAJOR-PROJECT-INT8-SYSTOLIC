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

    localparam signed [OUT_WIDTH-1:0] MAX_VALUE =
        (1 <<< (OUT_WIDTH-1)) - 1;

    localparam signed [OUT_WIDTH-1:0] MIN_VALUE =
        -(1 <<< (OUT_WIDTH-1));

    always @(*) begin

        // Extend input to avoid overflow during rounding
        rounded_value = {data_in[ACC_WIDTH-1], data_in};

        // Round to nearest, ties away from zero
        if (shift != 0) begin
    if (data_in >= 0)
        rounded_value =
            rounded_value + (1 <<< (shift-1));
    else
        rounded_value =
            rounded_value + (1 <<< (shift-1)) - 1;
end
        // Arithmetic right shift preserves sign
        shifted_value = rounded_value >>> shift;

        // Saturate to signed INT8 range
        if (shifted_value > MAX_VALUE)
            data_out = MAX_VALUE;
        else if (shifted_value < MIN_VALUE)
            data_out = MIN_VALUE;
        else
            data_out = shifted_value[OUT_WIDTH-1:0];

    end

endmodule
