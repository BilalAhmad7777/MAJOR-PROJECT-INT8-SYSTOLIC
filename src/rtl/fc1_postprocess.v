`timescale 1ns / 1ps

module fc1_postprocess (
    input wire signed [31:0] accumulator,
    input wire signed [31:0] bias,
    output reg signed [7:0] activation_int8
);

    localparam signed [31:0] MULTIPLIER = 638865;

    reg signed [32:0] biased_value;
    reg signed [32:0] relu_value;

    reg signed [63:0] product;
    reg signed [63:0] quotient;
    reg signed [63:0] rounded_value;

    reg [30:0] remainder;

    always @(*) begin
        // Add the INT32 bias using an extra sign bit.
        biased_value =
            {accumulator[31], accumulator}
            + {bias[31], bias};

        // ReLU.
        if (biased_value < 0)
            relu_value = 0;
        else
            relu_value = biased_value;

        // Q31 fixed-point multiplication.
        product =
    $signed({{31{relu_value[32]}}, relu_value})
    *
    $signed({{32{MULTIPLIER[31]}}, MULTIPLIER});

        quotient = product >>> 31;
        remainder = product[30:0];

        // Round to nearest, ties to even.
        rounded_value = quotient;

        if (remainder > 31'd1073741824)
            rounded_value = quotient + 1;
        else if ((remainder == 31'd1073741824)
                 && quotient[0])
            rounded_value = quotient + 1;

        // Saturate to unsigned-positive INT8 range.
        if (rounded_value < 0)
            activation_int8 = 0;
        else if (rounded_value > 127)
            activation_int8 = 127;
        else
            activation_int8 = rounded_value[7:0];
    end

endmodule
