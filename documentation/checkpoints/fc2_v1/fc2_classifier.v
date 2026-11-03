`timescale 1ns / 1ps

module fc2_classifier (
    input wire clk,
    input wire rst_n,
    input wire start,

    input wire [255:0] hidden_activations,
    input wire [2559:0] weight_matrix,
    input wire [319:0] bias_vector,

    output reg [319:0] logits,
    output reg done
);

    localparam IDLE    = 2'b00;
    localparam COMPUTE = 2'b01;
    localparam DONE    = 2'b10;

    reg [1:0] state;

    reg [5:0] reduction_index;
    reg [3:0] neuron_index;

    reg signed [31:0] accumulator;

    reg signed [7:0] activation_value;
    reg signed [7:0] weight_value;
    reg signed [15:0] product;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            reduction_index <= 0;
            neuron_index <= 0;
            accumulator <= 0;
            logits <= 0;
            done <= 0;
        end
        else begin
            done <= 0;

            case (state)

                IDLE: begin
                    if (start) begin
                        reduction_index <= 0;
                        neuron_index <= 0;
                        accumulator <= 0;
                        logits <= 0;
                        state <= COMPUTE;
                    end
                end

                COMPUTE: begin
                    activation_value =
                        $signed(hidden_activations[
                            reduction_index * 8 +: 8
                        ]);

                    weight_value =
                        $signed(weight_matrix[
                            neuron_index * 256 +
                            reduction_index * 8 +: 8
                        ]);

                    product =
                        activation_value * weight_value;

                    if (reduction_index == 31) begin

                        logits[neuron_index * 32 +: 32] <=
                            accumulator +
                            product +
                            $signed(bias_vector[
                                neuron_index * 32 +: 32
                            ]);

                        accumulator <= 0;
                        reduction_index <= 0;

                        if (neuron_index == 9) begin
                            state <= DONE;
                        end
                        else begin
                            neuron_index <= neuron_index + 1'b1;
                        end

                    end
                    else begin
                        accumulator <= accumulator + product;
                        reduction_index <= reduction_index + 1'b1;
                    end
                end

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
