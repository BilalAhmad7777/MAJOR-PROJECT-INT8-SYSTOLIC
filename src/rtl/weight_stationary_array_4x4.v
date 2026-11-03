`timescale 1ns / 1ps

module weight_stationary_array_4x4 #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input  wire                clk,
    input  wire                rst_n,

    input  wire                load_weights,
    input  wire                start,

    input  wire signed [127:0] activation_matrix,
    input  wire signed [127:0] weight_matrix,

    output wire signed [511:0] result_matrix,
    output wire                done
);

    // Delay `start` by 1 cycle if `start` was asserted immediately after `load_weights`
    reg load_d1;
    reg start_d1;

    always @(posedge clk) begin
        if (!rst_n) begin
            load_d1  <= 1'b0;
            start_d1 <= 1'b0;
        end else begin
            load_d1  <= load_weights;
            start_d1 <= start && load_d1;
        end
    end

    wire effective_start = (start && !load_d1) || start_d1;

    weight_stationary_array_4x4_fsm systolic_core (
        .clk               (clk),
        .rst_n             (rst_n),
        .load_weights      (load_weights),
        .start             (effective_start),
        .activation_matrix (activation_matrix),
        .weight_matrix     (weight_matrix),
        .result_matrix     (result_matrix),
        .done              (done)
    );

endmodule