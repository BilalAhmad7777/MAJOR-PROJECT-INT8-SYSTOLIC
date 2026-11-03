`timescale 1ns / 1ps

module fc1_tile_accumulator #(
    parameter TILE_COUNT = 196
)(
    input wire clk,
    input wire rst_n,
    input wire clear_acc,
    input wire tile_valid,
    input wire signed [511:0] tile_result_matrix,
    output wire signed [511:0] accumulated_matrix,
    output reg done
);

    reg signed [31:0] accumulator [0:15];
    reg [7:0] tile_count;

    integer i;
    genvar g;

    generate
        for (g = 0; g < 16; g = g + 1) begin : OUTPUT_PACK
            assign accumulated_matrix[g*32 +: 32] = accumulator[g];
        end
    endgenerate

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tile_count <= 0;
            done <= 0;
            for (i = 0; i < 16; i = i + 1)
                accumulator[i] <= 0;
        end else if (clear_acc) begin
            tile_count <= 0;
            done <= 0;
            for (i = 0; i < 16; i = i + 1)
                accumulator[i] <= 0;
        end else begin
            done <= 0;

            if (tile_valid && tile_count < TILE_COUNT) begin
                for (i = 0; i < 16; i = i + 1)
                    accumulator[i] <= accumulator[i]
                        + $signed(tile_result_matrix[i*32 +: 32]);

                if (tile_count == TILE_COUNT - 1)
                    done <= 1;

                tile_count <= tile_count + 1;
            end
        end
    end

endmodule
