
`timescale 1ns / 1ps

module mnist_accelerator_top (

    input wire clk,
    input wire rst_n,
    input wire start,

    // FC1 input tile interface
    input wire tile_valid,
    input wire signed [127:0] activation_matrix,
    input wire signed [127:0] fc1_weight_matrix,
    input wire signed [127:0] fc1_bias_group,

    output wire tile_ready,
    output wire fc1_busy,

    // FC2 model parameters
    input wire [2559:0] fc2_weight_matrix,
    input wire [319:0] fc2_bias_vector,

    // Four images x ten INT32 logits
    output reg [1279:0] batch_logits,

    output reg busy,
    output reg done
);

    // --------------------------------------------
    // Internal FC1 signals
    // --------------------------------------------

    wire fc1_done;
    wire [1023:0] hidden_output;

    // --------------------------------------------
    // Internal FC2 signals
    // --------------------------------------------

    reg fc2_start;
    wire fc2_done;

    wire [255:0] selected_hidden;
    wire [319:0] fc2_logits;

    // --------------------------------------------
    // Controller registers
    // --------------------------------------------

    localparam IDLE       = 3'd0;
    localparam WAIT_FC1   = 3'd1;
    localparam FC2_START  = 3'd2;
    localparam FC2_WAIT   = 3'd3;
    localparam FC2_CAPTURE= 3'd4;
    localparam FINISH     = 3'd5;

    reg [2:0] state;
    reg [1:0] image_index;

    // --------------------------------------------
    // FC1 controller
    // --------------------------------------------

    fc1_controller fc1_inst (

        .clk(clk),
        .rst_n(rst_n),

        .start(start),

        .tile_valid(tile_valid),

        .activation_matrix(activation_matrix),
        .weight_matrix(fc1_weight_matrix),
        .bias_group(fc1_bias_group),

        .tile_ready(tile_ready),
        .busy(fc1_busy),
        .done(fc1_done),

        .hidden_output(hidden_output)

    );

    // --------------------------------------------
    // Select 32 hidden activations
    // for the current image
    // --------------------------------------------

    assign selected_hidden =
        hidden_output[image_index*256 +: 256];

    // --------------------------------------------
    // FC2 classifier
    // --------------------------------------------

    fc2_classifier fc2_inst (

        .clk(clk),
        .rst_n(rst_n),

        .start(fc2_start),

        .hidden_activations(selected_hidden),

        .weight_matrix(fc2_weight_matrix),
        .bias_vector(fc2_bias_vector),

        .logits(fc2_logits),
        .done(fc2_done)

    );

    // --------------------------------------------
    // Main controller
    // --------------------------------------------

    always @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            state <= IDLE;

            image_index <= 0;

            fc2_start <= 0;

            batch_logits <= 0;

            busy <= 0;
            done <= 0;

        end

        else begin

            // Default pulse signals
            fc2_start <= 0;
            done <= 0;

            case (state)

                // --------------------------------
                // Wait for new inference request
                // --------------------------------

                IDLE: begin

                    busy <= 0;

                    if (start) begin

                        image_index <= 0;

                        batch_logits <= 0;

                        busy <= 1;

                        state <= WAIT_FC1;

                    end

                end

                // --------------------------------
                // Wait for FC1 batch completion
                // --------------------------------

                WAIT_FC1: begin

                    if (fc1_done) begin

                        image_index <= 0;

                        state <= FC2_START;

                    end

                end

                // --------------------------------
                // Launch FC2 for current image
                // --------------------------------

                FC2_START: begin

                    fc2_start <= 1;

                    state <= FC2_WAIT;

                end

                // --------------------------------
                // Wait for FC2 completion
                // --------------------------------

                FC2_WAIT: begin

                    if (fc2_done) begin

                        state <= FC2_CAPTURE;

                    end

                end

                // --------------------------------
                // Store ten INT32 logits
                // --------------------------------

                FC2_CAPTURE: begin

                    batch_logits[
                        image_index*320 +: 320
                    ] <= fc2_logits;

                    if (image_index == 3) begin

                        state <= FINISH;

                    end

                    else begin

                        image_index <= image_index + 1'b1;

                        state <= FC2_START;

                    end

                end

                // --------------------------------
                // All four images completed
                // --------------------------------

                FINISH: begin

                    busy <= 0;

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
