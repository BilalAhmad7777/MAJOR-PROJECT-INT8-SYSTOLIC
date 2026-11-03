`timescale 1ns / 1ps

module tb_fc2_classifier;

    reg clk, rst_n, start;
    reg [255:0] hidden_activations;
    reg [2559:0] weight_matrix;
    reg [319:0] bias_vector;

    wire [319:0] logits;
    wire done;

    reg [7:0] hidden_mem [0:127];
    reg [7:0] weight_mem [0:319];
    reg [31:0] bias_mem [0:9];
    reg [31:0] expected_mem [0:39];

    integer image, i, passed, failed, index;

    fc2_classifier dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .hidden_activations(hidden_activations),
        .weight_matrix(weight_matrix),
        .bias_vector(bias_vector),
        .logits(logits),
        .done(done)
    );

    always #5 clk = ~clk;

    initial begin
        clk = 0;
        rst_n = 0;
        start = 0;
        passed = 0;
        failed = 0;

        $display("===== FC2 CLASSIFIER TEST =====");

        $readmemh("verification/test_vectors/fc2_hardware/fc2_hidden_int8.hex", hidden_mem);
        $readmemh("verification/test_vectors/fc2_hardware/fc2_weights_int8.hex", weight_mem);
        $readmemh("verification/test_vectors/fc2_hardware/fc2_bias_int32.hex", bias_mem);
        $readmemh("verification/test_vectors/fc2_hardware/fc2_expected_logits_int32.hex", expected_mem);

        for (image = 0; image < 4; image = image + 1) begin

            for (i = 0; i < 32; i = i + 1)
                hidden_activations[i*8 +: 8] = hidden_mem[image*32+i];

            for (i = 0; i < 320; i = i + 1)
                weight_matrix[i*8 +: 8] = weight_mem[i];

            for (i = 0; i < 10; i = i + 1)
                bias_vector[i*32 +: 32] = bias_mem[i];

            #20;
            rst_n = 1;
            #10;

            start = 1;
            @(posedge clk);
            #1;
            start = 0;

            wait(done == 1);
            #1;

            $display("\nImage %0d:", image);

            for (i = 0; i < 10; i = i + 1) begin
                index = image*10+i;

                if ($signed(logits[i*32 +: 32]) ==
                    $signed(expected_mem[index])) begin
                    passed = passed + 1;
                    $display("[PASS] Neuron %0d | RTL: %0d | Python: %0d",
                        i, $signed(logits[i*32 +: 32]),
                        $signed(expected_mem[index]));
                end
                else begin
                    failed = failed + 1;
                    $display("[FAIL] Neuron %0d | RTL: %0d | Python: %0d",
                        i, $signed(logits[i*32 +: 32]),
                        $signed(expected_mem[index]));
                end
            end

            rst_n = 0;
            #10;
        end

        $display("\n===== FC2 CLASSIFIER RESULTS =====");
        $display("Passed: %0d", passed);
        $display("Failed: %0d", failed);

        if (passed == 40 && failed == 0)
            $display("[PASS] FC2 RTL MATCHES PYTHON");
        else
            $display("[FAIL] FC2 RTL DOES NOT MATCH PYTHON");

        $finish;
    end

    initial begin
        #100000;
        $display("[FAIL] Simulation timeout");
        $finish;
    end

endmodule
