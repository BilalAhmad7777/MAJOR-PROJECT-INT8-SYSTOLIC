`timescale 1ns / 1ps

module tb_fc2_arithmetic_audit;

    reg clk;
    reg rst_n;
    reg start;

    reg [255:0] hidden_activations;
    reg [2559:0] weight_matrix;
    reg [319:0] bias_vector;

    wire [319:0] logits;
    wire done;

    integer passed;
    integer failed;
    integer scenario;
    integer n;
    integer k;
    integer timeout_count;

    reg signed [7:0] activation_value;
    reg signed [7:0] weight_value;
    reg signed [31:0] bias_value;
    reg signed [63:0] reference_sum;
    reg [31:0] expected_value;

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

    task run_scenario;
        input integer scenario_id;
        begin
            // Build deterministic input vectors.
            for (k = 0; k < 32; k = k + 1) begin
                case (scenario_id)
                    0: begin
                        case (k % 4)
                            0: activation_value = -128;
                            1: activation_value = 127;
                            2: activation_value = -1;
                            3: activation_value = 0;
                        endcase
                    end

                    1: activation_value = -128;

                    2: activation_value = 127;

                    default: activation_value = (k % 2) ? -127 : 126;
                endcase

                hidden_activations[k*8 +: 8] = activation_value;
            end

            for (n = 0; n < 10; n = n + 1) begin
                for (k = 0; k < 32; k = k + 1) begin
                    case (scenario_id)
                        0: begin
                            case ((n+k) % 4)
                                0: weight_value = -128;
                                1: weight_value = 127;
                                2: weight_value = -1;
                                3: weight_value = 1;
                            endcase
                        end

                        1: weight_value = -128;

                        2: weight_value = -128;

                        default: weight_value = (k % 2) ? 127 : -127;
                    endcase

                    weight_matrix[n*256 + k*8 +: 8] = weight_value;
                end

                case (scenario_id)
                    0: bias_value = (n % 2) ? -100000 : 100000;
                    1: bias_value = 32'h7fffffff;
                    2: bias_value = 32'h80000000;
                    default: bias_value = (n % 2) ? -2147483647 : 2147483646;
                endcase

                bias_vector[n*32 +: 32] = bias_value;
            end

            // Start the actual RTL.
            @(negedge clk);
            start = 1'b1;

            @(negedge clk);
            start = 1'b0;

            // Wait for completion with timeout protection.
            timeout_count = 0;

            while (done !== 1'b1 && timeout_count < 1000) begin
                @(negedge clk);
                timeout_count = timeout_count + 1;
            end

            if (done !== 1'b1) begin
                $display("[FAIL] Scenario %0d: timeout", scenario_id);
                failed = failed + 1;
            end
            else begin
                // Independently calculate all 10 neuron outputs.
                for (n = 0; n < 10; n = n + 1) begin
                    reference_sum = 0;

                    for (k = 0; k < 32; k = k + 1) begin
                        activation_value =
                            $signed(hidden_activations[k*8 +: 8]);

                        weight_value =
                            $signed(weight_matrix[n*256 + k*8 +: 8]);

                        reference_sum = reference_sum +
                            activation_value * weight_value;
                    end

                    bias_value =
                        $signed(bias_vector[n*32 +: 32]);

                    reference_sum = reference_sum + bias_value;
                    expected_value = reference_sum[31:0];

                    if (logits[n*32 +: 32] === expected_value) begin
                        passed = passed + 1;
                    end
                    else begin
                        failed = failed + 1;

                        $display(
                            "[FAIL] Scenario %0d Neuron %0d Expected=%0d RTL=%0d",
                            scenario_id,
                            n,
                            $signed(expected_value),
                            $signed(logits[n*32 +: 32])
                        );
                    end
                end

                $display(
                    "[INFO] Scenario %0d completed: 10 neuron outputs checked",
                    scenario_id
                );
            end
        end
    endtask

    initial begin
        clk = 0;
        rst_n = 0;
        start = 0;

        hidden_activations = 0;
        weight_matrix = 0;
        bias_vector = 0;

        passed = 0;
        failed = 0;

        repeat (3) @(negedge clk);
        rst_n = 1;

        for (scenario = 0; scenario < 4; scenario = scenario + 1) begin
            run_scenario(scenario);
        end

        $display("\n========================================");
        $display("FC2 ARITHMETIC AUDIT RESULTS");
        $display("========================================");
        $display("Scenarios: 4");
        $display("Outputs checked: 40");
        $display("PASS: %0d", passed);
        $display("FAIL: %0d", failed);

        if (passed == 40 && failed == 0)
            $display("[PASS] FC2 ARITHMETIC AUDIT PASSED");
        else
            $display("[FAIL] FC2 ARITHMETIC AUDIT FAILED");

        $finish;
    end

endmodule
