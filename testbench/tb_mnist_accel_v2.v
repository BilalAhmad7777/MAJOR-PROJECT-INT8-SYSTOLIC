`timescale 1ns / 1ps
// Full-system test of mnist_accel_v2_top against the existing golden vectors.
// Run from the project root:   vvp simulation/mnist_accel_v2.vvp +BATCHES=125
// (+BATCHES=n runs n batches of 8 images, default 125 = all 1000 images)
module tb_mnist_accel_v2;

    localparam B = 8;
    localparam WORDS_PER_BATCH = B*196;

    reg         clk = 1'b0;
    reg         rst_n = 1'b0;
    reg         img_wr_en = 1'b0;
    reg  [10:0] img_wr_addr = 11'd0;
    reg  [31:0] img_wr_data = 32'd0;
    reg         start = 1'b0;
    wire        busy, done;
    wire [B*320-1:0] batch_logits;
    wire [B*4-1:0]   predicted_class;
    wire [31:0] perf_cycles, perf_mac_ops;

    always #5 clk = ~clk;

    mnist_accel_v2_top #(.BATCH(B)) dut (
        .clk(clk), .rst_n(rst_n),
        .img_wr_en(img_wr_en), .img_wr_addr(img_wr_addr), .img_wr_data(img_wr_data),
        .start(start), .busy(busy), .done(done),
        .batch_logits(batch_logits), .predicted_class(predicted_class),
        .perf_cycles(perf_cycles), .perf_mac_ops(perf_mac_ops)
    );

    reg [31:0] img_words  [0:1000*196-1];
    reg [31:0] exp_logits [0:9999];
    reg [3:0]  exp_pred   [0:999];
    reg [3:0]  labels     [0:999];

    integer num_batches, b, i, n, j, g;
    integer passed, failed, correct, images_done;
    integer got_pred, q, peak_active, qq;
    integer last_cycles, last_macs;
    reg signed [31:0] got_logit;
    real macs_per_cycle;

    // ---- peak simultaneous PE activity (hierarchical probe) ----
    always @(posedge clk) begin
        q = 0;
        for (qq = 0; qq < 16; qq = qq + 1) q = q + dut.pe_active[qq];
        if (q > peak_active) peak_active = q;
    end

    task load_batch;
        input integer bb;
        integer w;
        begin
            for (w = 0; w < WORDS_PER_BATCH; w = w + 1) begin
                @(negedge clk);
                img_wr_en   = 1'b1;
                img_wr_addr = w;
                img_wr_data = img_words[bb*WORDS_PER_BATCH + w];
            end
            @(negedge clk);
            img_wr_en = 1'b0;
        end
    endtask

    task pulse_start;
        begin
            @(negedge clk); start = 1'b1;
            @(negedge clk); start = 1'b0;
        end
    endtask

    task check_batch;
        input integer bb;
        integer ii, nn, gg, best;
        reg signed [31:0] lg;
        begin
            for (ii = 0; ii < B; ii = ii + 1) begin
                gg = bb*B + ii;
                for (nn = 0; nn < 10; nn = nn + 1) begin
                    lg = batch_logits[(ii*10 + nn)*32 +: 32];
                    if (lg === exp_logits[gg*10 + nn]) passed = passed + 1;
                    else begin
                        failed = failed + 1;
                        if (failed <= 10)
                            $display("[FAIL] image %0d logit %0d : got %0d expected %0d",
                                     gg, nn, lg, $signed(exp_logits[gg*10 + nn]));
                    end
                end
                best = predicted_class[ii*4 +: 4];
                if (best == exp_pred[gg]) passed = passed + 1;
                else begin
                    failed = failed + 1;
                    if (failed <= 10)
                        $display("[FAIL] image %0d prediction : got %0d expected %0d", gg, best, exp_pred[gg]);
                end
                if (best == labels[gg]) correct = correct + 1;
                images_done = images_done + 1;
            end
        end
    endtask

    initial begin
        passed = 0; failed = 0; correct = 0; images_done = 0; peak_active = 0;
        if (!$value$plusargs("BATCHES=%d", num_batches)) num_batches = 125;
        if (num_batches > 125) num_batches = 125;

        $readmemh("rom/images_1000_words.hex", img_words);
        $readmemh("verification/test_vectors/mnist_1000/expected_logits_int32.hex", exp_logits);
        $readmemh("verification/test_vectors/mnist_1000/predictions_python.txt", exp_pred);
        $readmemh("verification/test_vectors/mnist_1000/labels.txt", labels);

        $display("==========================================");
        $display("  MNIST ACCELERATOR v2 (streaming array)");
        $display("  %0d batches x %0d images", num_batches, B);
        $display("==========================================");

        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (4) @(negedge clk);

        // ---- reset-recovery test: interrupt a run, reset, then run batch 0 normally ----
        load_batch(0);
        pulse_start;
        repeat (3000) @(negedge clk);
        rst_n = 1'b0;
        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        repeat (3) @(negedge clk);
        if (busy !== 1'b0) begin
            failed = failed + 1;
            $display("[FAIL] busy not cleared by reset");
        end else passed = passed + 1;

        // ---- normal batches (back to back) ----
        for (b = 0; b < num_batches; b = b + 1) begin
            load_batch(b);
            pulse_start;
            wait (done === 1'b1);
            @(negedge clk);
            check_batch(b);
            last_cycles = perf_cycles;
            last_macs   = perf_mac_ops;
            if ((b % 25) == 0)
                $display("... batch %0d done, %0d cycles", b, perf_cycles);
        end

        macs_per_cycle = last_macs;
        macs_per_cycle = macs_per_cycle / last_cycles;

        $display("------------------------------------------");
        $display("images tested          : %0d", images_done);
        $display("correct vs MNIST labels: %0d / %0d", correct, images_done);
        $display("cycles per batch (8 img): %0d  -> %0d cycles/image", last_cycles, last_cycles / B);
        $display("PE MAC operations/batch : %0d (expected %0d)", last_macs, (1568*B + 24*B)*16);
        $display("average MACs per cycle  : %0.2f of 16 (peak seen: %0d)", macs_per_cycle, peak_active);
        $display("------------------------------------------");
        $display("PASS: %0d   FAIL: %0d", passed, failed);
        if (failed == 0) $display("RESULT: PASS (bit-exact vs Python golden model)");
        else             $display("RESULT: FAIL");
        $display("------------------------------------------");
        $finish;
    end

    initial begin
        #400000000;
        $display("TIMEOUT");
        $finish;
    end

endmodule
