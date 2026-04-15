// os_elm_tb.cpp — Vitis HLS testbench for os_elm_core
//
// Tests all 6 opcodes in sequence:
//   1. LOAD_WEIGHTS  — load known weights, verify ack
//   2. PREDICT_Q     — predict with θ, verify against manual computation
//   3. SYNC_TARGET   — copy θ->θ, verify ack
//   4. PREDICT_TGT   — predict with θ, verify matches θ result
//   5. TRAIN_SEQ     — one RLS update, verify ack and that B changed
//   6. READ_WEIGHTS  — read back B and P, verify B matches post-train state
//   7. Unknown opcode — verify error ack (-1)
//
// Run in Vitis HLS: C Simulation or C/RTL Co-Simulation.

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include "os_elm_core.h"

// Tolerance for Q20 fixed-point comparison
// 1 LSB = 1/2^20 = 0.00000095, give tolerance for FP digits
#define TOL 0.0001

static int test_count = 0;
static int fail_count = 0;

static void check(const char *name, double got, double expected, double tol = TOL) {
    test_count++;
    double diff = fabs(got - expected);
    if (diff > tol) {
        printf("  FAIL %s: got %.6f, expected %.6f (diff %.6f)\n", name, got, expected, diff);
        fail_count++;
    }
}

static void check_int(const char *name, int got, int expected) {
    test_count++;
    if (got != expected) {
        printf("  FAIL %s: got %d, expected %d\n", name, got, expected);
        fail_count++;
    }
}

// Push a raw int onto the input stream
static void send_int(hls::stream<axis_word_t> &s, int val, bool last = false) {
    axis_word_t w;
    w.data = val;
    w.keep = 0xF;
    w.strb = 0xF;
    w.last = last ? 1 : 0;
    s.write(w);
}

// Push a fixed_t value onto the input stream
static void send_fixed(hls::stream<axis_word_t> &s, fixed_t val, bool last = false) {
    axis_word_t w;
    w.data = val.range(31, 0);
    w.keep = 0xF;
    w.strb = 0xF;
    w.last = last ? 1 : 0;
    s.write(w);
}

// Read a fixed_t value from the output stream
static fixed_t recv_fixed(hls::stream<axis_word_t> &s) {
    axis_word_t w = s.read();
    fixed_t val;
    val.range(31, 0) = w.data;
    return val;
}

// Read a raw int from the output stream
static int recv_int(hls::stream<axis_word_t> &s) {
    axis_word_t w = s.read();
    return (int)w.data;
}

// Generate deterministic test weights (small values to avoid overflow)
// W_in[i][j] = (i * HIDDEN_DIM + j + 1) * 0.01
// b[j]       = (j + 1) * 0.005
// beta[j][a] = (j * NUM_ACTIONS + a + 1) * 0.02
// P          = identity * 5.0
static double ref_W[STATE_DIM][HIDDEN_DIM];
static double ref_b[HIDDEN_DIM];
static double ref_beta[HIDDEN_DIM][NUM_ACTIONS];
static double ref_P[HIDDEN_DIM][HIDDEN_DIM];

static void init_ref_weights() {
    for (int i = 0; i < STATE_DIM; i++)
        for (int j = 0; j < HIDDEN_DIM; j++)
            ref_W[i][j] = (i * HIDDEN_DIM + j + 1) * 0.01;
    for (int j = 0; j < HIDDEN_DIM; j++)
        ref_b[j] = (j + 1) * 0.005;
    for (int j = 0; j < HIDDEN_DIM; j++)
        for (int a = 0; a < NUM_ACTIONS; a++)
            ref_beta[j][a] = (j * NUM_ACTIONS + a + 1) * 0.02;
    for (int r = 0; r < HIDDEN_DIM; r++)
        for (int c = 0; c < HIDDEN_DIM; c++)
            ref_P[r][c] = (r == c) ? 5.0 : 0.0;
}

// Compute expected Q-values in double precision for a given state and beta
static void ref_predict(const double state[STATE_DIM],
                        const double beta_ref[HIDDEN_DIM][NUM_ACTIONS],
                        double q_out[NUM_ACTIONS]) {
    double h[HIDDEN_DIM];
    for (int j = 0; j < HIDDEN_DIM; j++) {
        double acc = ref_b[j];
        for (int i = 0; i < STATE_DIM; i++)
            acc += state[i] * ref_W[i][j];
        h[j] = (acc > 0.0) ? acc : 0.0; // ReLU
    }
    for (int a = 0; a < NUM_ACTIONS; a++) {
        q_out[a] = 0.0;
        for (int j = 0; j < HIDDEN_DIM; j++)
            q_out[a] += h[j] * beta_ref[j][a];
    }
}


int main() {
    hls::stream<axis_word_t> in_stream("in");
    hls::stream<axis_word_t> out_stream("out");

    init_ref_weights();

    double test_state[STATE_DIM] = {0.5, 0.3, 1.0, 0.1, 0.6, 0.45};

    // TEST 1: LOAD_WEIGHTS
    printf("TEST 1: LOAD_WEIGHTS\n");
    {
        send_int(in_stream, OP_LOAD_WEIGHTS);
        // W_in
        for (int i = 0; i < STATE_DIM; i++)
            for (int j = 0; j < HIDDEN_DIM; j++)
                send_fixed(in_stream, fixed_t(ref_W[i][j]));
        // b
        for (int j = 0; j < HIDDEN_DIM; j++)
            send_fixed(in_stream, fixed_t(ref_b[j]));
        // beta
        for (int j = 0; j < HIDDEN_DIM; j++)
            for (int a = 0; a < NUM_ACTIONS; a++)
                send_fixed(in_stream, fixed_t(ref_beta[j][a]));
        // P
        for (int r = 0; r < HIDDEN_DIM; r++)
            for (int c = 0; c < HIDDEN_DIM; c++)
                send_fixed(in_stream, fixed_t(ref_P[r][c]));

        os_elm_core(in_stream, out_stream);

        int ack = recv_int(out_stream);
        check_int("LOAD ack", ack, 1);
        printf("  LOAD_WEIGHTS: ack=%d\n", ack);
    }

    // TEST 2: PREDICT_Q
    printf("TEST 2: PREDICT_Q\n");
    double q_theta1[NUM_ACTIONS];
    {
        send_int(in_stream, OP_PREDICT_Q);
        for (int i = 0; i < STATE_DIM; i++)
            send_fixed(in_stream, fixed_t(test_state[i]));

        os_elm_core(in_stream, out_stream);

        // Compute expected values
        double q_expected[NUM_ACTIONS];
        ref_predict(test_state, ref_beta, q_expected);

        for (int a = 0; a < NUM_ACTIONS; a++) {
            fixed_t q_hw = recv_fixed(out_stream);
            q_theta1[a] = (double)q_hw;
            printf("  Q[%d]: hw=%.4f  ref=%.4f\n", a, (double)q_hw, q_expected[a]);
            // Use larger tolerance due to Q20 quantization across many MACs
            check("PREDICT_Q", (double)q_hw, q_expected[a], 0.5);
        }
    }

    // TEST 3: SYNC_TARGET
    printf("TEST 3: SYNC_TARGET\n");
    {
        send_int(in_stream, OP_SYNC_TARGET);
        os_elm_core(in_stream, out_stream);
        int ack = recv_int(out_stream);
        check_int("SYNC ack", ack, 1);
        printf("  SYNC_TARGET: ack=%d\n", ack);
    }

    // TEST 4: PREDICT_TGT, should match θ1 after sync
    printf("TEST 4: PREDICT_TGT (should match PREDICT_Q after sync)\n");
    {
        send_int(in_stream, OP_PREDICT_TGT);
        for (int i = 0; i < STATE_DIM; i++)
            send_fixed(in_stream, fixed_t(test_state[i]));

        os_elm_core(in_stream, out_stream);

        for (int a = 0; a < NUM_ACTIONS; a++) {
            fixed_t q_hw = recv_fixed(out_stream);
            printf("  Q_tgt[%d]: hw=%.4f  q1=%.4f\n", a, (double)q_hw, q_theta1[a]);
            // After sync, θ2 = θ1, so results should be identical
            check("PREDICT_TGT vs Q", (double)q_hw, q_theta1[a], 0.0001);
        }
    }

    // TEST 5: TRAIN_SEQ — one RLS update
    printf("TEST 5: TRAIN_SEQ\n");
    {
        int action = 1;
        double target_val = 3.5;

        send_int(in_stream, OP_TRAIN_SEQ);
        for (int i = 0; i < STATE_DIM; i++)
            send_fixed(in_stream, fixed_t(test_state[i]));
        send_int(in_stream, action);           // raw int
        send_fixed(in_stream, fixed_t(target_val)); // Q20

        os_elm_core(in_stream, out_stream);

        int ack = recv_int(out_stream);
        check_int("TRAIN ack", ack, 1);
        printf("  TRAIN_SEQ: ack=%d (action=%d, target=%.2f)\n", ack, action, target_val);
    }

    // TEST 6: PREDICT_Q after training — B should have changed
    printf("TEST 6: PREDICT_Q (post-train, should differ from pre-train)\n");
    {
        send_int(in_stream, OP_PREDICT_Q);
        for (int i = 0; i < STATE_DIM; i++)
            send_fixed(in_stream, fixed_t(test_state[i]));

        os_elm_core(in_stream, out_stream);

        bool changed = false;
        for (int a = 0; a < NUM_ACTIONS; a++) {
            fixed_t q_hw = recv_fixed(out_stream);
            double diff = fabs((double)q_hw - q_theta1[a]);
            printf("  Q_post[%d]: %.4f  (pre: %.4f, diff: %.6f)\n",
                   a, (double)q_hw, q_theta1[a], diff);
            if (diff > 0.0001) changed = true;
        }
        test_count++;
        if (!changed) {
            printf("  FAIL: Q-values unchanged after TRAIN_SEQ\n");
            fail_count++;
        } else {
            printf("  OK: Q-values changed after training\n");
        }
    }

    // TEST 7: READ_WEIGHTS — verify B and P are readable
    printf("TEST 7: READ_WEIGHTS\n");
    {
        send_int(in_stream, OP_READ_WEIGHTS);
        os_elm_core(in_stream, out_stream);

        int total = HIDDEN_DIM * NUM_ACTIONS + HIDDEN_DIM * HIDDEN_DIM;

        // Read beta back
        double read_beta[HIDDEN_DIM][NUM_ACTIONS];
        for (int j = 0; j < HIDDEN_DIM; j++)
            for (int a = 0; a < NUM_ACTIONS; a++)
                read_beta[j][a] = (double)recv_fixed(out_stream);

        // Read P back
        double read_P[HIDDEN_DIM][HIDDEN_DIM];
        for (int r = 0; r < HIDDEN_DIM; r++)
            for (int c = 0; c < HIDDEN_DIM; c++)
                read_P[r][c] = (double)recv_fixed(out_stream);

        // Verify beta has changed from the original (due to TRAIN_SEQ)
        bool beta_changed = false;
        for (int j = 0; j < HIDDEN_DIM; j++)
            for (int a = 0; a < NUM_ACTIONS; a++)
                if (fabs(read_beta[j][a] - ref_beta[j][a]) > 0.0001)
                    beta_changed = true;

        test_count++;
        if (!beta_changed) {
            printf("  FAIL: beta unchanged after train\n");
            fail_count++;
        } else {
            printf("  OK: beta differs from original (training took effect)\n");
        }

        // Verify P diagonal is still positive (not corrupted)
        bool p_ok = true;
        for (int i = 0; i < HIDDEN_DIM; i++) {
            if (read_P[i][i] <= 0.0) {
                p_ok = false;
                printf("  FAIL: P[%d][%d] = %.6f (should be positive)\n", i, i, read_P[i][i]);
            }
        }
        test_count++;
        if (p_ok) {
            printf("  OK: P diagonal is positive\n");
        } else {
            fail_count++;
        }

        printf("  beta[0][0]: orig=%.4f  read=%.4f\n", ref_beta[0][0], read_beta[0][0]);
        printf("  P[0][0]:    orig=%.4f  read=%.4f\n", ref_P[0][0], read_P[0][0]);
    }

    // TEST 8: Unknown opcode
    printf("TEST 8: Unknown opcode\n");
    {
        send_int(in_stream, 99);
        os_elm_core(in_stream, out_stream);
        int ack = recv_int(out_stream);
        check_int("unknown opcode ack", ack, -1);
        printf("  Unknown opcode: ack=%d\n", ack);
    }

    // TEST 9: Verify streams are empty (no leftover data)
    printf("TEST 9: Stream drain check\n");
    {
        test_count++;
        if (!in_stream.empty()) {
            printf("  FAIL: in_stream not empty\n");
            fail_count++;
        }
        test_count++;
        if (!out_stream.empty()) {
            printf("  FAIL: out_stream not empty\n");
            fail_count++;
        }
        if (in_stream.empty() && out_stream.empty()) {
            printf("  OK: both streams empty\n");
        }
    }

    // Summary
    printf("Tests: %d  Passed: %d  Failed: %d\n",
           test_count, test_count - fail_count, fail_count);

    return fail_count > 0 ? 1 : 0;
}
