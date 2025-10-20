#include "simple_test.h"


// 带ap_uint的版本
ap_uint<16> simple_add(ap_uint<16> input_state, ap_uint<16> threshold, int &spike) {
    #pragma HLS INTERFACE ap_ctrl_hs port=return
    #pragma HLS INTERFACE ap_none port=input_state
    #pragma HLS INTERFACE ap_none port=threshold
    #pragma HLS INTERFACE ap_vld port=spike

    ap_uint<16> state = input_state;
    
    if (state == 0) {
        state = DEFAULT_SEED;
    }

    // LFSR步骤
    ap_uint<1> feedback = state & 1;
    state = state >> 1;
    if (feedback) {
        state ^= LFSR_TAPS;
    }

    spike = (state < threshold)? 1 : 0;
    // return input_state + threshold;
    // return state + threshold;
    return state;
}
