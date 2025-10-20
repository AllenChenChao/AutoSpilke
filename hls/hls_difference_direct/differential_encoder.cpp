#include "differential_encoder.h"

spike_t differential_encoder(
    input_t current_input,
    input_t previous_input,
    input_t threshold
) {
    // HLS接口指令
    #pragma HLS INTERFACE ap_ctrl_hs port=return
    #pragma HLS INTERFACE ap_none port=current_input
    #pragma HLS INTERFACE ap_none port=previous_input
    #pragma HLS INTERFACE ap_none port=threshold
    // #pragma HLS PIPELINE II=1
    
    // 计算差分值
    input_t difference = current_input - previous_input;
    
    // 差分编码逻辑
    spike_t spike_output;
    
    if (difference >= threshold) {
        spike_output = 1;           // 正脉冲
    } else if (difference <= -threshold) {
        spike_output = -1;          // 负脉冲
    } else {
        spike_output = 0;           // 无脉冲
    }
    
    return spike_output;
}