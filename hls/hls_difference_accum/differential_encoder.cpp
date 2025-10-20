// differential_encoder.cpp
#include "differential_encoder.h"

spike_t differential_encoder(
    input_t current_input,
    input_t previous_input,
    input_t threshold,
    diff_t* cumulative_diff    // 新增参数
) {
    // 简洁的HLS接口
    #pragma HLS INTERFACE ap_ctrl_hs port=return
    #pragma HLS INTERFACE ap_none port=current_input
    #pragma HLS INTERFACE ap_none port=previous_input
    #pragma HLS INTERFACE ap_none port=threshold
    #pragma HLS INTERFACE ap_none port=cumulative_diff
    // #pragma HLS PIPELINE II=1
    
    // 计算差分值
    input_t difference = current_input - previous_input;
    
    // // 累积差分计算（通过指针更新外部变量）
    // if (cumulative_diff != nullptr) {
        *cumulative_diff += difference;
    // }
    
    // 原有差分编码逻辑
    spike_t spike_output;
    if (*cumulative_diff >= threshold) {
        *cumulative_diff = 0; 
        // *cumulative_diff -= threshold;
        spike_output = 1;           // 正脉冲
    } else if (*cumulative_diff <= -threshold) {
        *cumulative_diff = 0; 
        // *cumulative_diff += threshold;
        spike_output = -1;          // 负脉冲
    } else {
        spike_output = 0;           // 无脉冲
    }
    
    return spike_output;
}