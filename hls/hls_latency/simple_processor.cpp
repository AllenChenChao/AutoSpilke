#include "simple_processor.h"

position_t simple_processor(
    float_t input_value,
    timestep_t time_steps
) {
    // HLS接口指令
    #pragma HLS INTERFACE ap_ctrl_hs port=return
    #pragma HLS INTERFACE ap_none port=input_value
    #pragma HLS INTERFACE ap_none port=time_steps

    
    // 输入值归一化到[0,1]范围
    float_t normalized_input;
    if (input_value < 0) {
        normalized_input = 0;
    } else if (input_value > 1.0) {
        normalized_input = 1.0;
    } else {
        normalized_input = input_value;
    }

    float_t delay_float = (1 - normalized_input) * (int)(time_steps - 1);

    position_t spike_position = (position_t)delay_float;

    // 边界检查
    if (spike_position >= time_steps) {
        spike_position = time_steps - 1;
    }
    
    return spike_position;    

}