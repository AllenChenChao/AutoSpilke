#ifndef DIFFERENTIAL_ENCODER_H
#define DIFFERENTIAL_ENCODER_H

#include "ap_int.h"

// 数据类型定义
typedef float input_t;                    // 输入数据类型
typedef float diff_t;               // 差分值类型
typedef ap_int<2> spike_t;               // 脉冲值类型 (-1, 0, 1)

// 差分编码函数
spike_t differential_encoder(
    input_t current_input,     // 当前输入值
    input_t previous_input,    // 前一个输入值
    input_t threshold          // 阈值
);

#endif // DIFFERENTIAL_ENCODER_H