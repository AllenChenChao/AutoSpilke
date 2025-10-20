#ifndef SIMPLE_PROCESSOR_H
#define SIMPLE_PROCESSOR_H

#include "ap_int.h"

// 数据类型定义
typedef float float_t;              // 标准32位浮点数
typedef ap_uint<16> timestep_t;           // 16位无符号整数
typedef ap_uint<16> position_t;           // 16位无符号整数


// 简单处理器函数声明
position_t simple_processor(
    float_t input_value,      // 输入浮点数
    timestep_t time_steps  // 控制参数
);

#endif // SIMPLE_PROCESSOR_H