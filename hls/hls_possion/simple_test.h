#ifndef SIMPLE_TEST_H
#define SIMPLE_TEST_H

// #include "ap_int.h"

// #ifdef __VIVADO_SYNTH__
    #include <ap_int.h>
    #include <ap_fixed.h>
// #else
//     // Software simulation types
//     #include <cstdint>
//     typedef uint32_t ap_uint32;
//     template<int W> using ap_uint = uint32_t;
//     template<int W, int I> using ap_ufixed = uint32_t;
// #endif

// #define DEFAULT_SEED 0x12

const ap_uint<16> LFSR_TAPS = 0xB400;    // 16bit LFSR多项式: x^16 + x^14 + x^13 + x^11 + 1
const ap_uint<16> DEFAULT_SEED = 0xACE1;  // 16bit默认种子


// 带ap_uint的版本
ap_uint<16> simple_add(ap_uint<16> a, ap_uint<16> b, int &spike);



#endif