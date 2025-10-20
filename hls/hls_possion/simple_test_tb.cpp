#include "simple_test.h"
#include <iostream>
#include <iomanip>



void test_simple_add_uint8() {
    std::cout << "\n=== Test simple_add_uint8 Function ===" << std::endl;
    std::cout << "in  |thres|  next  | spike  " << std::endl;
    std::cout << "----|-----|--------|--------" << std::endl;
    
    ap_uint<16> test_thresholds[] = {655, 1310, 3276, 13107, 32768, 52428, 65535};
    ap_uint<16> a = 1000;

    for (int i = 0; i < 7; i++) {
        ap_uint<16> threshold = test_thresholds[i];
        int spike;
        ap_uint<16> result = simple_add(a, threshold, spike);
        
        std::cout << std::setw(5) << (int)a << " | ";
        std::cout << std::setw(7) << (int)threshold << " | ";
        std::cout << std::setw(8) << (int)result << " | ";
        std::cout << std::setw(5) << spike << std::endl;

        a = result;

    }
}





int main() {
    std::cout << "Simple HLS Test for simple_add_uint8 Function" << std::endl;
    std::cout << "======================" << std::endl;
    
    // 运行所有测试

    test_simple_add_uint8();

    
    std::cout << "\nAll Tests Completed!" << std::endl;
    std::cout << "If all tests display ✓, the basic interface works normally." << std::endl;
    std::cout << "You can continue testing the more complex Poisson encoding interface." << std::endl;
    
    return 0;
}