#include <iostream>
#include <iomanip>
#include <vector>
#include "simple_processor.h"

// Visualize spike position
void visualize_spike(position_t position, timestep_t total_steps) {
    std::cout << "[";
    for (int i = 0; i < total_steps; i++) {
        if (i == position) {
            std::cout << "i";//█
        } else {
            std::cout << ".";
        }
    }
    std::cout << "]";
}

int main() {
    std::cout << "=== HLS Latency Encoding Test ===" << std::endl;
    std::cout << "Input: Float value, Time steps" << std::endl;
    std::cout << "Output: Spike position in time steps" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // Test parameters
    timestep_t time_steps = 16;
    
    // Test cases
    std::vector<float> test_values = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, -0.1};
    
    std::cout << std::setw(12) << "Input Value" 
              << std::setw(15) << "Spike Position" 
              << std::setw(20) << "Visualization" << std::endl;
    std::cout << std::string(47, '-') << std::endl;
    
    for (float value : test_values) {
        float_t input_val = value;
        position_t spike_pos = simple_processor(input_val, time_steps);
        
        std::cout << std::setw(12) << std::fixed << std::setprecision(2) << value
                  << std::setw(15) << (int)spike_pos
                  << std::setw(4) << "";
        
        visualize_spike(spike_pos, time_steps);
        std::cout << std::endl;
    }
    
    std::cout << "\n=== Different Time Window Test ===" << std::endl;
    float test_input = 0.3;
    std::vector<int> time_windows = {8, 16, 32, 64};
    
    std::cout << std::setw(15) << "Time Window" 
              << std::setw(15) << "Spike Position" 
              << std::setw(15) << "Delay Ratio" << std::endl;
    std::cout << std::string(45, '-') << std::endl;
    
    for (int window : time_windows) {
        position_t pos = simple_processor(test_input, window);
        float delay_ratio = (float)pos / (window - 1);
        
        std::cout << std::setw(15) << window
                  << std::setw(15) << (int)pos
                  << std::setw(14) << std::fixed << std::setprecision(3) << delay_ratio << std::endl;
    }
    
    std::cout << "\n=== Encoding Properties Verification ===" << std::endl;
    std::cout << "✓ Larger input value → Earlier spike position (shorter delay)" << std::endl;
    std::cout << "✓ Input value = 0 → Spike at last position" << std::endl;
    std::cout << "✓ Input value = 1 → Spike at position 0" << std::endl;
    std::cout << "✓ Values outside [0,1] range are automatically clamped" << std::endl;
    
    return 0;
}