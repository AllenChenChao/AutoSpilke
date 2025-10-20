#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "differential_encoder.h"

// 可视化脉冲序列（不变）
void visualize_spikes(const std::vector<spike_t>& spikes) {
    std::cout << "[";
    for (size_t i = 0; i < spikes.size(); i++) {
        if (spikes[i] == 1) {
            std::cout << "+";
        } else if (spikes[i] == -1) {
            std::cout << "-";
        } else {
            std::cout << "·";
        }
    }
    std::cout << "]";
}

// 生成测试信号（不变）
std::vector<float> generate_test_signal(int length) {
    std::vector<float> signal(length);
    for (int i = 0; i < length; i++) {
        float t = (float)i / length;
        signal[i] = 0.5 * t                    // 线性增长
                  + 0.2 * sin(2 * 3.14159 * t * 3)  // 正弦波
                  + (i > length/2 ? 0.3 : 0.0); // 阶跃变化
    }
    return signal;
}

int main() {
    std::cout << "=== Differential Encoder HLS Test ===" << std::endl;
    std::cout << "Logic: spike = +1 if diff >= threshold" << std::endl;
    std::cout << "       spike = -1 if diff <= -threshold" << std::endl;
    std::cout << "       spike =  0 otherwise" << std::endl;
    std::cout << std::string(65, '=') << std::endl;
    
    // 测试参数
    input_t threshold = 0.1f;
    int signal_length = 20;
    diff_t cumulative_diff = 0.0f;  // 初始化累积差分变量
    
    // 生成测试信号
    std::vector<float> test_signal = generate_test_signal(signal_length);
    std::vector<spike_t> spike_train;
    
    std::cout << "\nTest Signal and Encoding Results:" << std::endl;
    // 增加累积差分列
    std::cout << std::setw(6) << "Step" 
              << std::setw(12) << "Signal" 
              << std::setw(12) << "Difference"
              << std::setw(8) << "Spike" 
              << std::setw(18) << "Cumulative Diff"  // 新增列
              << std::setw(15) << "Explanation" << std::endl;
    std::cout << std::string(71, '-') << std::endl;
    
    // 差分编码测试（修改部分）
    for (int i = 0; i < signal_length; i++) {
        input_t current = test_signal[i];
        input_t previous = (i == 0) ? 0.0f : test_signal[i-1];
        
        // 调用修改后的编码器，传入累积差分指针
        spike_t spike = differential_encoder(current, previous, threshold, &cumulative_diff);
        spike_train.push_back(spike);
        
        input_t diff = current - previous;
        
        std::string explanation;
        if (spike == 1) explanation = "Positive spike";
        else if (spike == -1) explanation = "Negative spike";
        else explanation = "No spike";
        
        // 输出累积差分
        std::cout << std::setw(6) << i
                  << std::setw(12) << std::fixed << std::setprecision(3) << current
                  << std::setw(12) << std::fixed << std::setprecision(3) << diff
                  << std::setw(8) << (int)spike
                  << std::setw(18) << std::fixed << std::setprecision(3) << cumulative_diff  // 新增输出
                  << "  " << explanation << std::endl;
    }
    
    // 以下部分保持不变，省略...
    std::cout << "\nSpike Train Visualization:" << std::endl;
    std::cout << "Pattern: ";
    visualize_spikes(spike_train);
    std::cout << std::endl;
    
    // 统计信息
    int positive_spikes = 0, negative_spikes = 0, no_spikes = 0;
    for (spike_t spike : spike_train) {
        if (spike == 1) positive_spikes++;
        else if (spike == -1) negative_spikes++;
        else no_spikes++;
    }
    
    std::cout << "\nEncoding Statistics:" << std::endl;
    std::cout << "  Positive spikes: " << positive_spikes << std::endl;
    std::cout << "  Negative spikes: " << negative_spikes << std::endl;
    std::cout << "  No spikes: " << no_spikes << std::endl;
    std::cout << "  Sparsity: " << std::fixed << std::setprecision(1) 
              << (float)(positive_spikes + negative_spikes) / signal_length * 100 << "%" << std::endl;
    
    // 不同阈值测试（需要增加累积差分变量）
    std::cout << "\n=== Threshold Sensitivity Test ===" << std::endl;
    std::vector<float> thresholds = {0.05f, 0.1f, 0.2f, 0.3f};
    
    std::cout << std::setw(12) << "Threshold" 
              << std::setw(15) << "Total Spikes" 
              << std::setw(12) << "Sparsity%" << std::endl;
    std::cout << std::string(39, '-') << std::endl;
    
    for (float thresh : thresholds) {
        int total_spikes = 0;
        diff_t temp_cumulative = 0.0f;  // 每个阈值测试使用独立的累积变量
        
        for (int i = 1; i < signal_length; i++) {
            spike_t spike = differential_encoder(test_signal[i], test_signal[i-1], thresh, &temp_cumulative);
            if (spike != 0) total_spikes++;
        }
        
        float sparsity = (float)total_spikes / (signal_length - 1) * 100;
        
        std::cout << std::setw(12) << std::fixed << std::setprecision(3) << thresh
                  << std::setw(15) << total_spikes
                  << std::setw(11) << std::fixed << std::setprecision(1) << sparsity << "%" << std::endl;
    }
    
    // 边界测试（增加累积差分变量）
    std::cout << "\n=== Boundary Tests ===" << std::endl;
    
    struct BoundaryTest {
        float current, previous, threshold;
        const char* description;
    } boundary_tests[] = {
        {1.0f, 0.9f, 0.05f, "Just above threshold"},
        {1.0f, 0.95f, 0.05f, "Exactly at threshold"},
        {1.0f, 0.96f, 0.05f, "Just below threshold"},
        {0.5f, 0.7f, 0.1f, "Negative difference"},
        {0.0f, 0.0f, 0.1f, "No change"},
        {-1.0f, 1.0f, 0.5f, "Large negative change"}
    };
    
    for (auto& test : boundary_tests) {
        diff_t test_cumulative = 0.0f;  // 每个边界测试使用独立的累积变量
        spike_t result = differential_encoder(test.current, test.previous, test.threshold, &test_cumulative);
        float diff = test.current - test.previous;
        
        std::cout << test.description << ": "
                  << "diff=" << std::fixed << std::setprecision(3) << diff
                  << ", cumulative=" << test_cumulative
                  << " → spike=" << (int)result << std::endl;
    }
    
    std::cout << "\n✅ Differential Encoding Test Complete!" << std::endl;
    return 0;
}