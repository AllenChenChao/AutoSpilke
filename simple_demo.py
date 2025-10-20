#!/usr/bin/env python3
"""
Simple Spike Encoding Demo
超简化神经脉冲编码演示

帮你理解核心概念的最简版本
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import spike

def simple_encode_demo():
    """
    超简单编码演示：信号强度 → 脉冲概率
    """
    print("=" * 50)
    print("超简化神经编码演示")
    print("=" * 50)
    
    # 创建简单信号
    signal = np.array([0.2, 0.8, 1.0, 0.5, 0.1])
    print(f"\n原始信号: {signal}")
    
    # 编码
    print(f"\n🧠 编码过程:")
    spikes = spike.encode_poisson(signal, neurons=10, time_window=5.0, max_rate=100.0)
    
    print(f"脉冲矩阵形状: {spikes.shape}")
    print(f"总脉冲数: {np.sum(spikes)}")
    print("脉冲矩阵:")
    print(spikes.astype(int))
    
    # 解码  
    print(f"\n🔄 解码过程:")
    reconstructed = spike.decode_poisson(spikes, max_rate=100.0, dt=1.0)
    
    # 评估
    print(f"\n📊 结果比较:")
    print(f"原始信号:   {signal}")
    print(f"重构信号:   {reconstructed}")
    
    # 计算误差
    error = np.mean((signal - reconstructed) ** 2)
    print(f"均方误差:   {error:.4f}")
    
    if error < 0.1:
        print("✅ 重构质量: 优秀")
    elif error < 0.3:
        print("✅ 重构质量: 良好")
    else:
        print("⚠️ 重构质量: 需要改进")


def rate_poisson_demo():
    """
    速率泊松编码演示：特征值直接控制发放率
    """
    print("\n" + "=" * 50)
    print("速率泊松编码演示")
    print("=" * 50)
    
    # 创建多特征数据
    data = np.array([[0.2, 0.8, 1.0, 0.5, 0.1],
                     [0.9, 0.3, 0.6, 0.7, 0.4]])
    print(f"\n原始特征数据:")
    print(f"样本1: {data[0]}")
    print(f"样本2: {data[1]}")
    
    # 编码
    print(f"\n🧠 速率泊松编码:")
    spikes = spike.encode_poisson_rate(data, time_steps=50, max_rate=80.0, seed=42)
    
    print(f"脉冲张量形状: {spikes.shape}")
    print(f"总脉冲数: {np.sum(spikes)}")
    
    # 显示第一个样本的脉冲模式
    print(f"\n样本1的脉冲模式 (前3个特征, 前10个时间步):")
    for feat in range(3):
        spikes_str = ''.join(['█' if s > 0 else '·' for s in spikes[0, feat, :10]])
        print(f"特征{feat}: {spikes_str}")
    
    # 解码  
    print(f"\n🔄 解码过程:")
    reconstructed = spike.decode_poisson_rate(spikes, max_rate=80.0)
    
    # 评估
    print(f"\n📊 结果比较:")
    print(f"原始样本1: {data[0]}")
    print(f"重构样本1: {reconstructed[0]}")
    print(f"原始样本2: {data[1]}")
    print(f"重构样本2: {reconstructed[1]}")
    
    # 计算误差
    mse = np.mean((data - reconstructed) ** 2)
    print(f"整体均方误差: {mse:.4f}")
    
    if mse < 0.1:
        print("✅ 重构质量: 优秀")
    elif mse < 0.3:
        print("✅ 重构质量: 良好")
    else:
        print("⚠️ 重构质量: 需要调参")

# def compare_encoding_methods():
#     """
#     对比两种泊松编码方法
#     """
#     print("\n" + "=" * 50)
#     print("泊松编码方法对比")
#     print("=" * 50)
    
#     # 测试信号
#     signal = np.array([0.2, 0.8, 0.5, 0.9, 0.3])
#     print(f"\n测试信号: {signal}")
    
#     # 方法1: 时间窗口泊松编码
#     print(f"\n📊 时间窗口泊松编码:")
#     result1 = spike.test_encoder('poisson', signal, 
#                                 neurons=10, time_window=20.0, max_rate=100.0)
#     print(f"  输出形状: {result1['spikes'].shape}")
#     print(f"  重构信号: {result1['reconstructed']}")
#     print(f"  MSE: {result1['metrics']['mse']:.4f}")
    
#     # 方法2: 速率泊松编码
#     print(f"\n📊 速率泊松编码:")
#     data_2d = signal.reshape(1, -1)  # 转换为2D格式
#     result2 = spike.test_encoder('poisson_rate', data_2d, 
#                                 time_steps=50, max_rate=100.0, seed=42)
#     print(f"  输出形状: {result2['spikes'].shape}")
#     print(f"  重构信号: {result2['reconstructed'].flatten()}")
#     print(f"  MSE: {result2['metrics']['mse']:.4f}")
    
#     print(f"\n🎯 方法对比:")
#     print(f"  时间窗口编码 - 适合: 信号重构、高精度需求")
#     print(f"  速率编码     - 适合: 神经网络、实时处理")

def compare_encoding_methods():
    """
    对比两种泊松编码方法
    """
    print("\n" + "=" * 50)
    print("泊松编码方法对比")
    print("=" * 50)
    
    # 测试信号
    signal = np.array([0.2, 0.8, 0.5, 0.9, 0.3])
    print(f"\n测试信号: {signal}")
    
    # 方法1: 时间窗口泊松编码
    print(f"\n📊 时间窗口泊松编码:")
    result1 = spike.test_encoder('poisson', signal, 
                                neurons=10, time_window=20.0, max_rate=100.0)
    print(f"  输出形状: {result1['spikes'].shape}")
    print(f"  重构信号: {result1['reconstructed']}")
    print(f"  MSE: {result1['metrics']['mse']:.4f}")
    
    # 方法2: 速率泊松编码
    print(f"\n📊 速率泊松编码:")
    data_2d = signal.reshape(1, -1)  # 转换为2D格式
    result2 = spike.test_encoder('poisson_rate', data_2d, 
                                time_steps=50, max_rate=100.0, seed=42)
    print(f"  输出形状: {result2['spikes'].shape}")
    print(f"  重构信号: {result2['reconstructed'].flatten()}")
    print(f"  MSE: {result2['metrics']['mse']:.4f}")
    
    print(f"\n🎯 方法对比:")
    print(f"  时间窗口编码 - 适合: 信号重构、高精度需求")
    print(f"  速率编码     - 适合: 神经网络、实时处理")

def test_different_signals():
    """
    测试不同信号类型
    """
    print("\n" + "=" * 50)
    print("测试不同信号类型")
    print("=" * 50)
    
    signals = {
        '正弦波': spike.signal_sine(20, freq=1.0),
        '方波': spike.signal_square(20, freq=1.0),
        '噪声': spike.signal_noise(20, amplitude=1.0),
        '脉冲': spike.signal_pulse(20, width=5)
    }
    
    for name, signal in signals.items():
        print(f"\n{name}:")
        try:
            result = spike.test_encoder('poisson', signal, neurons=20, time_window=20.0)
            metrics = result['metrics']
            
            print(f"  MSE: {metrics['mse']:.6f}")
            print(f"  相关系数: {metrics['correlation']:.4f}")
            print(f"  脉冲数: {metrics['total_spikes']}")
            
            # 质量评估
            if metrics['mse'] < 0.1:
                quality = "优秀 ✅"
            elif metrics['mse'] < 0.3:
                quality = "良好 ✅"
            elif metrics['mse'] < 0.5:
                quality = "可接受 ⚠️"
            else:
                quality = "差 ❌"
            
            print(f"  质量: {quality}")
            
        except Exception as e:
            print(f"  错误: {e}")

def test_parameter_effects():
    """
    测试参数影响
    """
    print("\n" + "=" * 50)
    print("参数影响测试")
    print("=" * 50)
    
    signal = spike.signal_sine(20, freq=1.0)
    
    print("\n🧠 神经元数量的影响:")
    neuron_counts = [5, 10, 20, 30]
    
    for neurons in neuron_counts:
        try:
            result = spike.test_encoder('poisson', signal, neurons=neurons, time_window=20.0)
            mse = result['metrics']['mse']
            spikes = result['metrics']['total_spikes']
            print(f"  {neurons:2d}个神经元: MSE={mse:.6f}, 脉冲={spikes}")
        except Exception as e:
            print(f"  {neurons:2d}个神经元: 错误 - {e}")
    
    print("\n⚡ 发放率的影响:")
    rates = [50, 100, 150, 200]
    
    for rate in rates:
        try:
            result = spike.test_encoder('poisson', signal, neurons=20, max_rate=rate, time_window=20.0)
            mse = result['metrics']['mse']
            spikes = result['metrics']['total_spikes']
            print(f"  {rate:3d}Hz: MSE={mse:.6f}, 脉冲={spikes}")
        except Exception as e:
            print(f"  {rate:3d}Hz: 错误 - {e}")

def compare_encoders():
    """
    比较不同编码器
    """
    print("\n" + "=" * 50)
    print("编码器对比")
    print("=" * 50)
    
    signal = spike.signal_sine(20, freq=2.0)
    encoders = ['poisson', 'latency', 'rate']
    
    results = {}
    for encoder in encoders:
        try:
            result = spike.test_encoder(encoder, signal, neurons=20, time_window=20.0)
            results[encoder] = result['metrics']
            
            print(f"\n{encoder.capitalize()}编码:")
            print(f"  MSE: {result['metrics']['mse']:.6f}")
            print(f"  相关系数: {result['metrics']['correlation']:.4f}")
            print(f"  脉冲数: {result['metrics']['total_spikes']}")
            
        except Exception as e:
            print(f"\n{encoder.capitalize()}编码: 失败 - {e}")
    
    # 找最佳编码器
    if results:
        best = min(results.keys(), key=lambda x: results[x]['mse'])
        print(f"\n🏆 最佳编码器: {best} (MSE: {results[best]['mse']:.6f})")

def understand_concepts():
    """
    理解核心概念
    """
    print("\n" + "=" * 50)
    print("核心概念理解")
    print("=" * 50)
    
    print("\n🎯 神经脉冲编码的核心思想:")
    print("- 强信号 → 多脉冲")
    print("- 弱信号 → 少脉冲")
    print("- 多神经元 → 降低噪声")
    print("- 统计平均 → 恢复信号")
    
    print("\n📊 质量指标:")
    print("- MSE < 0.1: 优秀")
    print("- MSE < 0.3: 良好")
    print("- MSE < 0.5: 可接受")
    print("- MSE > 0.5: 需要调参")
    
    print("\n🔧 参数选择建议:")
    print("- 神经元数量: 信号长度的0.8-1.2倍")
    print("- 发放率: 80-150Hz (生物学合理)")
    print("- 时间窗口: 等于或略大于信号长度")
    
    print("\n💡 实际应用:")
    print("- 神经形态计算: 低功耗芯片")
    print("- 脑机接口: 神经信号处理")
    print("- 人工智能: 脉冲神经网络")

def main():
    """
    主函数
    """
    print("🎯 神经脉冲编码 - 超简化演示")
    print("帮助你理解核心概念")
    
    try:
        # 基本演示
        simple_encode_demo()
        
        # 速率泊松编码演示
        rate_poisson_demo()

        # 不同信号测试
        test_different_signals()
        
        # 参数影响
        test_parameter_effects()
        
        # 编码器对比
        compare_encoders()
        
        # 概念理解
        understand_concepts()
        
        print("\n🎉 恭喜！你已经理解了神经编码的核心概念！")
        print("\n下一步:")
        print("1. 运行 demos/interactive_demo.py 看交互式演示")
        print("2. 阅读 src/spike.py 了解详细实现")
        print("3. 运行 tools/debugger.py 学习参数优化")
        
    except KeyboardInterrupt:
        print("\n\n演示被中断")
    except Exception as e:
        print(f"\n演示出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
