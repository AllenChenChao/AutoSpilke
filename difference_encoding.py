#!/usr/bin/env python3
"""
差分编码 - 简单有效
只编码变化，忽略静态
"""

import numpy as np
import matplotlib.pyplot as plt

def diff_encode(x, threshold=0.01):
    """
    差分编码：只在变化时发脉冲
    x: 输入信号
    threshold: 变化阈值
    返回: 脉冲序列 {1, -1, 0}
    """
    diff = np.diff(x, prepend=x[0])
    spikes = np.zeros_like(diff)
    spikes[diff > threshold] = 1
    spikes[diff < -threshold] = -1
    return spikes

def diff_decode(spikes):
    """
    差分解码：累加重建信号
    """
    return np.cumsum(spikes)

def demo():
    """演示差分编码"""
    # 测试信号：阶跃 + 平台 + 斜坡
    t = np.linspace(0, 10, 100)
    x = np.concatenate([
        np.zeros(20),        # 静态段
        np.ones(20) * 0.5,   # 阶跃
        np.ones(20) * 0.5,   # 平台（无变化）
        np.linspace(0.5, 1, 20),  # 斜坡
        np.ones(20)          # 静态段
    ])

    x = np.sin(t)  # 简单正弦波
    
    # 编码
    spikes = diff_encode(x, threshold=0.02)
    
    # 解码
    reconstructed = diff_decode(spikes)
    
    # 统计
    total_spikes = np.sum(np.abs(spikes))
    efficiency = total_spikes / len(x) * 100
    
    # 可视化
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(x, 'b-', linewidth=2, label='Original')
    plt.ylabel('Signal')
    plt.title('Differential Encoding Demo')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    colors = ['red' if s > 0 else 'blue' if s < 0 else 'gray' for s in spikes]
    plt.bar(range(len(spikes)), spikes, color=colors, alpha=0.7)
    plt.ylabel('Spikes')
    plt.title(f'Spike Activity (Efficiency: {efficiency:.1f}%)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.plot(x, 'b-', linewidth=2, label='Original')
    plt.plot(reconstructed, 'r--', linewidth=2, label='Reconstructed')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 打印结果
    print(f"原始信号长度: {len(x)}")
    print(f"总脉冲数: {total_spikes}")
    print(f"编码效率: {efficiency:.1f}% (越低越好)")
    print(f"重建误差: {np.mean((x - reconstructed)**2):.6f}")

if __name__ == "__main__":
    demo()