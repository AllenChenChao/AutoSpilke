#!/usr/bin/env python3
"""
泊松编码时间步长优化测试
测试不同时间步长下的编码-解码效果
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

def simple_poisson_encode(x):
    """
    简化的泊松编码：使用均匀随机数
    x: 输入概率张量 [0,1]
    返回: 二进制脉冲张量 {0,1}
    """
    return torch.rand_like(x).le(x).to(x)

def simple_poisson_decode(spike_sequence, method='mean'):
    """
    简化的泊松解码：从脉冲序列恢复特征值
    
    Args:
        spike_sequence: 脉冲序列 (n_features, time_steps) 或 (time_steps,)
        method: 解码方法 ('mean', 'median', 'mode')
    
    Returns:
        decoded_features: 解码后的特征值 [0,1]
    """
    if spike_sequence.dim() == 1:
        # 单个特征的情况
        if method == 'mean':
            return spike_sequence.float().mean()
        elif method == 'median':
            return spike_sequence.float().median().values
        elif method == 'mode':
            return spike_sequence.float().mode().values
    else:
        # 多个特征的情况
        if method == 'mean':
            return spike_sequence.float().mean(dim=1)  # 沿时间维度平均
        elif method == 'median':
            return spike_sequence.float().median(dim=1).values
        elif method == 'mode':
            return spike_sequence.float().mode(dim=1).values

def test_single_timestep(features, time_steps, seed=42, method='mean'):
    """
    测试单个时间步长的编码-解码效果
    
    Args:
        features: 原始特征值 tensor [0,1]
        time_steps: 时间步长
        seed: 随机种子
        method: 解码方法
    
    Returns:
        dict: 包含原始值、重构值、误差等信息
    """
    torch.manual_seed(seed)
    
    # 扩展到时间维度 (n_features, time_steps)
    if features.dim() == 1:
        expanded_features = features.unsqueeze(1).repeat(1, time_steps)
    else:
        expanded_features = features.repeat(1, time_steps)
    
    # 编码
    spikes = simple_poisson_encode(expanded_features)
    
    # 解码
    reconstructed = simple_poisson_decode(spikes, method=method)
    
    # 计算误差
    mse = torch.mean((features - reconstructed) ** 2).item()
    mae = torch.mean(torch.abs(features - reconstructed)).item()
    
    return {
        'original': features,
        'reconstructed': reconstructed,
        'spikes': spikes,
        'mse': mse,
        'mae': mae,
        'time_steps': time_steps,
        'total_spikes': spikes.sum().item()
    }

def find_optimal_timesteps(features, timestep_range, seed=42, method='mean'):
    """
    寻找最优时间步长
    
    Args:
        features: 测试特征值
        timestep_range: 时间步长范围 (list)
        seed: 随机种子
        method: 解码方法
    
    Returns:
        dict: 各时间步长的测试结果
    """
    print("🔍 寻找最优时间步长...")
    print("=" * 60)
    print(f"{'时间步长':>8} {'MSE':>10} {'MAE':>10} {'总脉冲':>8} {'效率':>8}")
    print("-" * 60)
    
    results = {}
    
    for ts in timestep_range:
        result = test_single_timestep(features, ts, seed=seed, method=method)
        efficiency = result['total_spikes'] / (len(features) * ts)  # 脉冲效率
        
        results[ts] = result
        results[ts]['efficiency'] = efficiency
        
        print(f"{ts:>8} {result['mse']:>10.6f} {result['mae']:>10.6f} "
              f"{int(result['total_spikes']):>8} {efficiency:>8.3f}")
    
    # 找出最佳时间步长
    best_ts_mse = min(results.keys(), key=lambda k: results[k]['mse'])
    best_ts_mae = min(results.keys(), key=lambda k: results[k]['mae'])
    
    print("-" * 60)
    print(f"🏆 最佳时间步长 (MSE): {best_ts_mse} (MSE: {results[best_ts_mse]['mse']:.6f})")
    print(f"🏆 最佳时间步长 (MAE): {best_ts_mae} (MAE: {results[best_ts_mae]['mae']:.6f})")
    
    return results

# def visualize_encoding_quality(features, timestep_range, seed=42):
#     """
#     可视化编码质量随时间步长的变化
#     """
#     results = find_optimal_timesteps(features, timestep_range, seed=seed)
    
#     # 提取数据
#     timesteps = list(results.keys())
#     mse_values = [results[ts]['mse'] for ts in timesteps]
#     mae_values = [results[ts]['mae'] for ts in timesteps]
#     efficiency_values = [results[ts]['efficiency'] for ts in timesteps]
    
#     # 绘图
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
#     # MSE vs 时间步长
#     ax1.plot(timesteps, mse_values, 'b-o', linewidth=2, markersize=6)
#     ax1.set_xlabel('时间步长')
#     ax1.set_ylabel('MSE')
#     ax1.set_title('均方误差 vs 时间步长')
#     ax1.grid(True, alpha=0.3)
    
#     # MAE vs 时间步长
#     ax2.plot(timesteps, mae_values, 'r-s', linewidth=2, markersize=6)
#     ax2.set_xlabel('时间步长')
#     ax2.set_ylabel('MAE')
#     ax2.set_title('平均绝对误差 vs 时间步长')
#     ax2.grid(True, alpha=0.3)
    
#     # 脉冲效率 vs 时间步长
#     ax3.plot(timesteps, efficiency_values, 'g-^', linewidth=2, markersize=6)
#     ax3.set_xlabel('时间步长')
#     ax3.set_ylabel('脉冲效率')
#     ax3.set_title('脉冲效率 vs 时间步长')
#     ax3.grid(True, alpha=0.3)
    
#     # 原始 vs 重构对比 (选择中等时间步长)
#     mid_ts = timesteps[len(timesteps)//2]
#     result = results[mid_ts]
#     ax4.scatter(result['original'].numpy(), result['reconstructed'].numpy(), 
#                alpha=0.7, s=50)
#     ax4.plot([0, 1], [0, 1], 'r--', linewidth=2, label='理想线')
#     ax4.set_xlabel('原始特征值')
#     ax4.set_ylabel('重构特征值')
#     ax4.set_title(f'原始 vs 重构 (时间步长={mid_ts})')
#     ax4.legend()
#     ax4.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.show()
    
#     return results

def visualize_encoding_quality(features, timestep_range, seed=42):
    """
    可视化编码质量随时间步长的变化
    """
    results = find_optimal_timesteps(features, timestep_range, seed=seed)
    
    # 提取数据
    timesteps = list(results.keys())
    mse_values = [results[ts]['mse'] for ts in timesteps]
    mae_values = [results[ts]['mae'] for ts in timesteps]
    efficiency_values = [results[ts]['efficiency'] for ts in timesteps]
    
    # 绘图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # MSE vs 时间步长
    ax1.plot(timesteps, mse_values, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('MSE')
    ax1.set_title('Mean Squared Error vs Time Steps')
    ax1.grid(True, alpha=0.3)
    
    # MAE vs 时间步长
    ax2.plot(timesteps, mae_values, 'r-s', linewidth=2, markersize=6)
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('MAE')
    ax2.set_title('Mean Absolute Error vs Time Steps')
    ax2.grid(True, alpha=0.3)
    
    # 脉冲效率 vs 时间步长
    ax3.plot(timesteps, efficiency_values, 'g-^', linewidth=2, markersize=6)
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Spike Efficiency')
    ax3.set_title('Spike Efficiency vs Time Steps')
    ax3.grid(True, alpha=0.3)
    
    # 原始 vs 重构对比 (选择中等时间步长)
    mid_ts = timesteps[len(timesteps)//2]
    result = results[mid_ts]
    ax4.scatter(result['original'].numpy(), result['reconstructed'].numpy(), 
               alpha=0.7, s=50)
    ax4.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Ideal Line')
    ax4.set_xlabel('Original Features')
    ax4.set_ylabel('Reconstructed Features')
    ax4.set_title(f'Original vs Reconstructed (Time Steps={mid_ts})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results

def demo_timestep_optimization():
    """
    演示时间步长优化过程
    """
    print("🎯 泊松编码时间步长优化演示")
    print("=" * 60)
    
    # 创建测试特征
    torch.manual_seed(42)
    # features = torch.tensor([0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9])
    features = torch.tensor([0.05, 0.5, 0.6, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.95])
    print(f"测试特征: {features.tolist()}")
    
    # 测试不同时间步长
    # timestep_range = [5, 10, 20, 50, 100, 200, 500, 1000]
    timestep_range = [x for x in range(1,31)]
    
    # 寻找最优时间步长
    results = find_optimal_timesteps(features, timestep_range, seed=42)
    
    # 详细分析几个关键时间步长
    print("\n" + "=" * 60)
    print("详细分析")
    print("=" * 60)
    
    key_timesteps = [10, 50, 200]
    for ts in key_timesteps:
        if ts in results:
            result = results[ts]
            print(f"\n时间步长 {ts}:")
            print(f"  原始特征: {result['original'].tolist()}")
            print(f"  重构特征: {result['reconstructed'].tolist()}")
            print(f"  MSE: {result['mse']:.6f}")
            print(f"  MAE: {result['mae']:.6f}")
            print(f"  脉冲效率: {result['efficiency']:.3f}")
    
    # 可视化
    print(f"\n📊 生成可视化图表...")
    visualize_encoding_quality(features, timestep_range, seed=42)
    
    return results

if __name__ == "__main__":
    demo_timestep_optimization()