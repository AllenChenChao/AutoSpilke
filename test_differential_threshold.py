#!/usr/bin/env python3
"""
Cumulative Difference Encoding Evaluation
基于累积差分编码的评估，参考泊松编码评估方法
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json

def cumulative_difference_encoding(features, threshold=0.3, reset_after_spike=True):
    """
    累积差分编码：积累微小变化，超过阈值时发放脉冲
    Args:
        features: 输入特征 (time_steps,) numpy array
        threshold: 脉冲触发阈值
        reset_after_spike: 脉冲后是否重置累积值
    Returns:
        spike_train: 脉冲序列 {-1, 0, +1}
        diff_signals: 差分信号
        cumulative_diff: 最终累积值
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim == 1:
        features = features.reshape(-1, 1)  # 转换为 (time_steps, 1)
    
    time_steps, num_features = features.shape
    spike_train = np.zeros((time_steps, num_features), dtype=np.int8)
    diff_signals = np.zeros((time_steps, num_features))
    cumulative_diff = np.zeros(num_features)
    
    for t in range(1, time_steps):
        current_diff = features[t] - features[t-1]
        diff_signals[t] = current_diff
        cumulative_diff += current_diff
        
        # 向量化阈值检测
        pos_spikes = cumulative_diff >= threshold
        neg_spikes = cumulative_diff <= -threshold
        
        spike_train[t, pos_spikes] = 1
        spike_train[t, neg_spikes] = -1
        
        if reset_after_spike:
            cumulative_diff[pos_spikes | neg_spikes] = 0
    
    return spike_train, diff_signals, cumulative_diff

def cumulative_difference_decode(spike_train, initial_value, threshold=0.3, reset_after_spike=True):
    """
    仅使用脉冲序列和初始值重构累积差分编码的信号
    Args:
        spike_train: 脉冲序列 (time_steps, num_features)
        initial_value: 原始信号的初始值
        threshold: 与编码时一致的阈值
        reset_after_spike: 与编码时保持一致的重置策略
    Returns:
        reconstructed: 重构的信号序列 (time_steps,)
    """
    time_steps, num_features = spike_train.shape
    reconstructed = np.zeros((time_steps, num_features), dtype=np.float64)
    reconstructed[0] = np.asarray(initial_value, dtype=np.float64).reshape(-1)
    
    simulated_cumulative = np.zeros(num_features, dtype=np.float64)
    
    for t in range(1, time_steps):
        for f in range(num_features):
            if spike_train[t, f] == 1:
                # 正向脉冲
                simulated_cumulative[f] = threshold
                reconstructed[t, f] = reconstructed[t-1, f] + simulated_cumulative[f]
                if reset_after_spike:
                    simulated_cumulative[f] = 0
                    
            elif spike_train[t, f] == -1:
                # 负向脉冲
                simulated_cumulative[f] = -threshold
                reconstructed[t, f] = reconstructed[t-1, f] + simulated_cumulative[f]
                if reset_after_spike:
                    simulated_cumulative[f] = 0
            else:
                # 无脉冲，估算微小变化
                if simulated_cumulative[f] > 0:
                    delta = min(0.1 * threshold, threshold - simulated_cumulative[f])
                    simulated_cumulative[f] += delta
                elif simulated_cumulative[f] < 0:
                    delta = max(-0.1 * threshold, -threshold - simulated_cumulative[f])
                    simulated_cumulative[f] += delta
                else:
                    delta = 0
                
                reconstructed[t, f] = reconstructed[t-1, f] + delta
    
    return reconstructed.flatten() if num_features == 1 else reconstructed

def test_diff_thresholds(features, thresholds):
    """测试不同阈值下的差分编码性能，类似于test_timesteps"""
    # 转换为numpy array
    if torch.is_tensor(features):
        features_np = features.detach().cpu().numpy()
    else:
        features_np = np.array(features)
    
    # 编码和解码
    spikes, diff_signals, _ = cumulative_difference_encoding(
        features_np, threshold=thresholds, reset_after_spike=True
    )
    reconstructed_np = cumulative_difference_decode(
        spikes, features_np[0], threshold=thresholds, reset_after_spike=True
    )
    
    # 转换回torch进行指标计算（保持与泊松编码一致）
    features_torch = torch.tensor(features_np, dtype=torch.float32)
    reconstructed_torch = torch.tensor(reconstructed_np, dtype=torch.float32)
    
    # 计算MSE
    mse = torch.mean((features_torch - reconstructed_torch) ** 2).item()
    
    # 计算SNR
    signal_power = torch.mean(features_torch ** 2).item()
    noise_power = mse
    snr = 10 * torch.log10(torch.tensor(signal_power / noise_power)).item()
    
    # 计算相关系数
    f_mean = torch.mean(features_torch)
    r_mean = torch.mean(reconstructed_torch)
    covariance = torch.mean((features_torch - f_mean) * (reconstructed_torch - r_mean)).item()
    std_features = torch.std(features_torch).item()
    std_reconstructed = torch.std(reconstructed_torch).item()
    correlation = covariance / (std_features * std_reconstructed + 1e-12)
    
    # 计算脉冲相关指标
    total_spikes = np.sum(np.abs(spikes))  # 正负脉冲总数
    spike_rate = total_spikes / len(features_np)  # 平均每时间步的脉冲数
    sparsity = (1 - spike_rate) * 100  # 稀疏度百分比
    
    # 比特效率（假设原始32位浮点数，脉冲用2位表示{-1,0,+1}）
    original_bits = len(features_np) * 32
    spike_bits = len(features_np) * 2  # 每时间步2位存储脉冲
    bit_efficiency = (1 - spike_bits / original_bits) * 100
    
    return mse, snr, correlation, reconstructed_torch, total_spikes, sparsity, bit_efficiency

def main():
    # 读取真实数据
    signals = ['ecg', 'eeg', 'ppg']
    data_path = "/mnt/i/ephy_proc/datasets/"
    signals = ['ecg']  # 测试ECG数据
    
    for signal in signals:
        with open(f"{data_path}{signal}.json", 'r') as f:
            signal_data = json.load(f)
            print(f"读取 {signal} 数据：共 {len(signal_data)} 个点")

        # 归一化数据到[0,1]
        data_tensor = torch.tensor(signal_data, dtype=torch.float32)
        min_val, max_val = data_tensor.min(), data_tensor.max()
        features = (data_tensor - min_val) / (max_val - min_val)
        
        # 测试不同的阈值范围（类似于泊松编码的时间步）
        # threshold_range = np.arange(0.0001, 0.061, 0.0005)  # 0.01到0.20，步长0.01
        threshold_range = np.arange(0.001, 0.61, 0.005)  # 0.01到0.20，步长0.01 ## only for EEG

        ## 指数分布
        # 生成指数分布的阈值范围：从 1e-3 (0.001) 到 1e-0 (1.0)，共 100 个点
        # threshold_range = np.logspace(-8, 0, num=100)  # 底数为10，起始指数-3，结束指数0
        
        mse_values = []
        snr_values = []
        correlation_values = []
        sparsity_values = []
        bit_efficiency_values = []
        total_spikes_values = []
        
        for threshold in threshold_range:
            mse, snr, corr, _, total_spikes, sparsity, bit_eff = test_diff_thresholds(features, threshold)
            mse_values.append(mse)
            snr_values.append(snr)
            correlation_values.append(corr)
            sparsity_values.append(sparsity)
            bit_efficiency_values.append(bit_eff)
            total_spikes_values.append(total_spikes)
        
        # 找到最佳阈值
        best_idx_mse = min(range(len(mse_values)), key=lambda i: mse_values[i])
        best_threshold_mse = threshold_range[best_idx_mse]
        print(f"Best threshold for MSE: {best_threshold_mse:.3f} (MSE: {mse_values[best_idx_mse]:.6f})")
        
        # 打印最后一个阈值的指标
        print(f"Final threshold={threshold_range[-1]:.3f}: "
              f"Sparsity={sparsity_values[-1]:.2f}%  "
              f"BitEfficiency={bit_efficiency_values[-1]:.2f}%  "
              f"TotalSpikes={int(total_spikes_values[-1])}")

        plt.rcParams.update({
            'font.family': 'Serif',  # 英文顶刊强制字体
            'axes.titlesize': 10,             # 图标题：10pt加粗
            'axes.labelsize': 9,              # 轴标签：9pt（如“Time Steps”“MSE”）
            'xtick.labelsize': 8,             # X轴刻度：8pt
            'ytick.labelsize': 8,             # Y轴刻度：8pt
            'legend.fontsize': 9,             # 图例：9pt
            'axes.titleweight': 'bold'        # 图标题加粗
        })        
        # 绘图（参考泊松编码的绘图方式）
        # fig, ax_mse = plt.subplots(figsize=(9, 5))
        fig, ax_mse = plt.subplots(figsize=(4.3, 2.5))

        ax_snr = ax_mse.twinx()
        ax_corr = ax_mse.twinx()
        # ax_corr.spines["right"].set_position(("axes", 1.12))
        ax_corr.spines["right"].set_position(("axes", 1.2))
        # 补充Y轴范围验证（确保Correlation在[0,1]，与代码一致）
        ax_corr.set_ylim(0, 1.0)

        ax_corr.spines["right"].set_visible(True)

        # 曲线
        # l1 = ax_mse.plot(threshold_range, mse_values, 'o-', color='#1f77b4', label='MSE')[0]
        # l2 = ax_snr.plot(threshold_range, snr_values, 's--', color='#d62728', label='SNR (dB)')[0]
        # l3 = ax_corr.plot(threshold_range, correlation_values, '^-', color='#2ca02c', label='Correlation')[0]
        # 修改后（线宽0.8pt，符号大小4pt，符合顶刊“清晰不拥挤”要求）：
        l1 = ax_mse.plot(threshold_range, mse_values, 'o-', color='#1f77b4', label='MSE', 
                        linewidth=0.8, markersize=4)[0]
        l2 = ax_snr.plot(threshold_range, snr_values, 's--', color='#d62728', label='SNR (dB)', 
                        linewidth=0.8, markersize=4)[0]
        l3 = ax_corr.plot(threshold_range, correlation_values, '^-', color='#2ca02c', label='Correlation', 
                        linewidth=0.8, markersize=4)[0]
    
        # 轴标签与范围
        ax_mse.set_xlabel('Difference Threshold')
        ax_mse.set_ylabel('MSE', color=l1.get_color())
        ax_snr.set_ylabel('SNR (dB)', color=l2.get_color())
        ax_corr.set_ylabel('Correlation', color=l3.get_color())
        ax_corr.set_ylim(0, 1.0)

        for ax, col in [(ax_mse, l1.get_color()), (ax_snr, l2.get_color()), (ax_corr, l3.get_color())]:
            ax.tick_params(axis='y', labelcolor=col)

        # 设置坐标轴线宽1pt（顶刊推荐1-1.5pt）
        for ax in [ax_mse, ax_snr, ax_corr]:
            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
        # 网格线宽0.5pt，透明度0.3（避免遮挡曲线）
        ax_mse.grid(True, alpha=0.3, linewidth=0.5)

        # 标记最佳点
        ax_mse.axvline(best_threshold_mse, color='green', linestyle='--', alpha=0.7, label=f'Best: {best_threshold_mse:.3f}')
        
        # 网格和图例
        # ax_mse.grid(True, alpha=0.35)
        # lines = [l1, l2, l3]
        # labels = [ln.get_label() for ln in lines]
        # ax_mse.legend(lines, labels, loc='upper right')
        # 4. 关键：设置图例在顶部居中
        ax.legend(
            [l1, l2, l3],  # 图例关联的线条
            ['MSE', 'SNR (dB)', 'Correlation'],  # 图例标签
            loc='upper center',  # 参考锚点为顶部居中
            bbox_to_anchor=(0.5, 1.25),  # 归一化坐标：水平居中(0.5)，垂直靠近顶部(0.98)
            ncol=3,  # 图例分3列显示（使横向更紧凑）
            fontsize='small',
            frameon=False  # 去掉图例边框，更简洁（可选）
        )        
        # ax_mse.set_title(f'Cumulative Difference Encoding Metrics vs Threshold ({signal.upper()})')
        fig.tight_layout()
        # 保存图片，符合发表要求（高分辨率、白色背景、去除多余边距）
        fig.savefig("Differential_encoding_metrics_vs_threshold.png", dpi=300, bbox_inches='tight', facecolor='white')

        
        # 第二个图：脉冲相关指标
        fig2, ax_rate = plt.subplots(figsize=(9, 5))
        lr1 = ax_rate.plot(threshold_range, sparsity_values, 'o-', color='#4444aa',
                          label='Sparsity (%)')[0]
        lr2 = ax_rate.plot(threshold_range, bit_efficiency_values, 's--', color='#aa44aa',
                          label='Bit Efficiency (%)')[0]
        ax_rate.set_xlabel('Difference Threshold')
        ax_rate.set_ylabel('Percent (%)', color='#222222')
        ax_rate.set_title(f'Spike Metrics vs Threshold ({signal.upper()})')
        ax_rate.grid(True, alpha=0.35)

        ax_spikes = ax_rate.twinx()
        lr3 = ax_spikes.plot(threshold_range, total_spikes_values, '^-', color='#228833',
                            label='Total Spikes')[0]
        ax_spikes.set_ylabel('Total Spikes', color='#228833')
        ax_spikes.tick_params(axis='y', labelcolor='#228833')

        lines2 = [lr1, lr2, lr3]
        ax_rate.legend(lines2, [l.get_label() for l in lines2], loc='upper left')
        fig2.tight_layout()

        plt.show()

if __name__ == "__main__":
    main()