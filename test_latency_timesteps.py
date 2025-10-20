#!/usr/bin/env python3
"""
Simple Latency timestep test
"""

import torch
import matplotlib.pyplot as plt
import numpy as np



def latency_encode(x, time_steps=100):
    """
    简化的延迟编码：信号强度决定发放时间
    x: 输入信号张量 [0,1]，值越大延迟越短
    time_steps: 时间窗口长度
    返回: 脉冲时间序列张量 [batch_size, time_steps] 或 [time_steps]
    """
    # 计算延迟时间：信号越强，延迟越短
    # delay = (1 - x) * (time_steps - 1)，范围[0, time_steps-1]
    delays = ((1.0 - x.clamp(0, 1)) * (time_steps - 1)).long()
    
    # 创建输出张量
    if x.dim() == 0:  # 标量
        spikes = torch.zeros(time_steps, dtype=x.dtype, device=x.device)
        spikes[delays] = 1.0
    elif x.dim() == 1:  # 1D张量
        spikes = torch.zeros(len(x), time_steps, dtype=x.dtype, device=x.device)
        for i, delay in enumerate(delays):
            spikes[i, delay] = 1.0
    else:  # 多维张量
        batch_shape = x.shape
        spikes = torch.zeros(*batch_shape, time_steps, dtype=x.dtype, device=x.device)
        flat_x = x.flatten()
        flat_delays = delays.flatten()
        flat_spikes = spikes.view(-1, time_steps)
        
        for i, delay in enumerate(flat_delays):
            flat_spikes[i, delay] = 1.0
            
    return spikes

def latency_decode(spikes):
    """
    简化的延迟解码：从发放时间恢复信号强度
    spikes: 脉冲时间序列 [..., time_steps]
    返回: 解码的信号值 [0,1]
    """
    # 找到第一个脉冲的位置
    spike_times = torch.argmax(spikes.float(), dim=-1)
    time_steps = spikes.shape[-1]
    
    # 转换回信号值：delay = (1-x)*(time_steps-1) => x = 1 - delay/(time_steps-1)
    decoded = 1.0 - spike_times.float() / (time_steps - 1)
    
    # 处理没有脉冲的情况（全零）
    no_spike_mask = spikes.sum(dim=-1) == 0
    decoded[no_spike_mask] = 0.0
    
    return decoded.clamp(0, 1)


# ...existing code...
def test_timesteps(features, timesteps):
    """Evaluate latency encoding at a given number of timesteps."""
    if timesteps < 2:
        raise ValueError("timesteps must be >= 2 for latency encoding.")
    torch.manual_seed(42)

    # 直接对原始 features 做一次编码（不要复制时间维）
    spikes = latency_encode(features, timesteps)      # shapes: 1D-> [F,T], 2D-> [B,F,T]
    reconstructed = latency_decode(spikes)            # shapes: [F] or [B,F]

    # 统一拉平成一维计算统计
    f_flat = features.reshape(-1)
    r_flat = reconstructed.reshape(-1)

    mse = torch.mean((f_flat - r_flat) ** 2).item()

    signal_power = torch.mean(f_flat ** 2).item()
    noise_power = mse
    snr = 10 * torch.log10(torch.tensor(signal_power / noise_power)).item()

    # 相关系数
    f_mean = f_flat.mean()
    r_mean = r_flat.mean()
    covariance = torch.mean((f_flat - f_mean) * (r_flat - r_mean)).item()
    std_f = torch.std(f_flat).item()
    std_r = torch.std(r_flat).item()
    correlation = covariance / (std_f * std_r + 1e-12)

    return mse, snr, correlation, reconstructed
# ...existing code...

def main():
    num_features = 1000
    # features = 0.2 + torch.rand(num_features) * 0.2
    
    features = torch.rand(num_features)

    
    assert torch.all(features < 1), "存在 >=1"
    assert torch.all(features > 0), "存在 <=0"

    # Latency 编码需要从 2 开始
    timestep_range = range(2, 51)

    mse_values, snr_values, corr_values = [], [], []
    for ts in timestep_range:
        mse, snr, corr, _ = test_timesteps(features, ts)
        mse_values.append(mse)
        snr_values.append(snr)
        corr_values.append(corr)

    best_idx = min(range(len(mse_values)), key=lambda i: mse_values[i])
    best_ts = list(timestep_range)[best_idx]
    print(f"Best (min MSE) timestep: {best_ts}  MSE={mse_values[best_idx]:.6f}")

    plt.rcParams.update({
        'font.family': 'Serif',  # 英文顶刊强制字体
        'axes.titlesize': 10,             # 图标题：10pt加粗
        'axes.labelsize': 9,              # 轴标签：9pt（如“Time Steps”“MSE”）
        'xtick.labelsize': 8,             # X轴刻度：8pt
        'ytick.labelsize': 8,             # Y轴刻度：8pt
        'legend.fontsize': 9,             # 图例：9pt
        'axes.titleweight': 'bold'        # 图标题加粗
    })

    # fig, ax_mse = plt.subplots(figsize=(9, 5))
    fig, ax_mse = plt.subplots(figsize=(4.0, 2.5))

    ax_snr = ax_mse.twinx()
    ax_corr = ax_mse.twinx()
    # ax_corr.spines["right"].set_position(("axes", 1.12))
    # 修改后（间距调整为1.2，避免文字拥挤）：
    ax_corr.spines["right"].set_position(("axes", 1.2))
    # 补充Y轴范围验证（确保Correlation在[0,1]，与代码一致）
    ax_corr.set_ylim(0, 1.0)

    ax_corr.spines["right"].set_visible(True)

    # l1 = ax_mse.plot(timestep_range, mse_values, 'o-', color='#1f77b4', label='MSE')[0]
    # l2 = ax_snr.plot(timestep_range, snr_values, 's--', color='#d62728', label='SNR (dB)')[0]
    # l3 = ax_corr.plot(timestep_range, corr_values, '^-', color='#2ca02c', label='Correlation')[0]
    # 修改后（线宽0.8pt，符号大小4pt，符合顶刊“清晰不拥挤”要求）：
    l1 = ax_mse.plot(timestep_range, mse_values, 'o-', color='#1f77b4', label='MSE', 
                    linewidth=0.8, markersize=4)[0]
    l2 = ax_snr.plot(timestep_range, snr_values, 's--', color='#d62728', label='SNR (dB)', 
                    linewidth=0.8, markersize=4)[0]
    l3 = ax_corr.plot(timestep_range, corr_values, '^-', color='#2ca02c', label='Correlation', 
                    linewidth=0.8, markersize=4)[0]
    # 轴标签与范围
    ax_mse.set_xlabel('Time Steps')
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

    # ax_mse.grid(True, alpha=0.35)
    # 网格线宽0.5pt，透明度0.3（避免遮挡曲线）
    ax_mse.grid(True, alpha=0.3, linewidth=0.5)

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

    # ax_mse.legend([l1, l2, l3], [l.get_label() for l in [l1, l2, l3]], loc='upper right')
    # ax_mse.set_title('Latency Encoding Metrics vs Time Steps')
    fig.tight_layout()
    # plt.show()
    # 保存图片，符合发表要求（高分辨率、白色背景、去除多余边距）
    fig.savefig("Latency_encoding_metrics_vs_timesteps.png", dpi=300, bbox_inches='tight', facecolor='white')
    print("Figure saved as latency_encoding_metrics_vs_timesteps.png")

# # ...existing code...


# def main():
#     num_features = 1000
#     features = (0.8 + torch.rand(num_features) * 0.2)


#     # 生成[0, 1]之间的随机分布特征
#     # 这里features已经是[0.8, 1.0)的均匀分布，如果你想要[0, 1)的均匀分布：
#     features = torch.rand(num_features)

#     # import json

#     # # 定义信号列表和文件路径
#     # signals = ['ecg', 'eeg', 'ppg']
#     # data_path = "/mnt/i/ephy_proc/datasets/"
#     # signals = ['ecg']
#     # # 读取每个信号的JSON数据
#     # for signal in signals:
#     #     with open(f"{data_path}{signal}.json", 'r') as f:
#     #         # 加载数据（直接得到包含2000个点的列表）
#     #         signal_data = json.load(f)
#     #         print(f"读取 {signal} 数据：共 {len(signal_data)} 个点，前5个点：{signal_data[:5]}")


    

#     # data_tensor = torch.tensor(signal_data, dtype=torch.float32)
#     # min_val, max_val = data_tensor.min(), data_tensor.max()
#     # features = (data_tensor - min_val) / (max_val - min_val)   # 归一化到


#     assert torch.all(features <= 1), f"值中存在 >1: {features[features > 1]}"
#     assert torch.all(features >= 0), f"值中存在 <0: {features[features < 0]}"
#     timestep_range = range(1, 51)
#     mse_values = []
#     snr_values = []
#     correlation_values = []
#     for ts in timestep_range:
#         mse, snr, correlation, _ = test_timesteps(features, ts)
#         mse_values.append(mse)
#         snr_values.append(snr)
#         correlation_values.append(correlation)
#     best_idx_mse = min(range(len(mse_values)), key=lambda i: mse_values[i])
#     best_ts_mse = list(timestep_range)[best_idx_mse]
#     print(f"Best timesteps for MSE: {best_ts_mse} (MSE: {mse_values[best_idx_mse]:.4f})")



#     plt.rcParams.update({
#         'font.family': 'Serif',  # 英文顶刊强制字体
#         'axes.titlesize': 10,             # 图标题：10pt加粗
#         'axes.labelsize': 9,              # 轴标签：9pt（如“Time Steps”“MSE”）
#         'xtick.labelsize': 8,             # X轴刻度：8pt
#         'ytick.labelsize': 8,             # Y轴刻度：8pt
#         'legend.fontsize': 9,             # 图例：9pt
#         'axes.titleweight': 'bold'        # 图标题加粗
#     })

#     # fig, ax_mse = plt.subplots(figsize=(9, 5))
#     # fig, ax_mse = plt.subplots(figsize=(3.5, 2.2))
#     fig, ax_mse = plt.subplots(figsize=(4.0, 2.5))

#     ax_snr = ax_mse.twinx()                 # 第二个y轴 (右侧)
#     ax_corr = ax_mse.twinx()                # 第三个y轴
#     # ax_corr.spines["right"].set_position(("axes", 1.12))  # 往右偏移
#     # 原代码：ax_corr.spines["right"].set_position(("axes", 1.12))
#     # 修改后（间距调整为1.2，避免文字拥挤）：
#     ax_corr.spines["right"].set_position(("axes", 1.2))
#     # 补充Y轴范围验证（确保Correlation在[0,1]，与代码一致）
#     ax_corr.set_ylim(0, 1.0)

#     ax_corr.spines["right"].set_visible(True)



#     # 曲线
#     # l1 = ax_mse.plot(timestep_range, mse_values, 'o-', color='#1f77b4', label='MSE')[0]
#     # l2 = ax_snr.plot(timestep_range, snr_values, 's--', color='#d62728', label='SNR (dB)')[0]
#     # l3 = ax_corr.plot(timestep_range, correlation_values, '^-', color='#2ca02c', label='Correlation')[0]
#     # 原代码：l1 = ax_mse.plot(timestep_range, mse_values, 'o-', color='#1f77b4', label='MSE')[0]
#     # 修改后（线宽0.8pt，符号大小4pt，符合顶刊“清晰不拥挤”要求）：
#     l1 = ax_mse.plot(timestep_range, mse_values, 'o-', color='#1f77b4', label='MSE', 
#                     linewidth=0.8, markersize=4)[0]
#     l2 = ax_snr.plot(timestep_range, snr_values, 's--', color='#d62728', label='SNR (dB)', 
#                     linewidth=0.8, markersize=4)[0]
#     l3 = ax_corr.plot(timestep_range, correlation_values, '^-', color='#2ca02c', label='Correlation', 
#                     linewidth=0.8, markersize=4)[0]
#     # 轴标签与范围
#     ax_mse.set_xlabel('Time Steps')
#     ax_mse.set_ylabel('MSE', color=l1.get_color())
#     ax_snr.set_ylabel('SNR (dB)', color=l2.get_color())
#     ax_corr.set_ylabel('Correlation', color=l3.get_color())
#     ax_corr.set_ylim(0, 1.0)

#     for ax, col in [(ax_mse, l1.get_color()), (ax_snr, l2.get_color()), (ax_corr, l3.get_color())]:
#         ax.tick_params(axis='y', labelcolor=col)

#     # 设置坐标轴线宽1pt（顶刊推荐1-1.5pt）
#     for ax in [ax_mse, ax_snr, ax_corr]:
#         for spine in ax.spines.values():
#             spine.set_linewidth(1.0)
#     # 网格线宽0.5pt，透明度0.3（避免遮挡曲线）
#     ax_mse.grid(True, alpha=0.3, linewidth=0.5)
#     # # 网格只开主轴
#     # ax_mse.grid(True, alpha=0.35)

#     # # 合并图例
#     # lines = [l1, l2, l3]
#     # labels = [ln.get_label() for ln in lines]
#     # ax_mse.legend(lines, labels, loc='upper center')


#     # 4. 关键：设置图例在顶部居中
#     ax.legend(
#         [l1, l2, l3],  # 图例关联的线条
#         ['MSE', 'SNR (dB)', 'Correlation'],  # 图例标签
#         loc='upper center',  # 参考锚点为顶部居中
#         bbox_to_anchor=(0.5, 1.25),  # 归一化坐标：水平居中(0.5)，垂直靠近顶部(0.98)
#         ncol=3,  # 图例分3列显示（使横向更紧凑）
#         fontsize='small',
#         frameon=False  # 去掉图例边框，更简洁（可选）
#     )

#     # ax_mse.set_title('Poisson Encoding Metrics vs Time Steps (Single Figure)')
#     fig.tight_layout()
#     # plt.show()
#     # 保存图片，符合发表要求（高分辨率、白色背景、去除多余边距）
#     fig.savefig("poisson_encoding_metrics_vs_timesteps.png", dpi=300, bbox_inches='tight', facecolor='white')
#     print("Figure saved as poisson_encoding_metrics_vs_timesteps.png")



if __name__ == "__main__":
    main()