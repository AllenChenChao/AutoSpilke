#!/usr/bin/env python3
"""
测试简化的延迟编码实现
基于时间延迟的高效实现
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

def simple_latency_encode(x, time_steps=100):
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

def simple_latency_encode_with_noise(x, time_steps=100, noise_std=0.1):
    """
    带噪声的延迟编码：添加时间抖动以增加鲁棒性
    x: 输入信号张量 [0,1]
    time_steps: 时间窗口长度  
    noise_std: 时间噪声标准差（相对于time_steps）
    返回: 脉冲时间序列张量
    """
    # 基础延迟
    base_delays = (1.0 - x.clamp(0, 1)) * (time_steps - 1)
    
    # 添加高斯噪声
    noise = torch.randn_like(base_delays) * noise_std * time_steps
    noisy_delays = (base_delays + noise).clamp(0, time_steps - 1).long()
    
    # 生成脉冲
    if x.dim() == 0:
        spikes = torch.zeros(time_steps, dtype=x.dtype, device=x.device)
        spikes[noisy_delays] = 1.0
    elif x.dim() == 1:
        spikes = torch.zeros(len(x), time_steps, dtype=x.dtype, device=x.device)
        for i, delay in enumerate(noisy_delays):
            spikes[i, delay] = 1.0
    else:
        batch_shape = x.shape
        spikes = torch.zeros(*batch_shape, time_steps, dtype=x.dtype, device=x.device)
        flat_delays = noisy_delays.flatten()
        flat_spikes = spikes.view(-1, time_steps)
        
        for i, delay in enumerate(flat_delays):
            flat_spikes[i, delay] = 1.0
            
    return spikes

def simple_latency_decode(spikes):
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

def test_basic_functionality():
    """测试基本功能"""
    print("🧪 测试1: 基本功能")
    
    # 测试数据：不同的信号强度
    signals = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    time_steps = 10
    
    print(f"输入信号: {signals}")
    
    # 编码
    spikes = simple_latency_encode(signals, time_steps)
    print(f"输出形状: {spikes.shape}")
    
    # 显示每个信号的发放时间
    for i, signal in enumerate(signals):
        spike_time = torch.argmax(spikes[i])
        print(f"信号 {signal:.1f} -> 延迟 {spike_time} 步")
    
    # 解码测试
    decoded = simple_latency_decode(spikes)
    print(f"解码结果: {decoded}")
    print(f"重建误差: {torch.abs(signals - decoded)}")
    
    return True

def test_encoding_properties():
    """测试编码特性"""
    print("\n🧪 测试2: 编码特性")
    
    time_steps = 50
    
    # 强信号应该早发放
    strong_signal = torch.tensor(0.9)
    weak_signal = torch.tensor(0.1)
    
    strong_spikes = simple_latency_encode(strong_signal, time_steps)
    weak_spikes = simple_latency_encode(weak_signal, time_steps)
    
    strong_time = torch.argmax(strong_spikes)
    weak_time = torch.argmax(weak_spikes)
    
    print(f"强信号({strong_signal:.1f})发放时间: {strong_time}")
    print(f"弱信号({weak_signal:.1f})发放时间: {weak_time}")
    
    early_firing = strong_time < weak_time
    print(f"强信号比弱信号早发放: {'✅' if early_firing else '❌'}")
    
    return early_firing

def test_batch_processing():
    """测试批处理"""
    print("\n🧪 测试3: 批处理")
    
    # 模拟神经网络输入：(batch_size, features)
    batch_size, features = 3, 5
    time_steps = 20
    
    # 随机信号
    torch.manual_seed(42)
    signals = torch.rand(batch_size, features)
    
    print(f"输入形状: {signals.shape}")
    print(f"信号范围: [{signals.min():.3f}, {signals.max():.3f}]")
    
    # 编码
    spikes = simple_latency_encode(signals, time_steps)
    
    print(f"输出形状: {spikes.shape}")
    print(f"每个神经元发放一次脉冲: {(spikes.sum(dim=-1) == 1).all()}")
    
    # 解码
    decoded = simple_latency_decode(spikes)
    reconstruction_error = torch.nn.functional.mse_loss(signals, decoded)
    
    print(f"重建误差(MSE): {reconstruction_error:.6f}")
    
    return reconstruction_error < 0.01  # 小误差容忍

def test_noise_robustness():
    """测试噪声鲁棒性"""
    print("\n🧪 测试4: 噪声鲁棒性")
    
    signals = torch.tensor([0.2, 0.5, 0.8])
    time_steps = 30
    
    print(f"原始信号: {signals}")
    
    # 无噪声编码
    torch.manual_seed(42)
    clean_spikes = simple_latency_encode(signals, time_steps)
    clean_decoded = simple_latency_decode(clean_spikes)
    
    # 有噪声编码
    torch.manual_seed(42)
    noisy_spikes = simple_latency_encode_with_noise(signals, time_steps, noise_std=0.05)
    noisy_decoded = simple_latency_decode(noisy_spikes)
    
    print(f"无噪声解码: {clean_decoded}")
    print(f"有噪声解码: {noisy_decoded}")
    
    clean_error = torch.abs(signals - clean_decoded).mean()
    noisy_error = torch.abs(signals - noisy_decoded).mean()
    
    print(f"无噪声误差: {clean_error:.4f}")
    print(f"有噪声误差: {noisy_error:.4f}")
    
    return noisy_error < 0.1  # 噪声下仍有合理精度

## 之前到终端的方法
# def visualize_encoding():
#     """可视化编码结果"""
#     print("\n🧪 测试5: 可视化")
    
#     # 不同强度的信号
#     signals = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
#     time_steps = 20
    
#     print("延迟编码可视化:")
#     print("信号  延迟模式")
#     print("-" * 30)
    
#     spikes = simple_latency_encode(signals, time_steps)
    
#     for i, signal in enumerate(signals):
#         # 转换为可视化字符
#         pattern = ''.join(['█' if s > 0 else '·' for s in spikes[i]])
#         spike_time = torch.argmax(spikes[i]).item()
        
#         print(f"{signal:.1f}   {pattern} (t={spike_time})")
    
#     return True

## doubao的方法
# def visualize_encoding():
#     """可视化编码结果为图片"""
#     print("\n🧪 测试5: 可视化")
    
#     # 不同强度的信号
#     signals = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
#     time_steps = 20
    
#     # 生成延迟编码脉冲
#     spikes = simple_latency_encode(signals, time_steps)
    
#     # 创建文本内容
#     text_content = [
#         "Latency Encoding Pattern (20 time steps):",
#         ""
#     ]
    
#     # 简洁的表头
#     text_content.extend([
#         "ENCODING RESULTS:",
#         "─" * 50,
#         "Sample  Value  Spike Time    Pattern",
#         "──────  ─────  ──────────    ───────"
#     ])
    
#     # 生成数据行
#     for sample in range(len(signals)):
#         spike_pattern = ''.join(['█' if s > 0 else '·' for s in spikes[sample]])
#         spike_time = torch.argmax(spikes[sample]).item()
        
#         text_content.append(
#             f"{sample:>6}  {signals[sample]:>5.1f}  {spike_time:>10}    {spike_pattern}"
#         )

#     # 添加统计信息
#     text_content.extend(["", "Statistical Summary:", "─" * 50])
#     total_spikes = spikes.sum().item()
#     avg_spikes = total_spikes / len(signals)
#     text_content.append(f"Total spikes: {int(total_spikes)}")
#     text_content.append(f"Average spikes per sample: {avg_spikes:.1f}")
#     text_content.append(f"Spike efficiency: 1 spike per sample (fixed)")

#     # 动态计算图片尺寸
#     text_lines = len(text_content)
#     max_char = max(len(line) for line in text_content)
#     font_size = 11
    
#     # 计算适合的画布尺寸
#     fig_width = (max_char / 10) + 0.5  # 等宽字体字符宽度估算
#     fig_height = (text_lines / 15) + 0.5  # 等宽字体行高估算
    
#     # 创建画布
#     fig, ax = plt.subplots(figsize=(fig_width, fig_height))
#     ax.set_facecolor('#f8f9fa')  # 浅灰色背景
#     ax.axis('off')  # 隐藏坐标轴

#     # 在图片中显示文本
#     ax.text(0.00, 1.00, '\n'.join(text_content), transform=ax.transAxes,
#             verticalalignment='top', fontsize=font_size, 
#             bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
#                      edgecolor='#cccccc'))
    
#     # 保存图片
#     plt.tight_layout()
#     plt.savefig('latency_encoding_visualization.pdf', dpi=300, bbox_inches='tight', 
#                 facecolor=fig.get_facecolor(), edgecolor='none')
#     plt.savefig('latency_encoding_visualization.png', dpi=600, bbox_inches='tight',
#                 facecolor=fig.get_facecolor(), edgecolor='none')
#     plt.close()
    
#     print("已生成延迟编码可视化图片: latency_encoding_visualization.pdf/png")
#     return True

## deepseek的方法
# def visualize_encoding():
#     """Generate high-quality image for latency encoding visualization (English only)"""
#     # Test data for latency encoding
#     signals = torch.tensor([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
#     time_steps = 20
    
#     # Generate latency encoding spikes
#     spikes = simple_latency_encode(signals, time_steps)
    
#     # Set professional style
#     plt.rcParams['font.family'] = 'DejaVu Sans Mono'
#     plt.rcParams['font.size'] = 11
    
#     # Create text content
#     text_content = [
#         "Latency Encoding Pattern (20 time steps):",
#         "Method: spike_time = (1 - input_value) × (time_steps - 1)",
#         ""
#     ]
    
#     # Header with latency-specific information
#     text_content.extend([
#         "ENCODING RESULTS:",
#         "─" * 70,
#         "Sample  Value  Latency  Spike Time  Pattern",
#         "──────  ─────  ───────  ──────────  ───────"
#     ])
    
#     # Data rows with latency information
#     for sample in range(len(signals)):
#         spike_pattern = ''.join(['█' if s > 0 else '·' for s in spikes[sample]])
#         spike_time = torch.argmax(spikes[sample]).item()
#         latency = (1.0 - signals[sample].item())  # Normalized latency [0,1]
        
#         text_content.append(
#             f"{sample:>6}  {signals[sample]:>5.1f}  {latency:>7.2f}  {spike_time:>10d}    {spike_pattern}"
#         )

#     # Statistical analysis specific to latency encoding
#     text_content.extend(["", "STATISTICAL ANALYSIS:", "─" * 70])
#     total_spikes = spikes.sum().item()
#     avg_latency = torch.argmax(spikes.float(), dim=1).float().mean().item()
#     time_efficiency = (time_steps - avg_latency) / time_steps  # Early firing efficiency
    
#     text_content.append(f"Total spikes: {int(total_spikes)} (one spike per neuron)")
#     text_content.append(f"Average latency: {avg_latency:.1f} time steps")
#     text_content.append(f"Time efficiency: {time_efficiency:.1%} (early firing advantage)")
#     text_content.append(f"Energy consumption: {len(signals)} spikes total")
    
#     # Latency encoding characteristics
#     text_content.extend([
#         "",
#         "ENCODING CHARACTERISTICS:",
#         "─" * 70,
#         "✓ Single spike per input signal",
#         "✓ Earlier firing for stronger inputs", 
#         "✓ Fixed energy consumption regardless of input values",
#         "✓ Temporal coding preserves precise amplitude information",
#         "✓ Suitable for fast decision-making applications"
#     ])

#     # Dynamic figure size calculation
#     text_lines = len(text_content)
#     max_char = max(len(line) for line in text_content)
#     fig_width = (max_char / 10) + 1.0  # Wider for additional columns
#     fig_height = (text_lines / 14) + 0.8
    
#     # Create figure
#     fig, ax = plt.subplots(figsize=(fig_width, fig_height))
#     ax.set_facecolor('#f8f9fa')
#     ax.axis('off')

#     # Display text
#     ax.text(0.02, 0.98, '\n'.join(text_content), transform=ax.transAxes,
#             verticalalignment='top', fontsize=10,
#             bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
#                      edgecolor='#cccccc', pad=12),
#             linespacing=1.1)
    
#     # Professional title
#     ax.text(0.5, 0.98, 'Latency (Time-to-First-Spike) Encoding Analysis', 
#             transform=ax.transAxes, horizontalalignment='center',
#             fontsize=12, fontweight='bold', fontfamily='DejaVu Sans',
#             bbox=dict(boxstyle='round', facecolor='#2E5D7A', alpha=0.9, 
#                      edgecolor='#1a3a5a', linewidth=2, pad=8),
#             color='white')
    
#     # Save high-quality images
#     plt.tight_layout()
#     plt.savefig('latency_encoding_analysis.pdf', dpi=300, bbox_inches='tight', 
#                 facecolor='white', edgecolor='none')
#     plt.savefig('latency_encoding_analysis.png', dpi=600, bbox_inches='tight',
#                 facecolor='white', edgecolor='none')
#     plt.close()
    
#     print("Generated latency encoding analysis: latency_encoding_analysis.pdf/png")
    
#     # Terminal output for verification
#     print(f"\nLatency Encoding Statistics:")
#     print(f"Signals: {len(signals)}")
#     print(f"Time steps: {time_steps}")
#     print(f"Total spikes: {int(total_spikes)}")
#     print(f"Average latency: {avg_latency:.1f} steps")
#     print(f"Time efficiency: {time_efficiency:.1%}")
    
#     # Show latency progression
#     print(f"\nLatency progression:")
#     for sample in range(len(signals)):
#         spike_time = torch.argmax(spikes[sample]).item()
#         print(f"Value {signals[sample]:.1f} -> Spike at t={spike_time}")
    
#     return True

## augument的方法 不对，没学到
# def visualize_encoding():
#     """Generate high-quality image with terminal-style output (English only)"""
#     # Test data
#     # data = np.array([[0.0], [0.1],[0.3], [0.5], [0.7], [0.9], [1.0]])
#     # spikes = encode_poisson_rate_test(data, time_steps=20, seed=42)
#     signals = torch.tensor([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
#     time_steps = 20
#     spikes = simple_latency_encode(signals, time_steps)
    
#     # Set style - mimic terminal appearance
#     plt.rcParams['font.family'] = 'DejaVu Sans Mono'
#     plt.rcParams['font.size'] = 11
    
#     # Create text content in English
#     text_content = [
#         "Poisson Encoding Pattern (20 time steps):",
#         ""
#     ]
    
#     # 简洁的表头
#     text_content.extend([
#         "",
#         "ENCODING RESULTS:",
#         "─" * 50,
#         "Sample  Value  Spikes    Pattern",
#         "──────  ─────  ──────    ───────"
#     ])
    
#     # 简洁的数据行
#     for sample in range(data.shape[0]):
#         spike_pattern = ''.join(['█' if s > 0 else '·' for s in spikes[sample, 0, :]])
#         count = int(np.sum(spikes[sample, 0, :]))
        
#         text_content.append(
#             f"{sample:>6}  {data[sample,0]:>5.1f}  {count:>3d}/{20}    {spike_pattern}"
#         )

#     # Add statistics
#     text_content.extend(["", "Statistical Summary:", "─" * 50])
#     total_spikes = np.sum(spikes)
#     avg_rate = total_spikes / len(data) / 20
#     text_content.append(f"Total spikes: {int(total_spikes)}")
#     text_content.append(f"Spike sparsity: {avg_rate*100:.1f}%")
#     text_content.append(f"Bit Efficiency: {total_spikes/(len(data)*32)*100:.1f}%")

#     # 动态计算figsize
#     text_lines = len(text_content)
#     max_char = max(len(line) for line in text_content)
    
#     fig_width = (max_char / 10) + 0.5  
#     fig_height = (text_lines / 15) + 0.5  
    
#     # 创建画布
#     fig, ax = plt.subplots(figsize=(fig_width, fig_height))
#     ax.set_facecolor('#f8f9fa')
#     ax.axis('off') 

#     # Display text in the figure
#     ax.text(0.00, 1.00, '\n'.join(text_content), transform=ax.transAxes,
#             verticalalignment='top', fontsize=11, 
#             bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
#                      edgecolor='#cccccc'))
    
#     # Save high-quality images
#     plt.tight_layout()
#     plt.savefig('terminal_output_style.pdf', dpi=300, bbox_inches='tight', 
#                 facecolor=fig.get_facecolor(), edgecolor='none')
#     plt.savefig('terminal_output_style.png', dpi=600, bbox_inches='tight',
#                 facecolor=fig.get_facecolor(), edgecolor='none')
#     plt.close()
    
#     # 输出到图片而不是终端
#     print("Generated terminal-style output image: terminal_output_style.pdf/png")

## Copilot Grok的方法
def visualize_encoding():
    """Generate high-quality image with terminal-style output for latency encoding"""
    # Test data
    signals = torch.tensor([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    time_steps = 20
    
    # Generate spikes using latency encoding
    spikes = simple_latency_encode(signals, time_steps)
    
    # Set style - mimic terminal appearance
    plt.rcParams['font.family'] = 'DejaVu Sans Mono'
    plt.rcParams['font.size'] = 11
    
    # Create text content in English
    text_content = [
        "Latency Encoding Pattern (20 time steps):",
        ""
    ]
    
    # Table header
    text_content.extend([
        "",
        "ENCODING RESULTS:",
        "─" * 50,
        "Sample  Value  Spike Time  Pattern",
        "──────  ─────  ──────────  ───────"
    ])
    
    # Data rows
    for sample in range(len(signals)):
        spike_pattern = ''.join(['█' if s > 0 else '·' for s in spikes[sample]])
        spike_time = torch.argmax(spikes[sample]).item()
        
        text_content.append(
            f"{sample:>6}  {signals[sample]:>5.1f}  {spike_time:>9d}    {spike_pattern}"
        )

    # Add statistics
    text_content.extend(["", "Statistical Summary:", "─" * 50])
    total_spikes = spikes.sum().item()
    avg_spike_time = torch.argmax(spikes.float(), dim=1).float().mean().item()
    text_content.append(f"Total spikes: {int(total_spikes)}")
    text_content.append(f"Average spike time: {avg_spike_time:.1f}")
    text_content.append(f"Spike sparsity: {(total_spikes / (len(signals) * time_steps)) * 100:.1f}%")

    # Dynamic figsize
    text_lines = len(text_content)
    max_char = max(len(line) for line in text_content)
    
    fig_width = (max_char / 10) + 0.5
    fig_height = (text_lines / 15) + 0.5
    
    # Create figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_facecolor('#f8f9fa')
    ax.axis('off')

    # Display text
    ax.text(0.00, 1.00, '\n'.join(text_content), transform=ax.transAxes,
            verticalalignment='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
                     edgecolor='#cccccc'))
    
    # Save images
    plt.tight_layout()
    plt.savefig('latency_terminal_style.pdf', dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig('latency_terminal_style.png', dpi=600, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    
    print("Generated latency encoding terminal-style image: latency_terminal_style.pdf/png")

def compare_with_poisson():
    """与泊松编码对比"""
    print("\n🧪 测试6: 与泊松编码对比")
    
    def simple_poisson_encode(x):
        return torch.rand_like(x).le(x).to(x)
    
    signals = torch.tensor([0.2, 0.5, 0.8])
    time_steps = 10
    
    print(f"信号: {signals}")
    print()
    
    # 泊松编码（多次重复）
    torch.manual_seed(42)
    x_expanded = signals.repeat_interleave(time_steps).view(len(signals), time_steps)
    poisson_spikes = simple_poisson_encode(x_expanded)
    
    # 延迟编码
    latency_spikes = simple_latency_encode(signals, time_steps)
    
    print("泊松编码结果:")
    for i, signal in enumerate(signals):
        pattern = ''.join(['█' if s > 0 else '·' for s in poisson_spikes[i]])
        rate = poisson_spikes[i].mean().item()
        print(f"{signal:.1f}: {pattern} (率={rate:.2f})")
    
    print("\n延迟编码结果:")
    for i, signal in enumerate(signals):
        pattern = ''.join(['█' if s > 0 else '·' for s in latency_spikes[i]])
        spike_time = torch.argmax(latency_spikes[i]).item()
        print(f"{signal:.1f}: {pattern} (时间={spike_time})")
    
    # 功耗对比
    poisson_total_spikes = poisson_spikes.sum().item()
    latency_total_spikes = latency_spikes.sum().item()
    
    print(f"\n功耗对比:")
    print(f"泊松编码总脉冲数: {poisson_total_spikes}")
    print(f"延迟编码总脉冲数: {latency_total_spikes}")
    print(f"延迟编码功耗降低: {(1 - latency_total_spikes/poisson_total_spikes)*100:.1f}%")
    
    return True

def run_all_tests():
    """运行所有测试"""
    print("=" * 50)
    print("🚀 简化延迟编码测试套件")
    print("=" * 50)
    
    tests = [
        # test_basic_functionality,
        # test_encoding_properties,
        # test_batch_processing,
        # test_noise_robustness,
        visualize_encoding,
        # compare_with_poisson
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"💥 测试失败: {e}")
            results.append(False)
    
    # 总结
    print("\n" + "=" * 50)
    print("📊 测试总结")
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！延迟编码实现正确。")
    else:
        print("⚠️ 部分测试失败，请检查实现。")

if __name__ == "__main__":
    run_all_tests()