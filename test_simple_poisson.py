#!/usr/bin/env python3
"""
测试简化的泊松编码实现
基于均匀随机数的高效实现
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

def test_basic_functionality():
    """测试基本功能"""
    print("🧪 测试1: 基本功能")
    
    # 测试数据：不同的发放概率
    probs = torch.tensor([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    print(f"输入概率: {probs}")
    
    # 编码
    torch.manual_seed(42)
    spikes = simple_poisson_encode(probs)
    print(f"输出脉冲: {spikes}")
    print(f"输出类型: {spikes.dtype}")
    print(f"输出范围: [{spikes.min():.0f}, {spikes.max():.0f}]")
    
    return True

def test_statistical_properties():
    """测试统计特性"""
    print("\n🧪 测试2: 统计特性")
    
    # 大量重复测试
    prob = 0.3  # 30%发放概率
    n_trials = 10000
    
    torch.manual_seed(42)
    x = torch.full((n_trials,), prob)
    spikes = simple_poisson_encode(x)
    
    observed_rate = spikes.mean().item()
    print(f"期望发放率: {prob:.3f}")
    print(f"观测发放率: {observed_rate:.3f}")
    print(f"误差: {abs(prob - observed_rate):.3f}")
    
    # 检查是否在合理范围内（3σ）
    std_error = np.sqrt(prob * (1-prob) / n_trials)
    tolerance = 3 * std_error
    
    success = abs(prob - observed_rate) < tolerance
    print(f"统计检验: {'✅ 通过' if success else '❌ 失败'}")
    
    return success

def test_different_rates():
    """测试不同发放率"""
    print("\n🧪 测试3: 不同发放率")
    
    rates = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    n_samples = 1000
    
    print("概率 -> 观测率 (误差)")
    print("-" * 25)
    
    all_passed = True
    for rate in rates:
        torch.manual_seed(42)
        x = torch.full((n_samples,), rate)
        spikes = simple_poisson_encode(x)
        observed = spikes.mean().item()
        error = abs(rate - observed)
        
        print(f"{rate:.2f} -> {observed:.3f} ({error:.3f})")
        
        # 宽松的误差检查
        if error > 0.05:  # 5%误差容忍
            all_passed = False
    
    print(f"结果: {'✅ 全部通过' if all_passed else '❌ 部分失败'}")
    return all_passed

def test_batch_processing():
    """测试批处理"""
    print("\n🧪 测试4: 批处理")
    
    # 模拟神经网络输入：(batch_size, features, time_steps)
    batch_size, features, time_steps = 4, 10, 50
    
    # 随机概率矩阵
    torch.manual_seed(42)
    probs = torch.rand(batch_size, features, time_steps) * 0.8  # [0, 0.8]
    
    print(f"输入形状: {probs.shape}")
    print(f"概率范围: [{probs.min():.3f}, {probs.max():.3f}]")
    
    # 编码
    spikes = simple_poisson_encode(probs)
    
    print(f"输出形状: {spikes.shape}")
    print(f"总脉冲数: {spikes.sum().item():.0f}")
    print(f"平均发放率: {spikes.mean():.3f}")
    
    # 检查每个batch的发放率
    for i in range(batch_size):
        batch_rate = spikes[i].mean().item()
        expected_rate = probs[i].mean().item()
        print(f"Batch {i}: 期望={expected_rate:.3f}, 观测={batch_rate:.3f}")
    
    return True

def test_reproducibility():
    """测试可重现性"""
    print("\n🧪 测试5: 可重现性")
    
    x = torch.tensor([0.2, 0.5, 0.8])
    
    # 相同种子
    torch.manual_seed(123)
    spikes1 = simple_poisson_encode(x)
    
    torch.manual_seed(123)
    spikes2 = simple_poisson_encode(x)
    
    identical = torch.equal(spikes1, spikes2)
    print(f"相同种子结果一致: {'✅' if identical else '❌'}")
    
    # 不同种子
    torch.manual_seed(456)
    spikes3 = simple_poisson_encode(x)
    
    different = not torch.equal(spikes1, spikes3)
    print(f"不同种子结果不同: {'✅' if different else '❌'}")
    
    return identical and different

## 原来直接输出到终端的可视化
# def visualize_encoding():
#     """可视化编码结果"""
#     print("\n🧪 测试6: 可视化")
    
#     # 不同概率的脉冲模式
#     # probs = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
#     probs = torch.tensor([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
#     time_steps = 20
    
#     print("脉冲模式可视化:")
#     print("概率  脉冲模式")
#     print("-" * 30)
    
#     torch.manual_seed(42)
#     for i, prob in enumerate(probs):
#         x = torch.full((time_steps,), prob)
#         spikes = simple_poisson_encode(x)
        
#         # 转换为可视化字符
#         pattern = ''.join(['█' if s > 0 else '·' for s in spikes])
#         rate = spikes.mean().item()
        
#         print(f"{prob:.1f}   {pattern} ({rate:.2f})")
    
#     return True


## doubao 模仿终端的可视化
# def visualize_encoding():
#     """Generate high-quality image with terminal-style output for Poisson encoding (using simple_poisson_encode)"""
#     # Test data
#     data = np.array([[0.0], [0.1], [0.3], [0.5], [0.7], [0.9], [1.0]])
#     time_steps = 20
    
#     # Generate spikes using simple_poisson_encode
#     torch.manual_seed(42)  # Ensure reproducibility
#     spikes = []
#     for value in data:
#         # Create tensor with the same value repeated for time steps
#         x = torch.full((time_steps,), float(value))
#         # Apply Poisson encoding using the provided function
#         spike_train = simple_poisson_encode(x)
#         spikes.append(spike_train.numpy())
#     spikes = np.array(spikes).reshape(len(data), 1, time_steps)
    
#     # Set style - mimic terminal appearance
#     plt.rcParams['font.family'] = 'DejaVu Sans Mono'  # Monospace font for terminal look
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
#     avg_rate = total_spikes / len(data) / time_steps
#     text_content.append(f"Total spikes: {int(total_spikes)}")
#     text_content.append(f"Spike sparsity: {avg_rate*100:.1f}%")
#     text_content.append(f"Bit Efficiency: {total_spikes/(len(data)*32)*100:.1f}%")


#     # 动态计算figsize
#     text_lines = len(text_content)  # 文本总行数
#     max_char = max(len(line) for line in text_content)  # 单行长最大字符数
    
#     # 宽度：10字符/英寸（等宽字体11号），加0.5英寸余量
#     fig_width = (max_char / 10) + 0.5  
#     # 高度：15行/英寸（等宽字体11号），加0.5英寸余量
#     fig_height = (text_lines / 15) + 0.5  
    
#     # 用动态尺寸创建画布
#     fig, ax = plt.subplots(figsize=(fig_width, fig_height))
#     ax.set_facecolor('#f8f9fa')  # 浅灰色背景
#     ax.axis('off')  # 隐藏坐标轴

#     # 在图中显示文本
#     ax.text(0.00, 1.00, '\n'.join(text_content), transform=ax.transAxes,
#             verticalalignment='top', fontsize=11, 
#             bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
#                      edgecolor='#cccccc'))
    
#     # 保存高质量图像
#     plt.tight_layout()
#     plt.savefig('terminal_output_style.pdf', dpi=300, bbox_inches='tight', 
#                 facecolor=fig.get_facecolor(), edgecolor='none')
#     plt.savefig('terminal_output_style.png', dpi=600, bbox_inches='tight',
#                 facecolor=fig.get_facecolor(), edgecolor='none')
#     plt.close()
    
#     print("Generated terminal-style output image: terminal_output_style.pdf/png")

## augment 
def visualize_encoding():
    """Generate high-quality image with terminal-style output using simple_poisson_encode"""
    # Test data
    data = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    time_steps = 20
    
    # 使用simple_poisson_encode生成脉冲
    torch.manual_seed(42)
    spikes = np.zeros((len(data), 1, time_steps))
    
    for sample in range(len(data)):
        # 为每个样本生成time_steps个脉冲
        prob_tensor = torch.full((time_steps,), data[sample])
        spike_tensor = simple_poisson_encode(prob_tensor)
        spikes[sample, 0, :] = spike_tensor.numpy()
    
    # Set style - mimic terminal appearance
    plt.rcParams['font.family'] = 'DejaVu Sans Mono'
    plt.rcParams['font.size'] = 11
    
    # Create text content in English
    text_content = [
        "Simple Poisson Encoding Pattern (20 time steps):",
        ""
    ]
    
    # 简洁的表头
    text_content.extend([
        "",
        "ENCODING RESULTS:",
        "─" * 50,
        "Sample  Value  Spikes    Pattern",
        "──────  ─────  ──────    ───────"
    ])
    
    # 简洁的数据行
    for sample in range(len(data)):
        spike_pattern = ''.join(['█' if s > 0 else '·' for s in spikes[sample, 0, :]])
        count = int(np.sum(spikes[sample, 0, :]))
        
        text_content.append(
            f"{sample:>6}  {data[sample]:>5.1f}  {count:>3d}/{time_steps}    {spike_pattern}"
        )

    # Add statistics
    text_content.extend(["", "Statistical Summary:", "─" * 50])
    total_spikes = np.sum(spikes)
    avg_rate = total_spikes / len(data) / time_steps
    text_content.append(f"Total spikes: {int(total_spikes)}")
    text_content.append(f"Spike sparsity: {avg_rate*100:.1f}%")
    text_content.append(f"Bit Efficiency: {total_spikes/(len(data)*32)*100:.1f}%")

    # 动态计算figsize
    text_lines = len(text_content)
    max_char = max(len(line) for line in text_content)
    
    fig_width = (max_char / 10) + 0.5  
    fig_height = (text_lines / 15) + 0.5  
    
    # 创建画布
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
    plt.savefig('simple_poisson_terminal.pdf', dpi=300, bbox_inches='tight', 
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig('simple_poisson_terminal.png', dpi=600, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    
    print("Generated simple poisson encoding image: simple_poisson_terminal.pdf/png")

def run_all_tests():
    """运行所有测试"""
    print("=" * 50)
    print("🚀 简化泊松编码测试套件")
    print("=" * 50)
    
    tests = [
        # test_basic_functionality,
        # test_statistical_properties,
        # test_different_rates,
        # test_batch_processing,
        # test_reproducibility,
        visualize_encoding
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
        print("🎉 所有测试通过！编码实现正确。")
    else:
        print("⚠️ 部分测试失败，请检查实现。")

if __name__ == "__main__":
    run_all_tests()