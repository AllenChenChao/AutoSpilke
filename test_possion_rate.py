#!/usr/bin/env python3
"""
独立测试新的速率泊松编码
测试encode_poisson_rate和decode_poisson_rate函数
"""



import sys
sys.path.append('src')
import numpy as np
import matplotlib.pyplot as plt
import torch

# 直接实现测试函数，避免依赖可能有问题的代码
def encode_poisson_rate_test(data, time_steps=10, dt=1.0, max_rate=100.0, seed=None):
    """测试版本的速率泊松编码"""
    if len(data.shape) == 1:
        data = data.reshape(1, -1)
    
    n_samples, n_features = data.shape
    
    # 归一化到[0,1]
    data_norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-8)
    
    spike_trains = np.zeros((n_samples, n_features, time_steps))

    print(f"data.min(axis=0): {data.min(axis=0)}")
    print(f"data.max(axis=0): {data.max(axis=0)}")
    print(f"n_samples: {n_samples}")
    print(f"n_features: {n_features}")
    print(f"time_steps: {time_steps}")
    print(f"dt: {dt}")
    print(f"max_rate: {max_rate}")
    print(f"seed: {seed}")
    print(f"data: {data}")
    print(f"data_norm: {data_norm}")


    if seed is not None:
        np.random.seed(seed)
    
    for i in range(n_samples):
        for j in range(n_features):
            # 转换为发放率 (参考您提供的代码逻辑)
            spike_rate = data_norm[i, j]  # 直接使用归一化值作为概率
            
            # 生成泊松脉冲
            spikes = np.random.poisson(spike_rate * dt, time_steps)
            
            spike_trains[i, j, :] = np.clip(spikes, 0, 1)

            print(f"spike_rate: {spike_rate}")
            print(f"dt: {dt}")
            print(f"time_steps: {time_steps}")
            print(f"spikes: {spikes}")
            # print(f"spike_trains before: {spike_trains}")
            print(f"spike_trains after: {spike_trains[i, j, :]}")
            

    return spike_trains

def decode_poisson_rate_test(spike_trains, dt=1.0, max_rate=100.0):
    """测试版本的速率泊松解码"""
    if spike_trains.ndim == 2:
        spike_trains = spike_trains.reshape(1, spike_trains.shape[0], spike_trains.shape[1])
    
    n_samples, n_features, time_steps = spike_trains.shape
    
    # 计算每个特征的脉冲计数
    spike_counts = np.sum(spike_trains, axis=2)
    
    # 转换回特征值 (简单平均)
    reconstructed = spike_counts / time_steps
    
    return reconstructed

def test_basic_functionality():
    """测试基本功能"""
    print("=" * 50)
    print("测试1: 基本功能测试")
    print("=" * 50)
    
    # 简单测试数据
    data = np.array([[0.0], [0.2], [0.8], [1.0], [0.5], [0.1]])
    print(f"原始数据: {data}")
    print(f"数据形状: {data.shape}")
    
    # 编码
    spikes = encode_poisson_rate_test(data, time_steps=50, seed=42)
    print(f"脉冲形状: {spikes.shape}")
    print(f"总脉冲数: {np.sum(spikes)}")
    
    # 解码
    reconstructed = decode_poisson_rate_test(spikes)
    print(f"重构数据: {reconstructed}")
    print(f"重构形状: {reconstructed.shape}")
    
    # 计算误差
    mse = np.mean((data - reconstructed) ** 2)
    print(f"MSE: {mse:.6f}")
    
    return mse < 0.5  # 宽松的测试条件

def test_multiple_samples():
    """测试多样本"""
    print("\n" + "=" * 50)
    print("测试2: 多样本测试")
    print("=" * 50)
    
    # 多样本数据
    data = np.array([[0.2, 0.8, 1.0],
                     [0.9, 0.3, 0.6],
                     [0.5, 0.7, 0.4]])
    print(f"原始数据:\n{data}")
    
    # 编码
    spikes = encode_poisson_rate_test(data, time_steps=100, seed=42)
    print(f"脉冲形状: {spikes.shape}")
    
    # 解码
    reconstructed = decode_poisson_rate_test(spikes)
    print(f"重构数据:\n{reconstructed}")
    
    # 逐样本比较
    for i in range(data.shape[0]):
        mse = np.mean((data[i] - reconstructed[i]) ** 2)
        print(f"样本{i} MSE: {mse:.6f}")
    
    total_mse = np.mean((data - reconstructed) ** 2)
    print(f"总体MSE: {total_mse:.6f}")
    
    return total_mse < 0.5

def test_edge_cases():
    """测试边界情况"""
    print("\n" + "=" * 50)
    print("测试3: 边界情况测试")
    print("=" * 50)
    
    # 测试全零
    data_zeros = np.array([[0.0], [0.0], [0.0]])
    spikes_zeros = encode_poisson_rate_test(data_zeros, time_steps=50, seed=42)
    print(f"全零数据脉冲数: {np.sum(spikes_zeros)}")
    
    # 测试全一
    data_ones = np.array([[1.0], [1.0], [1.0]])
    spikes_ones = encode_poisson_rate_test(data_ones, time_steps=50, seed=42)
    print(f"全一数据脉冲数: {np.sum(spikes_ones)}")
    
    # 测试单个值
    data_single = np.array([0.5])
    spikes_single = encode_poisson_rate_test(data_single, time_steps=50, seed=42)
    reconstructed_single = decode_poisson_rate_test(spikes_single)
    print(f"单值测试: {data_single} -> {reconstructed_single}")
    
    return True

def test_reproducibility():
    """测试可重现性"""
    print("\n" + "=" * 50)
    print("测试4: 可重现性测试")
    print("=" * 50)
    
    data = np.array([[0.3], [0.7], [0.9]])
    
    # 两次相同种子编码
    spikes1 = encode_poisson_rate_test(data, time_steps=50, seed=123)
    spikes2 = encode_poisson_rate_test(data, time_steps=50, seed=123)
    
    # 检查是否完全相同
    identical = np.array_equal(spikes1, spikes2)
    print(f"相同种子结果一致: {identical}")
    
    # 不同种子编码
    spikes3 = encode_poisson_rate_test(data, time_steps=50, seed=456)
    different = not np.array_equal(spikes1, spikes3)
    print(f"不同种子结果不同: {different}")
    
    return identical and different

# def visualize_encoding():
#     """可视化编码结果"""
#     print("\n" + "=" * 50)
#     print("测试5: 可视化编码")
#     print("=" * 50)
    
#     data = np.array([[0.0], [0.1], [0.5], [0.9], [1.0]])
#     spikes = encode_poisson_rate_test(data, time_steps=10, seed=42)

    
#     print("脉冲模式可视化 (前20个时间步):")
#     # for feat in range(data.shape[1]):
#     #     spike_pattern = ''.join(['█' if s > 0 else '·' for s in spikes[0, feat, :]])
#     #     print(f"样本{feat} (值={data[0,feat]:.1f}): {spike_pattern}")
#     for sample in range(data.shape[0]):  # 遍历样本，不是特征
#         spike_pattern = ''.join(['█' if s > 0 else '·' for s in spikes[sample, 0, :]])  # 第0个特征
#         print(f"样本{sample} (值={data[sample,0]:.1f}): {spike_pattern}")    
#     return True

# def visualize_encoding():
#     """可视化编码结果 - 模仿终端ASCII艺术的matplotlib图"""
#     print("\n" + "=" * 50)
#     print("测试5: 可视化编码")
#     print("=" * 50)
    
#     data = np.array([[0.0], [0.1], [0.5], [0.9], [1.0]])
#     spikes = encode_poisson_rate_test(data, time_steps=10, seed=42)
    
#     # 终端打印（保持原样）
#     print("脉冲模式可视化 (前10个时间步):")
#     for sample in range(data.shape[0]):
#         spike_pattern = ''.join(['█' if s > 0 else '·' for s in spikes[sample, 0, :]])
#         print(f"样本{sample} (值={data[sample,0]:.1f}): {spike_pattern}")
    
#     # matplotlib图：模仿终端布局（左边值，右边字符点阵）
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.set_xlim(-1, 10)  # 左边留空间放值
#     ax.set_ylim(-0.5, len(data) - 0.5)
#     ax.axis('off')  # 隐藏轴，纯“终端”风格
    
#     for i in range(len(data)):
#         # 左边：样本值（文本）
#         ax.text(-0.8, i, f'x={data[i,0]:.1f}:', ha='right', va='center', fontsize=12, fontfamily='monospace')
        
#         # 右边：脉冲字符（用text画ASCII）
#         for t in range(10):
#             char = '█' if spikes[i, 0, t] > 0 else '·'
#             ax.text(t, i, char, ha='center', va='center', fontsize=14, fontfamily='monospace', color='black')
    
#     plt.tight_layout()
#     plt.savefig('poisson_terminal_style.pdf', dpi=300, bbox_inches='tight')  # 矢量输出
#     plt.show()
    
#     return True

# 模仿终端进行画图
# def visualize_encoding():
#     """可视化编码结果 - 模仿终端ASCII艺术的matplotlib图（脉冲更靠近）"""
#     print("\n" + "=" * 50)
#     print("测试5: 可视化编码")
#     print("=" * 50)
    
#     data = np.array([[0.0], [0.1], [0.5], [0.9], [1.0]])
#     spikes = encode_poisson_rate_test(data, time_steps=10, seed=42)
    
#     # 终端打印（保持原样）
#     print("脉冲模式可视化 (前10个时间步):")
#     for sample in range(data.shape[0]):
#         spike_pattern = ''.join(['█' if s > 0 else '·' for s in spikes[sample, 0, :]])
#         print(f"样本{sample} (值={data[sample,0]:.1f}): {spike_pattern}")
    
#     # matplotlib图：模仿终端布局（左边值，右边字符点阵，更靠近）
#     fig, ax = plt.subplots(figsize=(5, 2))  # 缩小宽度，让字符更挤
#     ax.set_xlim(-1, 8)  # 缩短x范围，让10个字符挤在更小空间
#     ax.set_ylim(-0.5, len(data) - 0.5)
#     # ax.axis('off')  # 隐藏轴，纯“终端”风格
    
#     for i in range(len(data)):
#         # 左边：样本值（文本）
#         ax.text(-0.8, i, f'x={data[i,0]:.1f}:', ha='right', va='center', fontsize=10, fontfamily='monospace')
        
#         # 右边：脉冲字符（用text画ASCII，更小字体让位置靠近）
#         for t in range(10):
#             char = '█' if spikes[i, 0, t] > 0 else '·'
#             x_pos = -0.5 + t * 0.5  # 调整x位置，让字符间隔0.2（更近）
#             # x_pos = -0.5 + t * 0.8  # 调整x位置，让字符间隔0.8（更近）
#             ax.text(x_pos, i, char, ha='center', va='center', fontsize=12, fontfamily='monospace', color='black')
    
#     plt.tight_layout()
#     plt.savefig('poisson_terminal_compact.pdf', dpi=300, bbox_inches='tight')  # 矢量输出
#     plt.show()
    
#     return True


# ...existing code...

def generate_tikz_code(data, spikes, time_steps=10):
    """生成TikZ LaTeX代码，用图形代替字符"""
    tikz_code = []
    tikz_code.append("\\begin{tikzpicture}")
    
    y_offset = 0
    for i in range(len(data)):
        # 左边：样本值（文本）
        tikz_code.append(f"\\node[font=\\ttfamily] at (-1, {-y_offset}) {{x={data[i,0]:.1f}:}};")
        
        # 右边：脉冲图形
        for t in range(time_steps):
            x_pos = t * 0.5
            if spikes[i, 0, t] > 0:
                # '█' 代替：画小黑方块
                tikz_code.append(f"\\fill ({x_pos-0.2}, {-y_offset-0.2}) rectangle ({x_pos+0.2}, {-y_offset+0.2});")
            else:
                # '·' 代替：画小灰点或空白（可选画小圆）
                tikz_code.append(f"\\fill[gray] ({x_pos}, {-y_offset}) circle (0.05);")  # 小灰点
        
        y_offset += 1  # 下一行
    
    tikz_code.append("\\end{tikzpicture}")
    return "\n".join(tikz_code)

# tikz_code的代码生成
# def visualize_encoding():
#     """可视化编码结果 - 生成TikZ代码"""
#     print("\n" + "=" * 50)
#     print("测试5: 可视化编码")
#     print("=" * 50)
    
#     data = np.array([[0.0], [0.1], [0.5], [0.9], [1.0]])
#     spikes = encode_poisson_rate_test(data, time_steps=10, seed=42)
    
#     # 终端打印（保持原样）
#     print("脉冲模式可视化 (前10个时间步):")
#     for sample in range(data.shape[0]):
#         spike_pattern = ''.join(['█' if s > 0 else '·' for s in spikes[sample, 0, :]])
#         print(f"样本{sample} (值={data[sample,0]:.1f}): {spike_pattern}")
    
#     # 生成TikZ LaTeX代码
#     tikz_latex = generate_tikz_code(data, spikes)
#     print("\n📄 TikZ LaTeX代码（复制到.tex文件）：")
#     print(tikz_latex)
    
#     # 保存到文件（可选）
#     with open('poisson_tikz.tex', 'w') as f:
#         f.write(tikz_latex)
#     print("✅ TikZ代码已保存到 poisson_tikz.tex")
    
#     return True


def visualize_encoding():
    """Generate high-quality image with terminal-style output (English only)"""
    # Test data
    data = np.array([[0.0], [0.1],[0.3], [0.5], [0.7], [0.9], [1.0]])
    spikes = encode_poisson_rate_test(data, time_steps=20, seed=42)
    
    # # Create figure
    # fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set style - mimic terminal appearance
    plt.rcParams['font.family'] = 'DejaVu Sans Mono'  # Monospace font for terminal look
    plt.rcParams['font.size'] = 11
    # ax.set_facecolor('#f8f9fa')  # Light gray background
    
    # Hide axes
    # ax.axis('off')
    
    # Create text content in English
    text_content = [
        "Poisson Encoding Pattern (20 time steps):",
        ""
    ]
    
    # for sample in range(data.shape[0]):
    #     spike_pattern = ''.join(['█' if s > 0 else '·' for s in spikes[sample, 0, :]])
    #     count = int(np.sum(spikes[sample, 0, :]))
    #     text_content.append(f"Sample {sample} (value={data[sample,0]:.1f}, spikes={count:2d}/20):{spike_pattern}")
    
    # 简洁的表头
    text_content.extend([
        "",
        "ENCODING RESULTS:",
        "─" * 50,
        "Sample  Value  Spikes    Pattern",
        "──────  ─────  ──────    ───────"
    ])
    
    # 简洁的数据行
    for sample in range(data.shape[0]):
        spike_pattern = ''.join(['█' if s > 0 else '·' for s in spikes[sample, 0, :]])
        count = int(np.sum(spikes[sample, 0, :]))
        
        text_content.append(
            f"{sample:>6}  {data[sample,0]:>5.1f}  {count:>3d}/{20}    {spike_pattern}"
        )

    # Add statistics
    text_content.extend(["", "Statistical Summary:", "─" * 50])
    total_spikes = np.sum(spikes)
    avg_rate = total_spikes / len(data) / 20 #* 100  # Assuming max_rate=100Hz
    text_content.append(f"Total spikes: {int(total_spikes)}")
    # text_content.append(f"Average firing rate: {avg_rate:.2f}")# Hz")
    text_content.append(f"Spike sparsity: {avg_rate*100:.1f}%")# Hz")

    # text_content.append(f"Encoding efficiency: {total_spikes/(len(data)*20)*100:.1f}%")
    text_content.append(f"Bit Efficiency: {total_spikes/(len(data)*32)*100:.1f}%")


    # 2. 动态计算figsize（核心步骤）
    text_lines = len(text_content)  # 文本总行数
    max_char = max(len(line) for line in text_content)  # 单行长最大字符数
    font_size = 11  # 与你的字体大小匹配
    
    # 宽度：10字符/英寸（等宽字体11号），加0.5英寸余量
    fig_width = (max_char / 10) + 0.5  
    # 高度：15行/英寸（等宽字体11号），加0.5英寸余量
    fig_height = (text_lines / 15) + 0.5  
    
    # 3. 用动态尺寸创建画布（无多余空白）
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_facecolor('#f8f9fa')  # Light gray background
    # 后续原代码（隐藏轴、添加文本、保存）...
    ax.axis('off') 

    # Display text in the figure
    # ax.text(0.02, 0.98, '\n'.join(text_content), transform=ax.transAxes,
    #         verticalalignment='top', fontsize=11, 
    #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
    #                  edgecolor='#cccccc'))
    ax.text(0.00, 1.00, '\n'.join(text_content), transform=ax.transAxes,
            verticalalignment='top', fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
                     edgecolor='#cccccc'))
    
    # Add title
    # ax.set_title('Poisson Rate Encoding - Terminal Output Simulation', 
    #             fontsize=14, fontweight='bold', pad=20, fontfamily='DejaVu Sans')
    
    # Save high-quality images
    plt.tight_layout()
    plt.savefig('terminal_output_style.pdf', dpi=300, bbox_inches='tight', 
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig('terminal_output_style.png', dpi=600, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    # plt.savefig('terminal_output_style.pdf', dpi=300, bbox_inches=plt.tight_layout(rect=[0, 0, 1, 1]), facecolor=fig.get_facecolor(), edgecolor='none')
    # plt.savefig('terminal_output_style.png', dpi=600, bbox_inches=plt.tight_layout(rect=[0, 0, 1, 1]), facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    
    print("Generated terminal-style output image: terminal_output_style.pdf/png")


def main():
    """主测试函数"""
    print("🧪 速率泊松编码独立测试")
    print("测试未经验证的新代码...")
    
    tests = [
        # ("基本功能", test_basic_functionality),
        # ("多样本", test_multiple_samples),
        # ("边界情况", test_edge_cases),
        # ("可重现性", test_reproducibility),
        ("可视化", visualize_encoding)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                print(f"✅ {name}: 通过")
                passed += 1
            else:
                print(f"❌ {name}: 失败")
        except Exception as e:
            print(f"💥 {name}: 异常 - {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！代码可以使用。")
    else:
        print("⚠️ 部分测试失败，需要修复代码。")

if __name__ == "__main__":
    main()

