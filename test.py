# #!/usr/bin/env python3
# """
# Professional Poisson Encoding Visualization
# 符合Nature/Science/IEEE顶刊标准的泊松编码可视化
# """

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
# import seaborn as sns

# # 设置顶刊级别的全局参数
# rcParams.update({
#     'font.family': 'Arial',  # Nature/Science偏好Arial
#     'font.size': 8,          # Nature标准字体大小
#     'axes.linewidth': 0.8,
#     'axes.labelsize': 9,
#     'axes.titlesize': 10,
#     'xtick.labelsize': 8,
#     'ytick.labelsize': 8,
#     'legend.fontsize': 8,
#     'figure.dpi': 300,
#     'savefig.dpi': 300,
#     'savefig.format': 'pdf',
#     'axes.spines.top': False,    # Nature风格：去除上边框
#     'axes.spines.right': False,  # Nature风格：去除右边框
#     'axes.grid': True,
#     'grid.alpha': 0.3,
#     'grid.linewidth': 0.5
# })

# def generate_test_signal(duration=10, fs=100, freq=1.0, noise_level=0.1):
#     """生成测试信号"""
#     t = np.linspace(0, duration, int(duration * fs))
#     # 使用更复杂的信号：正弦波 + 调制 + 噪声
#     signal = np.sin(2 * np.pi * freq * t) * (1 + 0.3 * np.sin(2 * np.pi * 0.2 * t))
#     signal += noise_level * np.random.randn(len(t))
#     # 归一化到[0,1]
#     signal = (signal - signal.min()) / (signal.max() - signal.min())
#     return t, signal

# def poisson_encode_improved(signal, max_rate=50, dt=0.01):
#     """改进的泊松编码实现"""
#     # 计算瞬时发放率
#     lambda_t = signal * max_rate
    
#     # 生成泊松脉冲（更精确的实现）
#     spike_train = np.random.poisson(lambda_t * dt)
#     spike_train = np.clip(spike_train, 0, 1)  # 二值化
    
#     # 计算统计信息
#     avg_rate = np.mean(spike_train) / dt
#     sparsity = np.sum(spike_train) / len(spike_train)
    
#     return spike_train, lambda_t, avg_rate, sparsity

# def create_publication_figure():
#     """创建符合顶刊标准的图表"""
#     # 生成数据
#     t, signal = generate_test_signal(duration=10, fs=100)
#     dt = t[1] - t[0]
#     spike_train, lambda_t, avg_rate, sparsity = poisson_encode_improved(signal, max_rate=50, dt=dt)
    
#     # 创建图表 - Nature风格的单列布局
#     fig = plt.figure(figsize=(3.5, 4.5))  # Nature单列图宽度
    
#     # 子图布局
#     gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.8], hspace=0.4)
    
#     # === 子图a: 输入信号 ===
#     ax1 = fig.add_subplot(gs[0])
#     line1 = ax1.plot(t, signal, color='#2E86AB', linewidth=1.2, label='Input signal')
#     ax1.fill_between(t, 0, signal, alpha=0.2, color='#2E86AB')
    
#     ax1.set_ylabel('Normalized\namplitude', fontweight='bold')
#     ax1.set_ylim(0, 1.05)
#     ax1.set_xlim(0, 10)
#     ax1.text(-0.15, 1.05, 'a', transform=ax1.transAxes, fontsize=12, fontweight='bold')
    
#     # 添加统计信息
#     ax1.text(0.02, 0.95, f'μ = {np.mean(signal):.2f}\nσ = {np.std(signal):.2f}', 
#              transform=ax1.transAxes, fontsize=7, 
#              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
#     # === 子图b: 发放率 ===
#     ax2 = fig.add_subplot(gs[1])
#     ax2.plot(t, lambda_t, color='#A23B72', linewidth=1.2, label='Firing rate')
#     ax2.fill_between(t, 0, lambda_t, alpha=0.2, color='#A23B72')
    
#     ax2.set_ylabel('Firing rate\n(Hz)', fontweight='bold')
#     ax2.set_ylim(0, 55)
#     ax2.set_xlim(0, 10)
#     ax2.text(-0.15, 1.05, 'b', transform=ax2.transAxes, fontsize=12, fontweight='bold')
    
#     # === 子图c: 脉冲序列 ===
#     ax3 = fig.add_subplot(gs[2])
#     spike_times = t[spike_train > 0]
    
#     # 使用scatter plot代替vlines，更适合高密度数据
#     ax3.scatter(spike_times, np.ones(len(spike_times)), 
#                s=0.8, color='#F18F01', alpha=0.8, marker='|')
    
#     ax3.set_ylabel('Spikes', fontweight='bold')
#     ax3.set_xlabel('Time (s)', fontweight='bold')
#     ax3.set_ylim(0.5, 1.5)
#     ax3.set_xlim(0, 10)
#     ax3.set_yticks([1])
#     ax3.text(-0.15, 1.05, 'c', transform=ax3.transAxes, fontsize=12, fontweight='bold')
    
#     # 添加编码统计信息
#     stats_text = f'Rate: {avg_rate:.1f} Hz\nSparsity: {sparsity:.3f}'
#     ax3.text(0.98, 0.95, stats_text, transform=ax3.transAxes, fontsize=7,
#              ha='right', va='top',
#              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
#     # 添加方法说明
#     method_text = ('Poisson encoding: λ(t) = s(t) × λmax\n'
#                   f'λmax = 50 Hz, dt = {dt:.3f} s')
#     fig.text(0.02, 0.02, method_text, fontsize=7, style='italic')
    
#     # 保存高质量图片
#     plt.savefig('poisson_encoding_nature_style.pdf', 
#                 bbox_inches='tight', pad_inches=0.1)
#     plt.savefig('poisson_encoding_nature_style.png', 
#                 bbox_inches='tight', pad_inches=0.1, dpi=300)
    
#     return fig, (avg_rate, sparsity)

# def create_ieee_style_figure():
#     """创建IEEE风格的图表"""
#     # IEEE偏好双列布局
#     rcParams.update({
#         'font.family': 'Times New Roman',
#         'font.size': 9,
#     })
    
#     t, signal = generate_test_signal()
#     dt = t[1] - t[0]
#     spike_train, lambda_t, avg_rate, sparsity = poisson_encode_improved(signal)
    
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7, 5))
    
#     # 原始信号
#     ax1.plot(t, signal, 'b-', linewidth=1.5)
#     ax1.set_title('(a) Input Signal', fontweight='bold')
#     ax1.set_ylabel('Amplitude')
#     ax1.grid(True, alpha=0.3)
    
#     # 发放率
#     ax2.plot(t, lambda_t, 'r-', linewidth=1.5)
#     ax2.set_title('(b) Instantaneous Firing Rate', fontweight='bold')
#     ax2.set_ylabel('Rate (Hz)')
#     ax2.grid(True, alpha=0.3)
    
#     # 脉冲序列
#     spike_times = t[spike_train > 0]
#     ax3.vlines(spike_times, 0, 1, colors='green', linewidth=1.0)
#     ax3.set_title('(c) Poisson Spike Train', fontweight='bold')
#     ax3.set_ylabel('Spikes')
#     ax3.set_xlabel('Time (s)')
#     ax3.grid(True, alpha=0.3)
    
#     # 统计分析
#     hist, bins = np.histogram(np.diff(spike_times), bins=20)
#     ax4.bar(bins[:-1], hist, width=np.diff(bins), alpha=0.7, color='orange')
#     ax4.set_title('(d) Inter-Spike Interval Distribution', fontweight='bold')
#     ax4.set_xlabel('ISI (s)')
#     ax4.set_ylabel('Count')
#     ax4.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig('poisson_encoding_ieee_style.pdf', dpi=300, bbox_inches='tight')
    
#     return fig

# if __name__ == "__main__":
#     print("🎯 生成符合顶刊标准的泊松编码图表...")
    
#     # Nature/Science风格
#     print("📊 创建Nature风格图表...")
#     fig_nature, stats = create_publication_figure()
#     print(f"   平均发放率: {stats[0]:.2f} Hz")
#     print(f"   脉冲稀疏度: {stats[1]:.3f}")
    
#     # IEEE风格
#     print("📊 创建IEEE风格图表...")
#     fig_ieee = create_ieee_style_figure()
    
#     print("✅ 图表生成完成!")
#     print("   - poisson_encoding_nature_style.pdf (Nature/Science)")
#     print("   - poisson_encoding_ieee_style.pdf (IEEE)")
    
#     plt.show()



import matplotlib.pyplot as plt
import numpy as np
#!/usr/bin/env python3
"""
Poisson编码测试文件
简单测试泊松编码的正确性
"""

import numpy as np
import matplotlib.pyplot as plt

# 生成示例时间序列：正弦波（带噪声）
t = np.linspace(0, 10, 250)  # 时间轴（0-10秒，共1000个采样点）
x = np.sin(t) + 0.5 * np.random.randn(len(t))  # 原始时间序列（含噪声）
x = (x - x.min()) / (x.max() - x.min())  # 归一化到[0,1]

max_rate = 20  # 最大脉冲频率（Hz）
x_scaled = x * max_rate  # 映射到[0, max_rate]，作为泊松分布的强度参数λ(t)

print(f"x的长度: {len(x)}")
print(f"x_scaled的长度: {len(x_scaled)}")
print(f"x_scaled:")
print(x_scaled)

dt = 0.04  # 时间步长（单位：秒），即每个采样点的时间间隔
time_steps = len(t)  # 时间步数（与原始时间序列长度一致）

# 初始化脉冲序列（1表示发放脉冲，0表示不发放）
spike_train = np.zeros(time_steps)

# 为每个时间步生成泊松脉冲
for i in range(time_steps):
    # 当前时间步的λ(t) = x_scaled[i]，即该时刻的脉冲频率
    # 泊松分布的参数为：λ * dt（单个时间步内的平均脉冲数）
    lam = x_scaled[i] * dt
    # 生成0或1的脉冲（泊松分布中，当lam较小时，结果为0或1的概率最高）
    # spike = np.random.poisson(lam)
    spike = (np.random.rand(1) <= lam).astype(float)
    
    spike_train[i] = np.clip(spike, 0, 1)  # 确保脉冲非0即1
    print(f"x_scaled[i]: {x_scaled[i]}, lam: {lam}, spike: {spike}, spike_train[i]: {spike_train[i]}")

# print(f"spikes的长度: {len(spike)}")
# print(f"spikes的形状: {spike.shape}")  # 如果是多维数组

print(f"spike_train的长度: {len(spike_train)}")
print(f"spike_train的形状: {spike_train.shape}")  # 如果是多维数组
print(spike_train)


# # 绘图对比
# plt.figure(figsize=(12, 6))
# # 原始时间序列
# plt.subplot(2, 1, 1)
# plt.plot(t, x, label='Normalized Signal')
# plt.ylabel('x(t)')
# plt.legend()
# # 泊松脉冲序列
# plt.subplot(2, 1, 2)
# plt.eventplot(np.where(spike_train)[0] * dt, lineoffsets=0.5, linelengths=0.8, label='Poisson Spikes')
# plt.xlabel('time(s)')
# plt.ylabel('spikes')
# plt.legend()
# plt.tight_layout()
# plt.show()




time = t
signal = x
spikes = spike_train

# 设置顶刊级别的图表参数
plt.rcParams.update({
    'font.size': 12,
    # 'font.family': 'Times New Roman',
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 300
})

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 3.5), sharex=True)

# 上图：归一化信号
ax1.plot(time, signal, 'b-', linewidth=1.5, label='Input Signal')
ax1.set_ylabel('Amplitude', fontweight='bold')
ax1.set_ylim(0, 1.1)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')
ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontweight='bold')
# 去除上图的右边框和上边框
ax1.spines['right'].set_visible(False)  # 隐藏右边框
ax1.spines['top'].set_visible(False)   # 隐藏上边框


# 下图：泊松脉冲
spike_times = np.where(spikes > 0)[0] * dt
ax2.vlines(spike_times, 0, 1, colors='red', linewidth=1.2, label='Poisson Spikes')
ax2.set_ylabel('Spike', fontweight='bold')
ax2.set_xlabel('Time (s)', fontweight='bold')
ax2.set_ylim(-0.1, 1.2)
ax2.set_yticks([0, 1])
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')
ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontweight='bold')
# 去除上图的右边框和上边框
ax2.spines['right'].set_visible(False)  # 隐藏右边框
ax2.spines['top'].set_visible(False)   # 隐藏上边框
# # 添加参数信息
# param_text = f'λ_max = {max_rate} Hz, N = {num_neurons}, T = {duration} s'
# fig.text(0.5, 0.02, param_text, ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('poisson_encoding_improved.pdf', dpi=300, bbox_inches='tight')