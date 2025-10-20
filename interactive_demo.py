#!/usr/bin/env python3
"""
Interactive Spike Encoding Demo
交互式神经脉冲编码演示

Linus式的简洁演示 - 简洁、可靠、无废话
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import spike

# 可选的绘图支持 - 优雅降级
try:
    import matplotlib.pyplot as plt
    import matplotlib.widgets as widgets
    from matplotlib.gridspec import GridSpec

    # 在import matplotlib.pyplot as plt 之后添加
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
        
    HAS_PLOTTING = True
    plt.ion()  # 开启交互模式
except ImportError:
    print("警告: matplotlib未安装，绘图功能不可用")
    HAS_PLOTTING = False

# 常量定义 - 消除魔法数字
DEFAULT_LENGTH = 50
DEFAULT_NEURONS = 30
DEFAULT_TIME_WINDOW = 50.0

def plot_encoding_results(signal, encoder_name='poisson', **kwargs):
    """
    绘制编码前后的波形对比
    这就是Linus式的简洁绘图 - 一个函数搞定一切
    """
    if not HAS_PLOTTING:
        print("绘图功能不可用，请安装matplotlib")
        return None

    # 执行编码测试
    try:
        result = spike.test_encoder(encoder_name, signal, **kwargs)
        spikes = result['spikes']
        reconstructed = result['reconstructed']
        metrics = result['metrics']
    except Exception as e:
        print(f"编码失败: {e}")
        return None

    # 创建图形 - 简洁的布局
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle(f'{encoder_name.capitalize()} Encoding Results', fontsize=14, fontweight='bold')

    # 时间轴
    t_signal = np.arange(len(signal))
    t_spikes = np.arange(spikes.shape[1])
    t_recon = np.arange(len(reconstructed))

    # 1. 原始信号
    axes[0].plot(t_signal, signal, 'b-', linewidth=2, label='Original Signal')
    axes[0].set_title('Original Signal')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 2. 脉冲栅格图 - 确保脉冲清晰可见！
    n_show = min(20, spikes.shape[0])
    spike_display = spikes[:n_show, :]

    print(f"脉冲矩阵形状: {spikes.shape}, 总脉冲数: {np.sum(spikes)}")

    # 创建清晰的脉冲点 - 使用更大的点和明显颜色
    total_visible_spikes = 0
    for neuron in range(n_show):
        spike_times = np.where(spike_display[neuron, :] > 0)[0]
        if len(spike_times) > 0:
            axes[1].scatter(spike_times, [neuron] * len(spike_times),
                          s=50, c='red', marker='|', linewidth=3, alpha=0.9)
            total_visible_spikes += len(spike_times)

    axes[1].set_title(f'Spike Raster Plot ({total_visible_spikes} spikes shown)')
    axes[1].set_ylabel('Neuron ID')
    axes[1].set_xlim(-0.5, spike_display.shape[1] - 0.5)
    axes[1].set_ylim(-0.5, n_show-0.5)
    axes[1].grid(True, alpha=0.3)

    # 如果没有脉冲，显示警告
    if total_visible_spikes == 0:
        axes[1].text(0.5, 0.5, 'No spikes detected!\nCheck encoding parameters',
                    transform=axes[1].transAxes, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    # 3. 重构信号对比
    min_len = min(len(signal), len(reconstructed))
    axes[2].plot(t_signal[:min_len], signal[:min_len], 'b-', linewidth=2,
                label='Original Signal', alpha=0.8)
    axes[2].plot(t_recon[:min_len], reconstructed[:min_len], 'r--', linewidth=2,
                label='Reconstructed Signal', alpha=0.8)
    axes[2].set_title('Signal Reconstruction Comparison')
    axes[2].set_xlabel('Time Steps')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    # 添加性能指标文本
    metrics_text = f"MSE: {metrics['mse']:.6f}\n"
    metrics_text += f"Correlation: {metrics['correlation']:.4f}\n"
    metrics_text += f"Total Spikes: {metrics['total_spikes']}\n"
    metrics_text += f"Spike Density: {metrics['spike_density']:.4f}"

    axes[2].text(0.02, 0.98, metrics_text, transform=axes[2].transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.show()

    return fig

def create_interactive_demo():
    """
    创建交互式演示 - 滑块控制参数
    """
    if not HAS_PLOTTING:
        print("绘图功能不可用，请安装matplotlib")
        return

    # 生成测试信号
    signal = spike.signal_sine(20, freq=1.0)
    
    # 创建图形和子图
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(4, 2, height_ratios=[2, 2, 2, 1], hspace=0.3)
    
    # 子图
    ax_signal = fig.add_subplot(gs[0, :])
    ax_spikes = fig.add_subplot(gs[1, :])
    ax_recon = fig.add_subplot(gs[2, :])
    
    # 滑块区域
    ax_neurons = fig.add_subplot(gs[3, 0])
    ax_rate = fig.add_subplot(gs[3, 1])
    
    # 创建滑块
    slider_neurons = widgets.Slider(ax_neurons, 'Neurons', 5, 50, valinit=25, valfmt='%d')
    slider_rate = widgets.Slider(ax_rate, 'Max Rate (Hz)', 20, 200, valinit=100, valfmt='%d')
    
    def update_plot(val):
        """更新图形"""
        neurons = int(slider_neurons.val)
        max_rate = slider_rate.val
        
        # 重新编码
        try:
            result = spike.test_encoder('poisson', signal, 
                                      neurons=neurons, 
                                      time_window=20.0, 
                                      max_rate=max_rate)
            spikes = result['spikes']
            reconstructed = result['reconstructed']
            metrics = result['metrics']
            
            # 清除旧图
            ax_signal.clear()
            ax_spikes.clear()
            ax_recon.clear()
            
            # 绘制原始信号
            ax_signal.plot(signal, 'b-', linewidth=2, label='Original Signal')
            ax_signal.set_title('Original Signal')
            ax_signal.set_ylabel('Amplitude')
            ax_signal.grid(True, alpha=0.3)
            ax_signal.legend()
            
            # 绘制脉冲
            n_show = min(20, spikes.shape[0])
            for neuron in range(n_show):
                spike_times = np.where(spikes[neuron, :] > 0)[0]
                if len(spike_times) > 0:
                    ax_spikes.scatter(spike_times, [neuron] * len(spike_times),
                                    s=30, c='red', marker='|', alpha=0.8)
            
            ax_spikes.set_title(f'Spike Raster (Total: {np.sum(spikes)} spikes)')
            ax_spikes.set_ylabel('Neuron ID')
            ax_spikes.set_xlim(-0.5, spikes.shape[1] - 0.5)
            ax_spikes.grid(True, alpha=0.3)
            
            # 绘制重构信号
            min_len = min(len(signal), len(reconstructed))
            ax_recon.plot(signal[:min_len], 'b-', linewidth=2, label='Original', alpha=0.8)
            ax_recon.plot(reconstructed[:min_len], 'r--', linewidth=2, label='Reconstructed', alpha=0.8)
            ax_recon.set_title(f'Reconstruction (MSE: {metrics["mse"]:.6f})')
            ax_recon.set_xlabel('Time Steps')
            ax_recon.set_ylabel('Amplitude')
            ax_recon.grid(True, alpha=0.3)
            ax_recon.legend()
            
            plt.draw()
            
        except Exception as e:
            print(f"更新失败: {e}")
    
    # 连接滑块事件
    slider_neurons.on_changed(update_plot)
    slider_rate.on_changed(update_plot)
    
    # 初始绘制
    update_plot(None)
    
    plt.show()
    return fig

def demo_basic():
    """基础功能演示"""
    print("=== 基础功能演示 ===")
    
    signal = spike.signal_sine(DEFAULT_LENGTH, freq=2.0)
    
    try:
        result = spike.test_encoder('poisson', signal,
                                   neurons=DEFAULT_NEURONS,
                                   time_window=DEFAULT_TIME_WINDOW)
        metrics = result['metrics']
        
        print(f"Poisson Encoding:")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  相关系数: {metrics['correlation']:.4f}")
        print(f"  脉冲密度: {metrics['spike_density']:.4f}")
        
    except Exception as e:
        print(f"编码失败: {e}")

def main():
    """主函数"""
    print("Spike编码库演示 (Linus风格)")
    print("=" * 40)
    
    # 基础演示
    demo_basic()
    
    if HAS_PLOTTING:
        print("\n选择演示模式:")
        print("1. 静态结果图")
        print("2. 交互式演示")
        
        try:
            choice = input("请选择 (1/2): ").strip()
            
            if choice == "1":
                signal = spike.signal_sine(30, freq=2.0)
                fig = plot_encoding_results(signal, 'poisson', neurons=25, time_window=30.0)
                if fig:
                    input("按Enter关闭图形...")
                    plt.close(fig)
                    
            elif choice == "2":
                print("启动交互式演示...")
                fig = create_interactive_demo()
                if fig:
                    input("按Enter关闭交互式演示...")
                    plt.close(fig)
            else:
                print("无效选择")
                
        except KeyboardInterrupt:
            print("\n演示被中断")
    else:
        print("\n请安装matplotlib以使用图形演示: pip install matplotlib")
    
    print("\n演示完成！")

if __name__ == "__main__":
    main()
