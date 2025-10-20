import numpy as np
import matplotlib.pyplot as plt

def demonstrate_relationship():
    """演示泊松分布与均匀随机的关系"""
    
    # 1. 泊松过程的基本性质
    print("🔬 泊松过程与均匀随机的关系")
    print("=" * 50)
    
    # 参数设置
    lambda_rate = 0.3  # 泊松参数（平均发生率）
    n_trials = 10000
    
    # 方法1: 直接泊松采样
    np.random.seed(42)
    poisson_samples = np.random.poisson(lambda_rate, n_trials)
    poisson_binary = np.clip(poisson_samples, 0, 1)  # 转为二进制
    
    # 方法2: 均匀随机 + 阈值比较
    np.random.seed(42)
    uniform_samples = np.random.uniform(0, 1, n_trials)
    uniform_binary = (uniform_samples <= lambda_rate).astype(int)
    
    print(f"泊松方法平均值: {poisson_binary.mean():.4f}")
    print(f"均匀方法平均值: {uniform_binary.mean():.4f}")
    print(f"理论期望值: {lambda_rate:.4f}")
    
    return poisson_binary, uniform_binary

def explain_mathematical_basis():
    """解释数学基础"""
    print("\n📐 数学原理")
    print("=" * 30)
    
    print("1. 泊松分布的概率质量函数:")
    print("   P(X = k) = (λ^k * e^(-λ)) / k!")
    print("   当λ很小时，P(X = 0) ≈ 1-λ, P(X = 1) ≈ λ")
    
    print("\n2. 伯努利分布:")
    print("   P(X = 1) = p, P(X = 0) = 1-p")
    print("   当p = λ且λ很小时，泊松(λ) ≈ 伯努利(λ)")
    
    print("\n3. 均匀随机实现伯努利:")
    print("   U ~ Uniform(0,1)")
    print("   X = 1 if U ≤ p else 0")
    print("   则 X ~ 伯努利(p)")

def compare_accuracy():
    """比较不同λ值下的准确性"""
    print("\n🎯 准确性比较")
    print("=" * 30)
    
    lambda_values = [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0]
    n_trials = 10000
    
    print("λ值    泊松方法   均匀方法   理论值    误差1    误差2")
    print("-" * 55)
    
    for lam in lambda_values:
        # 泊松方法
        np.random.seed(42)
        poisson_result = np.clip(np.random.poisson(lam, n_trials), 0, 1).mean()
        
        # 均匀方法
        np.random.seed(42)
        uniform_result = (np.random.uniform(0, 1, n_trials) <= lam).mean()
        
        # 理论值（小λ时的近似）
        theoretical = lam if lam <= 1 else 1
        
        error1 = abs(poisson_result - theoretical)
        error2 = abs(uniform_result - theoretical)
        
        print(f"{lam:.2f}   {poisson_result:.4f}    {uniform_result:.4f}    "
              f"{theoretical:.4f}   {error1:.4f}   {error2:.4f}")

def visualize_distributions():
    """可视化分布差异"""
    print("\n📊 分布可视化")
    
    lambda_vals = [0.1, 0.5, 1.0, 2.0]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, lam in enumerate(lambda_vals):
        # 泊松分布的理论概率
        k_values = np.arange(0, 8)
        poisson_probs = [np.exp(-lam) * (lam**k) / np.math.factorial(k) 
                        for k in k_values]
        
        # 均匀随机近似（只有0和1）
        uniform_prob_0 = 1 - lam if lam <= 1 else 0
        uniform_prob_1 = lam if lam <= 1 else 1
        
        axes[i].bar(k_values, poisson_probs, alpha=0.7, label='泊松分布')
        axes[i].bar([0, 1], [uniform_prob_0, uniform_prob_1], 
                   alpha=0.7, label='均匀近似', width=0.5)
        
        axes[i].set_title(f'λ = {lam}')
        axes[i].set_xlabel('k (事件数)')
        axes[i].set_ylabel('概率')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('poisson_uniform_comparison.png', dpi=150)
    print("图表已保存为 poisson_uniform_comparison.png")

def practical_implications():
    """实际应用含义"""
    print("\n💡 实际应用含义")
    print("=" * 30)
    
    print("✅ 均匀随机方法的优势:")
    print("  - 计算简单：只需一次随机数生成和比较")
    print("  - GPU友好：完全向量化操作")
    print("  - 内存高效：就地操作")
    print("  - 数值稳定：避免阶乘计算")
    
    print("\n⚠️ 适用条件:")
    print("  - λ ≤ 1 时近似最准确")
    print("  - λ > 1 时会有系统性偏差")
    print("  - 只关心二进制输出（0/1脉冲）")
    
    print("\n🎯 神经编码中的应用:")
    print("  - 脉冲神经网络中，神经元每个时间步最多发放1个脉冲")
    print("  - 输入通常是归一化的概率值 [0,1]")
    print("  - 因此均匀随机方法完全适用且更高效")

if __name__ == "__main__":
    demonstrate_relationship()
    explain_mathematical_basis()
    compare_accuracy()
    practical_implications()
    # visualize_distributions()  # 需要matplotlib