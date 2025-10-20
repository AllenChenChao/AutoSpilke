import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
import matplotlib.ticker as ticker  # 添加此行

# 定义泊松分布的概率质量函数
def poisson_pmf(k, lam):
    return np.power(lam, k) * np.exp(-lam) / factorial(k)

# 生成k值，这里取0到20
ks = np.arange(0, 21)
ks = np.arange(0, 16)

# 不同的lam值

lams = [1, 5, 10]
# lams = [0.1, 0.2, 0.5, 0.8, 1]
# lams = [0.1, 0.3, 0.5, 0.7, 1]
lams = [0.1, 0.5, 1, 5]
# 绘制不同lam下的泊松分布
# for lam in lams:
#     if lams == [0.1, 0.2, 0.5, 0.8, 1]:
#         ks = np.arange(0, 5, 0.1)
#         pmf_values = poisson_pmf(ks, lam)
#         plt.plot(ks, pmf_values, marker='o', label=f'λ = {lam}')
#     elif lams == [0.1, 0.3, 0.5, 0.7, 1]:
#         ks = np.arange(0, 5)
#         pmf_values = poisson_pmf(ks, lam)
#         plt.plot(ks, pmf_values, marker='o', label=f'λ = {lam}')
#     else: 
#         pmf_values = poisson_pmf(ks, lam)
#         plt.plot(ks, pmf_values, marker='o', label=f'λ = {lam}')

# plt.xlabel('k (Number of events)')
# plt.ylabel('P(X = k) (Probability)')
# plt.title('Poisson Distribution')
# plt.legend()
# plt.grid(True)
# plt.show()


# ...existing code...

# 绘制不同lam下的泊松分布（顶刊风格）
plt.rcParams.update({
    'font.family': 'sans-serif', #'serif',  # 顶刊偏好
    'font.size': 9,
    'axes.linewidth': 0.8,
    'pdf.fonttype': 42,  # 矢量字体
    'savefig.dpi': 600
})

fig, ax = plt.subplots(figsize=(3.5, 2.5))  # IEEE单列宽

for lam in lams:
    pmf_values = poisson_pmf(ks, lam)
    # ax.plot(ks, pmf_values, marker='o', markersize=3, linewidth=1.5, 
    #         linestyle='-' if lam == 0.1 else '--' if lam == 0.5 else '-.', 
    #         label=f'λ = {lam}', color='black' if lam == 0.1 else '0.5' if lam == 0.5 else '0.2')  # 灰度友好
    # 修改线型和颜色，确保每个λ唯一
    if lam == 0.1:
        linestyle, color = '-', 'black'
    elif lam == 0.5:
        linestyle, color = '--', '0.4'
    elif lam == 1:
        linestyle, color = '-.', '0.6'
    else:  # lam == 5
        linestyle, color = ':', '0.2'
    
    ax.plot(ks, pmf_values, marker='o', markersize=3, linewidth=1.5, 
            linestyle=linestyle, color=color, label=f'λ = {lam}')



ax.set_xlabel('k (Number of Events)', fontsize=10)
ax.set_ylabel('P(X = k) (Probability)', fontsize=10)
# ax.set_title('Poisson Distribution PMF', fontsize=11, fontweight='bold')
ax.legend(frameon=False, fontsize=8)
ax.grid(True, alpha=0.3, linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 强制x轴显示整数刻度
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))  # 每1个单位一个刻度
ax.set_xticks([x for x in range(0,16,2)])  # 
ax.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])  # 

# 添加参数注释
ax.annotate('Parameters: k ∈ [0, 15], λ ∈ {0.1, 0.5, 1, 5}', 
            xy=(0.02, 1.02), xycoords='axes fraction', fontsize=7, 
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('poisson_pmf_ieee.pdf', bbox_inches='tight')  # 矢量PDF
plt.savefig('poisson_pmf_ieee.eps', bbox_inches='tight')  # EPS备份
plt.show()
