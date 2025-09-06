import os
from math import gamma

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import hilbert
from scipy.special import gamma
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置支持中文字体
plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def residuals_nakagami_lm(params, X, envelope_true, num_echoes):
    """
    用于 LM 优化的残差函数：模型包络 - Hilbert 包络。
    """
    betas = params[:num_echoes]
    tau1 = params[num_echoes]
    delta = params[num_echoes + 1]
    fc = params[num_echoes + 2]
    phi = params[num_echoes + 3]
    omega = params[num_echoes + 4]

    signal_model = nakagami_echo_simplified(betas, tau1, delta, fc, phi, omega, X)
    envelope_model = np.abs(hilbert(signal_model))

    return envelope_model - envelope_true


# ------------------------------ (1) 之前已经定义好的函数 ------------------------------

def nakagami_echo_simplified(betas, tau1, delta, fc, phi, omega, X):
    """
    生成简化后的 Nakagami 回波信号，m_fixed 固定为 1.3。
    betas: 长度 = N（回波数目）的数组，各回波的幅值 β_i
    tau1:  第一个回波到达时间 (秒)
    delta: 回波间隔 (秒)
    fc, phi, omega: 全局共享参数
    X: 时间向量 (秒)
    """
    m_fixed = 1.3
    num_echoes = len(betas)
    signal = np.zeros_like(X)

    for i in range(num_echoes):
        beta_i = betas[i]
        tau_i  = tau1 + i * delta

        t_shifted = X - tau_i
        t_pos = np.maximum(t_shifted, 0.0)
        u = (t_shifted > 0).astype(float)

        envelope_component = (
            beta_i
            * (2 * (m_fixed ** m_fixed))
            * (t_pos ** (2*m_fixed - 1))
            * np.exp(-(m_fixed / omega**2) * (t_pos**2))
            * u
        ) / (gamma(m_fixed) * (omega ** (2*m_fixed)))

        signal += envelope_component * np.cos(2*np.pi*fc*t_shifted + phi) * u

    return signal


def fitness_nakagami_simplified(PARAMS, X, envelope_true):
    """
    计算简化模型包络与真实包络之间的平方误差之和。
    PARAMS: [β₁,…,β_N, τ₁, Δ, f_c, φ, ω]，长度 = N + 5
    X: 时间向量 (秒)
    envelope_true: 真实信号包络，同 X 等长
    """
    num_echoes = len(PARAMS) - 5
    betas = PARAMS[0:num_echoes]
    tau1  = PARAMS[num_echoes]
    delta = PARAMS[num_echoes + 1]
    fc    = PARAMS[num_echoes + 2]
    phi   = PARAMS[num_echoes + 3]
    omega = PARAMS[num_echoes + 4]

    signal_model = nakagami_echo_simplified(betas, tau1, delta, fc, phi, omega, X)
    envelope_model = np.abs(hilbert(signal_model))

    return np.sum((envelope_model - envelope_true)**2)


def ade_optimize(bounds, fitness_func, pop_size=50, max_gen=200, F=0.5, CR=0.9, tol=1e-8):
    """
    自适应差分进化 (ADE)。
    bounds: 一个长度为 dim 的列表，每项为 (min, max)。
    fitness_func: 接受一个 dim 维向量，返回标量误差。
    返回最优的 dim 维参数向量。
    """
    dim = len(bounds)
    # 1. 初始化
    pop = np.array([
        np.random.uniform(low, high, size=pop_size)
        for (low, high) in bounds
    ]).T  # (pop_size, dim)

    fitness_values = np.array([fitness_func(ind) for ind in pop])
    best_idx = np.argmin(fitness_values)
    best = pop[best_idx].copy()
    best_fitness = fitness_values[best_idx]

    # 2. 迭代
    for gen in range(1, max_gen + 1):
        denom = max(1e-6, 1 + pop_size - gen)
        lam = np.exp(1 - pop_size / denom)
        F_adaptive = F * (2 ** lam)

        for i in range(pop_size):
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]

            mutant = a + F_adaptive * (b - c)
            low_bounds  = np.array([b_[0] for b_ in bounds])
            high_bounds = np.array([b_[1] for b_ in bounds])
            mutant = np.clip(mutant, low_bounds, high_bounds)

            cross_points = np.random.rand(dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            trial = np.where(cross_points, mutant, pop[i])

            trial_fitness = fitness_func(trial)
            if trial_fitness < fitness_values[i]:
                pop[i] = trial.copy()
                fitness_values[i] = trial_fitness
                if trial_fitness < best_fitness:
                    best = trial.copy()
                    best_fitness = trial_fitness

        if best_fitness <= tol:
            break

    # print(f"Gen {gen}, Best Fitness: {best_fitness:.4e}")
    return best


# ------------------------------ (2) 估计回波数目 & 参数 的函数 ------------------------------

def estimate_echo_count_and_params(X, envelope_true,
                                   echo_range=(2, 6),
                                   ade_kwargs=None):
    """
    在回波个数范围 echo_range 内循环，用 ADE 分别拟合每个 N，
    最后挑选出最优的 (N, 参数向量, 对应的误差值)。

    参数:
      X: 时间向量 (秒) (numpy 数组)
      envelope_true: 真实信号包络 (numpy 数组，同 X 长度)
      echo_range: (min_echoes, max_echoes)，例如 (2, 6)，表示尝试 N=2 到 N=6
      ade_kwargs: 传给 ade_optimize 的字典，比如 {'pop_size':50, 'max_gen':200}

    返回:
      best_overall: dict，包含以下字段
        {
          'best_fitness': float,     # 最小的 SSE
          'best_N': int,             # 最优回波数目
          'best_params': np.ndarray   # 最优参数向量，对应长度 = best_N + 5
        }
    """
    if ade_kwargs is None:
        ade_kwargs = {'pop_size': 30, 'max_gen': 200, 'F': 0.5, 'CR': 0.9, 'tol': 1e-8}

    best_overall = {
        'best_fitness': np.inf,
        'best_N': None,
        'best_params': None
    }

    for N in range(echo_range[0], echo_range[1] + 1):
        # 1. 构造 bounds —— 前 N 项是 β 的范围，后面依次是 τ₁、Δ、f_c、φ、ω
        bounds = []
        # β_i ∈ [-1.0, -0.5]
        for _ in range(N):
            bounds.append((-1.0e-6, -0.5e-6))
        # τ₁ ∈ [0.1e-6, 0.4e-6]
        bounds.append((0.1e-6, 0.4e-6))
        # Δ ∈ [1.3e-6, 4e-6]
        bounds.append((1.3e-6, 4.0e-6))
        # f_c ∈ [4.2e6, 5.8e6]
        bounds.append((4.2e6, 5.8e6))
        # φ ∈ [0, 2π]
        bounds.append((0.0, 2*np.pi))
        # ω ∈ [0.2e-6, 1.0e-6]
        bounds.append((0.2e-6, 1.0e-6))

        # 2. 定义当前 N 下的 fitness 包装函数
        def fitness_for_N(param_vec):
            return fitness_nakagami_simplified(param_vec, X, envelope_true)

        # 3. 用 ADE 优化这 N+5 维问题
        print(f"\n>>> 尝试 N = {N} 个回波，维度 dim = {len(bounds)}")
        best_params_N = ade_optimize(bounds, fitness_for_N, **ade_kwargs)


        # 这里加的是LM算法
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])
        result_lm = least_squares(
            residuals_nakagami_lm,
            x0=best_params_N,
            bounds=(lower_bounds, upper_bounds),
            args=(X, envelope_true, N),
            method='trf'  # LM-like optimizer
        )
        best_params_N = result_lm.x.copy()
        best_fitness_N = np.sum(result_lm.fun ** 2)
        # 4. 计算当前最优参数的误差
        # best_fitness_N = fitness_for_N(best_params_N)
        print(f"    N = {N} 最优 SSE = {best_fitness_N:.4e}")

        #这里加的是LM算法




        # 5. 更新全局最优
        if best_fitness_N < best_overall['best_fitness']:
            best_overall['best_fitness'] = best_fitness_N
            best_overall['best_N'] = N
            best_overall['best_params'] = best_params_N.copy()

    return best_overall

# ------------------------------ (2) 真数据处理与误差计算 ------------------------------

# # 请将下列路径修改为你的真实数据所在的绝对路径
# target_dir = r"D:\PyCharm\Project\swin_transformer_paper\compare_method_testdata\target"
# label_dir  = r"D:\PyCharm\Project\swin_transformer_paper\compare_method_testdata\label"



# ##5dB
# target_dir   = r"D:\PyCharm\Project\swin_transformer_paper\5dB__compare_method\target_data"
# label_dir = r"D:\PyCharm\Project\swin_transformer_paper\5dB__compare_method\label"
#
#
if __name__ == "__main__":

    # # ##10dB
    target_dir   = r"D:\AA_CodeText\Jiang_program\swin_transformer_paper\swin_transformer_paper\data_set\25dB__compare_method\target_data"
    label_dir = r"D:\AA_CodeText\Jiang_program\swin_transformer_paper\swin_transformer_paper\data_set\25dB__compare_method\label"
    db_num = 25
    print(f"========New Address: {target_dir},DB={db_num}==========")

    fs = 500e6  # 采样率 500 MHz
    predicted_tofs = []
    true_tofs     = []
    errors        = []

    # 逐个文件读取并排序
    target_files = sorted(os.listdir(target_dir), key=lambda fn: int(os.path.splitext(fn)[0]))
    label_files  = sorted(os.listdir(label_dir), key=lambda fn: int(os.path.splitext(fn)[0]))

    for tfile, lfile in zip(target_files, label_files):

        # 1) 读取原始信号
        signal = pd.read_csv(os.path.join(target_dir, tfile), header=None).iloc[:,0].dropna().values

        # 2) 读取真实 TOF（单位 µs）
        true_tof = float(pd.read_csv(os.path.join(label_dir, lfile), header=None).iloc[0,0])

        # 3) 计算信号包络
        X = np.arange(len(signal)) / fs
        envelope_true = np.abs(hilbert(signal))

        # 4) 用 ADE 拟合最优回波个数及参数
        result = estimate_echo_count_and_params(
            X, envelope_true, echo_range=(2,6),
            ade_kwargs={'pop_size':20, 'max_gen':50}
        )

        N_opt      = result['best_N']
        params_opt = result['best_params']

        # 拆解 delta
        delta_opt = params_opt[N_opt + 1]  # 索引：N_opt->tau1, N_opt+1->delta

        # 预测 TOF (µs)
        predicted_tof = delta_opt * 1e6
        predicted_tofs.append(predicted_tof)
        true_tofs.append(true_tof)
        errors.append(predicted_tof - true_tof)

       # 将结果转为 numpy 数组
       #  predicted_arr = np.array([p[0] if isinstance(p, list) and p else np.nan for p in predicted_tofs])
        predicted_arr = np.array(predicted_tofs)
        true_arr = np.array(true_tofs)
        errors_arr = np.array(errors)

        # 去除 NaN（有些信号预测失败）
        mask = ~np.isnan(predicted_arr) & ~np.isnan(true_arr)
        predicted_clean = predicted_arr[mask]
        true_clean = true_arr[mask]
        errors_clean = errors_arr[mask]

        # MAE / MSE / R²
        mae = mean_absolute_error(true_clean, predicted_clean)
        mse = mean_squared_error(true_clean, predicted_clean)
        r2 = r2_score(true_clean, predicted_clean)

        # MARE 和最大相对误差（排除 true==0）
        nonzero_mask = true_clean != 0
        rel_errors = np.abs((predicted_clean[nonzero_mask] - true_clean[nonzero_mask]) / true_clean[nonzero_mask])
        mare = np.mean(rel_errors)
        max_rel = np.max(rel_errors)

        # 打印结果
        print(f"有效样本数:         {len(true_clean)}")
        print("===== 误差指标 =====")
        print(f"MAE:               {mae:.4f} μs")
        print(f"MSE:               {mse:.4f} μs²")
        print(f"R²:                {r2:.4f}")
        print(f"平均相对误差 MARE:  {mare:.4f}")
        print(f"最大相对误差:      {max_rel:.4f}")



    np.savetxt(f'Resolut/{db_num}dB_ADE_LM.txt', errors)


    errors = np.loadtxt(f'Resolut/{db_num}dB_ADE_LM.txt')
    ###画柱状图
    # 假设 errors 是你的误差列表（注意：errors中最好是纯数值，没有nan）
    # 先把errors中的nan去掉或者过滤
    errors_clean = [e for e in errors if not np.isnan(e)]

    # 设定区间宽度
    bin_width = 0.02  # μs

    # 计算最大最小误差
    min_error = min(errors_clean)
    max_error = max(errors_clean)

    # 计算对称的最大绝对值作为边界
    max_abs_error = max(abs(min(errors_clean)), abs(max(errors_clean)))
    # 创建对称 bin 边界（包含0），比如从 -1.0 到 +1.0，步长为0.2
    bins = np.arange(-max_abs_error, max_abs_error + bin_width, bin_width)





    # # 生成分段的边界
    # bins = np.arange(min_error, max_error + bin_width, bin_width)

    # 计算每个区间的样本数
    hist, bin_edges = np.histogram(errors_clean, bins=bins)

    # 绘图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(bin_edges[:-1], hist, width=bin_width, align='edge', edgecolor='black')
    plt.xlabel('Δt(μs)')
    plt.ylabel('Sample Count')
    # plt.title('Distribution of Absolute Prediction Errors')
    # 设置刻度线宽度
    plt.tick_params(width=1)
    # plt.xlabel('绝对误差区间 (μs)')
    # plt.ylabel('样本数量')
    # plt.title('预测绝对误差分布柱状图')
    # 横坐标刻度保留两位小数并旋转

    # ✅ 添加数值标签
    for bar in bars:
        height = bar.get_height()
        if height > 0:  # 只标非0柱子
            plt.text(bar.get_x() + bar.get_width() / 2,
                     height,
                     f'{int(height)}',  # 如果你希望是整数，可以用 int(height)
                     ha='center',
                     va='bottom',
                     fontsize=12,
                     fontname='Times New Roman')

    xtick_pos = bin_edges  # 所有边界，包括最后一个
    xtick_labels = [f'{x:.2f}' for x in xtick_pos]
    plt.xticks(xtick_pos, labels=xtick_labels, fontsize=16, fontname='Times New Roman')
    # plt.xticks(bin_edges[::max(1, len(bin_edges) // 15)], rotation=0)  # 避免x轴过密
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # 保存图像为高分辨率文件
    plt.savefig(f"Resolut/{db_num}dB_ADEandLM.jpg", dpi=600, bbox_inches='tight')
    plt.show()