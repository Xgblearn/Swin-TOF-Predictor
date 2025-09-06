import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def generate_signal(fs, duration, tau1, tau2, f0=5e6, sigma=50e-9, SNR_dB=8):
    t = np.arange(0, duration, 1 / fs)
    s1 = np.exp(-((t - tau1) / sigma) ** 2) * np.sin(2 * np.pi * f0 * (t - tau1))
    s2 = 0.7 * np.exp(-((t - tau2) / sigma) ** 2) * np.sin(2 * np.pi * f0 * (t - tau2))
    signal = s1 + s2
    Ps = np.mean(signal ** 2)
    Pn = Ps / (10 ** (SNR_dB / 10))
    noise = np.random.normal(0, np.sqrt(Pn), len(t))
    return t, s1, s2, signal + noise

# 参数设置
fs = 100e6
duration = 10e-6
tau1 = 3e-6
tau2 = 5.2e-6
t, s1, s2, signal_noisy = generate_signal(fs, duration, tau1, tau2)

# 取两个回波信号
win_size = 450
center1 = int(tau1 * fs)
center2 = int(tau2 * fs)
x1 = signal_noisy[center1 - win_size // 2:center1 + win_size // 2]
x2 = signal_noisy[center2 - win_size // 2:center2 + win_size // 2]

# VSS-LMS：整体结构建模
N = len(x1)
w = np.zeros(N)
mu = 1e-4
alpha, gamma = 0.97, 0.01
KP, KI, KD = 0.3, 0.2, 0.05
e = x2 - np.dot(w, x1)
p = KP * np.mean(e**2) + KI * np.sum(e**2) + KD * (e[-1]**2 - e[-2]**2)
p = np.clip(p, 0, 1e6)
mu = alpha * mu + gamma * p
mu = min(mu, 1e-2)
w += mu * e * x1

# 样条拟合
cs = CubicSpline(np.arange(len(w)), w)
t_fine = np.linspace(0, len(w) - 1, 10000)
w_interp = cs(t_fine)
TOF_index = t_fine[np.argmax(w_interp)]
TOF_time = TOF_index / fs
true_TOF = tau2 - tau1

# 可视化
plt.plot(t_fine, w_interp, label='Cubic Spline of w(n)')
plt.axvline(TOF_index, color='r', linestyle='--', label='Estimated TOF')
plt.title("Impulse Response & TOF Estimation")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

print(f"Estimated TOF: {TOF_time*1e6:.2f} µs")
print(f"True TOF     : {true_TOF*1e6:.2f} µs")
