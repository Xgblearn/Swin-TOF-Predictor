import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt


class BayesianNN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=20, output_dim=1, dropout_prob=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2 * output_dim)  # 输出均值和方差的对数
        self.dropout = nn.Dropout(p=dropout_prob)  # 使用Dropout近似贝叶斯推断

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # 训练和测试时均开启Dropout
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        # 输出均值和方差的对数值（确保方差为正）
        mean, logvar = torch.chunk(x, 2, dim=-1)
        return mean, logvar

def heteroscedastic_loss(y_pred, y_true):
    mean, logvar = y_pred
    var = torch.exp(logvar)  # 将logvar转换为方差
    loss = 0.5 * (torch.exp(-logvar) * (y_true - mean)**2 + logvar).mean()
    return loss

# 生成合成数据（y = sin(x) + 噪声）
x = np.linspace(-5, 5, 1000)
y = np.sin(x) + 0.1 * np.random.randn(1000)

# 转换为PyTorch张量
x_tensor = torch.FloatTensor(x).unsqueeze(-1)
y_tensor = torch.FloatTensor(y).unsqueeze(-1)

# 初始化模型和优化器
model = BayesianNN(input_dim=1, hidden_dim=20, output_dim=1, dropout_prob=0.2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练循环
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    mean_pred, logvar_pred = model(x_tensor)
    loss = heteroscedastic_loss((mean_pred, logvar_pred), y_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# def predict(model, x, n_samples=50):
#     model.eval()
#     with torch.no_grad():
#         # 多次采样（蒙特卡洛Dropout）
#         preds = [model(x) for _ in range(n_samples)]
#         means = torch.stack([m for m, _ in preds])
#         logvars = torch.stack([lv for _, lv in preds])
#
#         # 计算均值和方差
#         mean = means.mean(dim=0)
#         var = torch.exp(logvars).mean(dim=0) + means.var(dim=0)
#         return mean.numpy(), var.numpy()


def predict(model, x, n_samples=50):
    model.eval()
    with torch.no_grad():
        preds = [model(x) for _ in range(n_samples)]
        means = torch.stack([m.squeeze() for m, _ in preds])  # 压缩输出
        logvars = torch.stack([lv.squeeze() for _, lv in preds])

        mean = means.mean(dim=0).numpy()  # 形状 (200,)
        var = torch.exp(logvars).mean(dim=0).numpy() + means.var(dim=0).numpy()
        return mean, var




# # 生成测试数据
# x_test = np.linspace(-6, 6, 200)
# x_test_tensor = torch.FloatTensor(x_test).unsqueeze(-1)
#
# # 预测
# mean_pred, var_pred = predict(model, x_test_tensor, n_samples=50)
# std_pred = np.sqrt(var_pred)
# print(mean_pred.shape)  # 预期输出应为 (200,)
# print(std_pred.shape)   # 预期输出应为 (200,)
# # 可视化
# plt.figure(figsize=(10, 6))
# plt.scatter(x, y, alpha=0.3, label="Training Data")
# plt.plot(x_test, mean_pred, 'r-', label="Predicted Mean")
# plt.fill_between(x_test, mean_pred - 2 * std_pred, mean_pred + 2 * std_pred,
#                  color='red', alpha=0.2, label="95% Confidence Interval")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Bayesian Neural Network with MC Dropout")
# plt.legend()
# plt.show()


# 确保x_test为一维
x_test = np.linspace(-6, 6, 200)  # 形状 (200,)
x_test_tensor = torch.FloatTensor(x_test).unsqueeze(-1)
# 预测并压缩维度
mean_pred, var_pred = predict(model, x_test_tensor, n_samples=50)
std_pred = np.sqrt(var_pred)

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.3, label="Training Data")
plt.plot(x_test, mean_pred, 'r-', label="Predicted Mean")
plt.fill_between(x_test,
                 mean_pred - 2 * std_pred,
                 mean_pred + 2 * std_pred,
                 color='red', alpha=0.2, label="95% Confidence Interval")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Bayesian Neural Network with MC Dropout")
plt.legend()
plt.show()