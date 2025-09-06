import torch

def log_cosh_loss(pred, target):
    diff = pred - target
    return torch.mean(torch.log(torch.cosh(diff + 1e-12)))

def smooth_l1_loss(pred, target, beta=1.0):
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * (diff ** 2) / beta, diff - 0.5 * beta)
    return loss.mean()

def aleatoric_loss(mu, log_var, target, min_logvar=-5.0, max_logvar=5.0):
    log_var = torch.clamp(log_var, min=min_logvar, max=max_logvar)
    precision = torch.exp(-log_var)
    loss = precision * (target - mu) ** 2 + log_var
    return loss.mean()
