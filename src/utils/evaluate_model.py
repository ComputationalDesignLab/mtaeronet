import torch

def calculate_nrmse(y_true, y_pred):
    _max, _ = torch.max(y_true, dim=0)
    _min, _ = torch.min(y_true, dim=0)
    return torch.sqrt(torch.mean((y_true - y_pred)**2, dim=0))/(_max - _min)

