import torch

class Standardize:

    def __init__(self, data):
        self.mean = torch.mean(data, dim=0)
        self.std = torch.std(data, dim=0)

    def transform(self, data):
        return (data - self.mean)/(self.std + 1e-8)

    def inverse_transform(self, scaled_data):
        return self.mean + self.std * scaled_data
