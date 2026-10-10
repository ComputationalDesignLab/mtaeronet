import torch
import torch.nn as nn
from .mlp import MLP, MLPHyperparameters

def train_mt_net(model, train_x, train_cp, train_s, epochs, learning_rate):

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        optimizer.zero_grad()
        cp_preds, scalar_preds = model(train_x)
        loss = loss_function(cp_preds, train_cp) + loss_function(scalar_preds, train_s)
        loss.backward()
        optimizer.step()
    return model

def train_mt_net_weighted(model, train_x, train_cp, train_s, epochs, learning_rate, weight):

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        optimizer.zero_grad()
        cp_preds, scalar_preds = model(train_x)
        loss = weight*loss_function(cp_preds, train_cp) + (1-weight)*loss_function(scalar_preds, train_s)
        loss.backward()
        optimizer.step()
    return model

class MTAeroNet(nn.Module):

    def __init__(self, cp_decoder_hyps, scalar_hyps, scalar_dim):

        super().__init__()
        self.cp_decoder_hyps = cp_decoder_hyps
        self.scalar_hyps = scalar_hyps
        self.scalar_dim = scalar_dim

        # defining networks
        self.cp_decoder = MLP(hyperparameters=cp_decoder_hyps)
        self.scalar_nn = MLP(hyperparameters=scalar_hyps)

    def cp_decoding(self, z):
        return self.cp_decoder(z)

    def scalar_pred(self, x):
        return self.scalar_nn(x)

    def forward(self, x):
        latent_scalar = self.scalar_nn(x)
        latents = latent_scalar[:,:-self.scalar_dim]
        scalar_preds = latent_scalar[:,-self.scalar_dim:]
        cp_preds = self.cp_decoder(latents)
        return cp_preds, scalar_preds
