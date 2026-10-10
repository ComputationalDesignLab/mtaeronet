import torch
import torch.nn as nn
from .mlp import MLP, MLPHyperparameters

def train_mh_net(model, train_x, train_cp, train_cf, train_s, epochs, learning_rate):

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        optimizer.zero_grad()
        cp_preds, cf_preds, scalar_preds = model(train_x)
        loss = loss_function(cp_preds, train_cp) + loss_function(cf_preds, train_cf) + loss_function(scalar_preds, train_s)
        loss.backward()
        optimizer.step()
    return model

def train_mh_net_weighted(model, train_x, train_cp, train_cf, train_s, epochs, learning_rate, weight):

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        optimizer.zero_grad()
        cp_preds, cf_preds, scalar_preds = model(train_x)
        loss = weight*(loss_function(cp_preds, train_cp) + loss_function(cf_preds, train_cf)) + (1-weight)*loss_function(scalar_preds, train_s)
        loss.backward()
        optimizer.step()
    return model

class MHAeroNet(nn.Module):

    def __init__(self, cp_decoder_hyps, scalar_hyps, cf_decoder_hyps, scalar_dim):
        super().__init__()
        self.cp_decoder_hyps = cp_decoder_hyps
        self.cf_decoder_hyps = cf_decoder_hyps
        self.scalar_net_hyps = scalar_hyps
        self.scalar_dim = scalar_dim

        # defining networks
        self.cp_decoder = MLP(hyperparameters=cp_decoder_hyps)
        self.cf_decoder = MLP(hyperparameters=cf_decoder_hyps)
        self.scalar_nn = MLP(hyperparameters=scalar_hyps)

    def cp_decoding(self, z):
        output = self.cp_decoder(z)
        return output

    def cf_decoding(self, z):
        output = self.cf_decoder(z)
        return output

    def scalar_pred(self, z):
        return self.scalar_nn(z)

    def forward(self, x):
        latent_scalar = self.scalar_nn(x)
        latents = latent_scalar[:,:-self.scalar_dim]
        scalar_preds = latent_scalar[:,-self.scalar_dim:]
        cp_preds = self.cp_decoder(latents)
        cf_preds = self.cf_decoder(latents)
        return cp_preds, cf_preds, scalar_preds
