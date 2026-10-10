import torch
import torch.nn as nn
from .mlp import MLP, MLPHyperparameters

def train_mha_net(model, train_x, train_y, train_s, epochs, learning_rate):

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        optimizer.zero_grad()
        encoding = model.encoding(train_y)
        reconstruction = model.decoding(encoding)
        latent_preds = model.latent_pred(train_x)
        scalar_preds = model.scalar_pred(latent_preds)
        loss = loss_function(reconstruction, train_y) + loss_function(encoding, latent_preds) + loss_function(scalar_preds, train_s)
        loss.backward()
        optimizer.step()
    return model

def train_mha_net_weighted(model, train_x, train_y, train_s, epochs, learning_rate, weight):

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        optimizer.zero_grad()
        encoding = model.encoding(train_y)
        reconstruction = model.decoding(encoding)
        latent_preds = model.latent_pred(train_x)
        scalar_preds = model.scalar_pred(latent_preds)
        loss = weight*loss_function(reconstruction, train_y) + (1-weight)*(loss_function(encoding, latent_preds) + loss_function(scalar_preds, train_s))
        loss.backward()
        optimizer.step()
    return model
    
class MHANet(nn.Module):

    def __init__(self, encoder_hyps, latent_hyps, decoder_hyps, scalar_hyps):
        super().__init__()
        self.encoder_hyps = encoder_hyps
        self.decoder_hyps = decoder_hyps
        self.latent_hyps = latent_hyps
        self.scalar_hyps = scalar_hyps

        # defining networks
        self.encoder = MLP(hyperparameters=encoder_hyps)
        self.decoder = MLP(hyperparameters=decoder_hyps)
        self.latent_nn = MLP(hyperparameters=latent_hyps)
        self.scalar_nn = MLP(hyperparameters=scalar_hyps)

    def encoding(self, y):
        return self.encoder(y)

    def decoding(self, z):
        return self.decoder(z)

    def latent_pred(self, x):
        return self.latent_nn(x)

    def scalar_pred(self, x):
        return self.scalar_nn(x)

    def forward(self, x):
        latent_preds = self.latent_nn(x)
        scalar_preds = self.scalar_nn(latent_preds)
        decoded_preds = self.decoder(latent_preds)
        return latent_preds, scalar_preds, decoded_preds