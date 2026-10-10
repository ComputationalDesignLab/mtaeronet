import os
import pickle
import argparse
import torch
import torch.nn as nn
import numpy as np
import h5py
from scipy.stats import norm
from scipy.io import loadmat, savemat
from src.models.mlp import MLPHyperparameters, train_net, MLP
from src.models.mh_aero_net import MHAeroNet, train_mh_net
from src.utils.standardize import Standardize
from src.qmc.qmc_runner import QMCRunner
from dataclasses import asdict

kwargs = {"dtype": torch.float32, "device": "cuda" if torch.cuda.is_available() else "cpu"}

# parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument("--samples", help="number of samples")
args = parser.parse_args()

# constants
NUM_SAMPLES = int(args.samples)
METHOD = "mhaero"
TYPE = "balanced"
mach_rv = norm(loc=0.80, scale=0.04)
alpha_rv = norm(loc=2.2, scale=0.5)
twist_rv = norm(loc=0.0, scale=0.5)
DISTRIBUTION_LIST = [alpha_rv, mach_rv] + [twist_rv] * 7
N = int(NUM_SAMPLES/50)

# extracting the results to set the seed
data = loadmat(f"./results/mhaero/wing/mhaero_results_{NUM_SAMPLES}_{TYPE}.mat")
cp_nrmse = np.mean(data["cp_nrmse"], axis=-1)
cf_nrmse = np.mean(data["cf_nrmse"], axis=-1)
sca_nrmse = np.mean(data["sca_nrmse"], axis=-1)
tot_nrmse = cp_nrmse + cf_nrmse + sca_nrmse
index = np.argmin(tot_nrmse)
hyp_ind, seed = np.unravel_index(index, tot_nrmse.shape)

# import training data
X = []
S = []
Y = []
Z = []
for i in range(N):
    train_data = loadmat(f"./data/journal_CRM_{50*(i+1)}_data.mat")
    _x = train_data["x"]
    _s = train_data["coeff"]
    print(_s.shape)
    _y = train_data["coefPressure"][:50]
    _z = train_data["skinFriction"][:50]
    X.append(_x)
    S.append(_s)
    Y.append(_y)
    Z.append(_z)

train_x = torch.from_numpy(np.vstack(X)).to(**kwargs)
train_s = torch.from_numpy(np.vstack(S)).to(**kwargs)
train_y = torch.from_numpy(np.vstack(Y)).to(**kwargs)
train_z = torch.from_numpy(np.vstack(Z)).to(**kwargs)

# transformed data
x_transform = Standardize(train_x)
x_std = x_transform.transform(train_x)
y_transform = Standardize(train_y)
y_std = y_transform.transform(train_y)
z_transform = Standardize(train_z)
z_std = z_transform.transform(train_z)
s_transform = Standardize(train_s)
s_std = s_transform.transform(train_s)

# activation choices
activation_dict = {
    1: nn.SiLU(),
    2: nn.Tanh(),
    3: nn.ReLU(),
    4: nn.LeakyReLU()
}

with open(f"./saved_models/{METHOD}/wing/hyps/{hyp_ind+1}/hyp_{NUM_SAMPLES}.pkl", "rb") as file:
    hyp_dict = pickle.load(file)
values = hyp_dict["vals"]
norm_values = (values - np.min(values, axis=0))/(np.max(values, axis=0) - np.min(values, axis=0))
norms = np.linalg.norm(norm_values, axis=1)
if TYPE == "balanced":
    index = np.argmin(norms)
elif TYPE == "vector":
    index = np.argmin(norm_values[:,0])
elif TYPE == "scalar":
    index = np.argmin(norm_values[:,1])
params = hyp_dict["params"][index]

hidden_layers = [params["n_neurons"]] * params["n_layers"]
        
dec_layers = []
for i in range(2):
    first_layer = params[f"n_neuro_first_{i+1}"]
    shrinkage = params[f"shrinkage_{i+1}"]
    n_layers = params[f"num_layers_{i+1}"]
    layers = []
    layer = first_layer
    for _ in range(n_layers):
        layers.append(layer)
        layer = int(layer*shrinkage)
    decoder_layers = layers[::-1]
    dec_layers.append(decoder_layers)

latent_dim = params["latent"]
lr = 1e-3
epochs = 3000
activation = activation_dict[params["activation"]]

cp_decoder_hyps = MLPHyperparameters(hidden_layers=dec_layers[0], input_dim=latent_dim,
                                    output_dim=y_std.shape[-1], activation=activation)
cf_decoder_hyps = MLPHyperparameters(hidden_layers=dec_layers[1], input_dim=latent_dim,
                                    output_dim=z_std.shape[-1], activation=activation)
scalar_hyps = MLPHyperparameters(hidden_layers=hidden_layers, input_dim=x_std.shape[-1],
                                    output_dim=s_std.shape[-1] + latent_dim, activation=activation)

# loading the model
state_dictionary = torch.load(f"./saved_models/mhaero/wing/models/{hyp_ind+1}/mhaero_model_{hyp_ind+1}_{NUM_SAMPLES}_{seed+1}_{TYPE}.pth")
aero_net = MHAeroNet(cp_decoder_hyps, scalar_hyps, cf_decoder_hyps, s_std.shape[-1]).to(**kwargs)
aero_net.load_state_dict(state_dictionary)

# running the uq
alpha_mu = 2.2
alpha_sigma = 0.5
mach_mu = 0.80
mach_sigma = 0.04
l_bounds = [alpha_mu - 3*alpha_sigma, mach_mu - 3*mach_sigma] + [-1.5] * 7
u_bounds = [alpha_mu + 3*alpha_sigma, mach_mu + 3*mach_sigma] + [1.5] * 7
runner = QMCRunner(model=aero_net, dim=x_std.shape[-1], distribution_list=DISTRIBUTION_LIST,
                        l_bounds=l_bounds, u_bounds=u_bounds)
    
for i in range(4, 13):
    cp_results, cf_results, sca_results = runner.run(po2=i, in_transform=x_transform, cp_transform=y_transform,
                            cf_transform=z_transform, sca_transform=s_transform)
    print(sca_results.predictions, sca_results.mean_predictions)
    # saving the results
    results_dict = asdict(sca_results)
    os.makedirs("./results/mhaero/wing/uq/", exist_ok=True)
    with h5py.File(f"./results/mhaero/wing/uq/mhaero_sca_{NUM_SAMPLES}_{TYPE}_{i}.h5", "w") as file:
        for key, var in results_dict.items():
            group = file.create_group(key)
            group.create_dataset(f"{key}", data=var)

    results_dict = asdict(cp_results)
    with h5py.File(f"./results/mhaero/wing/uq/mhaero_cp_{NUM_SAMPLES}_{TYPE}_{i}.h5", "w") as file:
        for key, var in results_dict.items():
            group = file.create_group(key)
            group.create_dataset(f"{key}", data=var)

    results_dict = asdict(cf_results)
    with h5py.File(f"./results/mhaero/wing/uq/mhaero_cf_{NUM_SAMPLES}_{TYPE}_{i}.h5", "w") as file:
        for key, var in results_dict.items():
            group = file.create_group(key)
            group.create_dataset(f"{key}", data=var)

