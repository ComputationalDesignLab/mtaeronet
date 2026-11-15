import os
import pickle
import argparse
import torch
import torch.nn as nn
import numpy as np
import time
from scipy.io import loadmat, savemat
from src.models.mlp import MLPHyperparameters, train_net, MLP
from src.models.mha_net import MHANet, train_mha_net
from src.models.mt_aero_net import MTAeroNet, train_mt_net
from src.models.mh_aero_net import MHAeroNet, train_mh_net
from src.utils.standardize import Standardize
from src.utils.evaluate_model import calculate_nrmse
from src.utils.set_seed import set_seed

kwargs = {"dtype": torch.float32, "device": "cuda" if torch.cuda.is_available() else "cpu"}

# parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument("--samples", help="number of samples")
parser.add_argument("--method", help="modeling method to use")
args = parser.parse_args()

# constants
NUM_SAMPLES = int(args.samples)
METHOD = args.method
N = int(NUM_SAMPLES/50)
TYPE = "balanced"

# import training data
X = []
S = []
Y = []
Z = []
for i in range(N):
    train_data = loadmat(f"./data/journal_CRM_{50*(i+1)}_data.mat")
    _x = train_data["x"]
    _s = train_data["coeff"]
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

# import testing data
test_data = loadmat("./data/journal_CRM_50_test_data.mat")
test_x = torch.from_numpy(test_data["x"]).to(**kwargs)
test_s = torch.from_numpy(test_data["coeff"]).to(**kwargs)
test_y = torch.from_numpy(test_data["coefPressure"]).to(**kwargs)
test_z = torch.from_numpy(test_data["skinFriction"]).to(**kwargs)

# activation choices
activation_dict = {
        1: nn.SiLU(),
        2: nn.Tanh(),
        3: nn.ReLU(),
        4: nn.LeakyReLU()
    }

if METHOD=="coeff":

    hyp_error = torch.zeros(5,10,3)
    hyp_preds = torch.zeros(5,10,test_x.shape[0],3)
    train_times = np.zeros((5,10,1))
    test_times = np.zeros((5,10,1))

    for j in range(5):
        # get the hyperparameters
        with open(f"./saved_models/{METHOD}/wing/hyps/{j+1}/hyp_{NUM_SAMPLES}.pkl", "rb") as file:
            hyp_dict = pickle.load(file)
        values = hyp_dict["vals"]
        index = np.argmin(values)
        params = hyp_dict["params"][index]

        hidden_layers = [params["n_neurons"]] * params["n_layers"]
        lr = 1e-3
        epochs = 3000
        activation = activation_dict[params["activation"]]

        # coefficient only network
        hyps = MLPHyperparameters(hidden_layers=hidden_layers, input_dim=x_std.shape[-1],
                                                    output_dim=s_std.shape[-1], activation=activation)
        for i in range(10):
            set_seed(i)
            coeff_network = MLP(hyps).to(**kwargs)
            t1 = time.perf_counter()
            coeff_network = train_net(coeff_network, x_std, s_std, epochs=epochs, learning_rate=lr)
            t2 = time.perf_counter()    

            # predicting using trained network
            t3 = time.perf_counter()
            coeff_network_preds = coeff_network(x_transform.transform(test_x))
            t4 = time.perf_counter()
            coeff_network_preds = s_transform.inverse_transform(coeff_network_preds)
            coeff_nrmse = calculate_nrmse(test_s, coeff_network_preds)        

            # printing results
            print(f"##### Printing results for {NUM_SAMPLES}, hyperparameters {j+1} and iteration {i+1} #####")
            print(f"NRMSE: {coeff_nrmse}")

            # saving the model
            os.makedirs(f"./saved_models/coeff/wing/models/{j+1}", exist_ok=True)
            torch.save(coeff_network.state_dict(), f"./saved_models/coeff/wing/models/{j+1}/coeff_model_{j+1}_{NUM_SAMPLES}_{i+1}.pth")
            
            hyp_error[j,i,:] = coeff_nrmse
            hyp_preds[j,i,:,:] = coeff_network_preds
            train_times[j,i,:] = t2-t1
            test_times[j,i,:] = t4-t3
        
    # saving the results
    results = {"nrmse": hyp_error.detach().cpu().numpy(), "preds": hyp_preds.detach().cpu().numpy(),
                "train_times": train_times, "test_times": test_times}
    os.makedirs("./results/coeff/wing/", exist_ok=True)
    savemat(f"./results/coeff/wing/coeff_results_{NUM_SAMPLES}.mat", results)

elif METHOD=="pressure":

    hyp_error = torch.zeros(5,10,train_y.shape[-1])
    hyp_preds = torch.zeros(5,10,test_x.shape[0],train_y.shape[-1])
    train_times = np.zeros((5,10,1))
    test_times = np.zeros((5,10,1))

    for j in range(5):

        # get the hyperparameters
        with open(f"./saved_models/{METHOD}/wing/hyps/{j+1}/hyp_{NUM_SAMPLES}.pkl", "rb") as file:
            hyp_dict = pickle.load(file)
        values = hyp_dict["vals"]
        index = np.argmin(values)
        params = hyp_dict["params"][index]

        first_layer = params["n_neuro_first"]
        shrinkage = params["shrinkage"]
        n_layers = params["num_layers"]
        layers = []
        layer = first_layer
        for _ in range(n_layers):
            layers.append(layer)
            layer = int(layer*shrinkage)
        hidden_layers = layers[::-1]
        lr = 1e-3
        epochs = 3000
        activation = activation_dict[params["activation"]]

        hyps = MLPHyperparameters(hidden_layers=hidden_layers, input_dim=x_std.shape[-1],
                                            output_dim=y_std.shape[-1], activation=activation)

        for i in range(10):
            set_seed(i)
            cp_network = MLP(hyps).to(**kwargs)
            t1 = time.perf_counter()
            cp_network = train_net(cp_network, x_std, y_std, epochs=epochs, learning_rate=lr)
            t2 = time.perf_counter()

            # predicting using trained network
            t3 = time.perf_counter()
            cp_network_preds = cp_network(x_transform.transform(test_x))
            t4 = time.perf_counter()
            cp_network_preds = y_transform.inverse_transform(cp_network_preds)
        
            # evaluating cp network
            cp_nrmse = calculate_nrmse(test_y, cp_network_preds)       

            # printing results
            print(f"##### Printing results for {NUM_SAMPLES} and iteration {i+1} #####")
            print(f"NRMSE: {np.mean(cp_nrmse.detach().cpu().numpy())}")

            # saving the model
            os.makedirs(f"./saved_models/pressure/wing/models/{j+1}", exist_ok=True)
            torch.save(cp_network.state_dict(), f"./saved_models/pressure/wing/models/{j+1}/pressure_model_{j+1}_{NUM_SAMPLES}_{i+1}.pth")
            
            hyp_error[j,i,:] = cp_nrmse
            hyp_preds[j,i,:,:] = cp_network_preds
            train_times[j,i,:] = t2-t1
            test_times[j,i,:] = t4-t3

    # saving the results
    results = {"nrmse": hyp_error.detach().cpu().numpy(), "preds": hyp_preds.detach().cpu().numpy(),
                "train_times": train_times, "test_times": test_times}
    os.makedirs("./results/pressure/wing/", exist_ok=True)
    savemat(f"./results/pressure/wing/pressure_results_{NUM_SAMPLES}.mat", results)

elif METHOD=="friction":

    hyp_error = torch.zeros(5,10,train_y.shape[-1])
    hyp_preds = torch.zeros(5,10,test_x.shape[0],train_y.shape[-1])
    train_times = np.zeros((5,10,1))
    test_times = np.zeros((5,10,1))

    for j in range(5):

        # get the hyperparameters
        with open(f"./saved_models/{METHOD}/wing/hyps/{j+1}/hyp_{NUM_SAMPLES}.pkl", "rb") as file:
            hyp_dict = pickle.load(file)
        values = hyp_dict["vals"]
        index = np.argmin(values)
        params = hyp_dict["params"][index]

        first_layer = params["n_neuro_first"]
        shrinkage = params["shrinkage"]
        n_layers = params["num_layers"]
        layers = []
        layer = first_layer
        for _ in range(n_layers):
            layers.append(layer)
            layer = int(layer*shrinkage)
        hidden_layers = layers[::-1]
        lr = 1e-3
        epochs = 3000
        activation = activation_dict[params["activation"]]

        hyps = MLPHyperparameters(hidden_layers=hidden_layers, input_dim=x_std.shape[-1],
                                        output_dim=z_std.shape[-1], activation=activation)

        for i in range(10):
            set_seed(i)
            cf_network = MLP(hyps).to(**kwargs)
            t1 = time.perf_counter()
            cf_network = train_net(cf_network, x_std, z_std, epochs=epochs, learning_rate=lr)
            t2 = time.perf_counter()

            # predicting using trained network
            t3 = time.perf_counter()
            cf_network_preds = cf_network(x_transform.transform(test_x))
            t4 = time.perf_counter()
            cf_network_preds = z_transform.inverse_transform(cf_network_preds)

            # evaluating cp network
            cf_nrmse = calculate_nrmse(test_z, cf_network_preds)

            # printing results
            print(f"##### Printing results for {NUM_SAMPLES} and iteration {i+1} #####")
            print(f"NRMSE: {np.mean(cf_nrmse.detach().cpu().numpy())}")

            # saving the model
            os.makedirs(f"./saved_models/friction/wing/models/{j+1}", exist_ok=True)
            torch.save(cf_network.state_dict(), f"./saved_models/friction/wing/models/friction_model_{j+1}_{NUM_SAMPLES}_{i+1}.pth")
            
            hyp_error[j,i,:] = cf_nrmse
            hyp_preds[j,i,:,:] = cf_network_preds
            train_times[j,i,:] = t2-t1
            test_times[j,i,:] = t4-t3

    # saving the results
    results = {"nrmse": hyp_error.detach().cpu().numpy(), "preds": hyp_preds.detach().cpu().numpy(),
                "train_times": train_times, "test_times": test_times}
    os.makedirs("./results/friction/wing/", exist_ok=True)
    savemat(f"./results/friction/wing/friction_results_{NUM_SAMPLES}.mat", results)

elif METHOD == "mha-pressure":

    vec_error = torch.zeros(5,10,train_y.shape[-1])
    vec_preds = torch.zeros(5,10,test_x.shape[0],train_y.shape[-1])
    sca_error = torch.zeros(5,10,3)
    sca_preds = torch.zeros(5,10,test_x.shape[0],3)
    train_times = np.zeros((5,10,1))
    test_times = np.zeros((5,10,1))

    for j in range(5):
        # get the hyperparameters
        with open(f"./saved_models/{METHOD}/wing/hyps/{j+1}/hyp_{NUM_SAMPLES}.pkl", "rb") as file:
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

        # mha network
        hidden_layers = [params["n_neurons"]] * params["n_layers"]
        hidden_layers_1 = [params["n_neurons_1"]] * params["n_layers_1"]
        first_layer = params["n_neuro_first"]
        shrinkage = params["shrinkage"]
        n_layers = params["num_layers"]
        layers = []
        layer = first_layer
        for _ in range(n_layers):
            layers.append(layer)
            layer = int(layer*shrinkage)
        encoder_layers = layers
        decoder_layers = layers[::-1]
        latent_dim = params["latent"]
        activation = activation_dict[params["activation"]]
        lr = 1e-3
        epochs = 3000

        encoder_hyps1 = MLPHyperparameters(hidden_layers=encoder_layers, input_dim=y_std.shape[-1], 
                                                    output_dim=latent_dim, activation=activation)
        latent_hyps1 = MLPHyperparameters(hidden_layers=hidden_layers, input_dim=x_std.shape[-1],
                                                    output_dim=latent_dim, activation=activation)
        decoder_hyps1 = MLPHyperparameters(hidden_layers=decoder_layers, input_dim=latent_dim,
                                                    output_dim=y_std.shape[-1], activation=activation)
        scalar_hyps1 = MLPHyperparameters(hidden_layers=hidden_layers_1, input_dim=latent_dim,
                                                        output_dim=s_std.shape[-1], activation=activation)

        for i in range(10):
            set_seed(i)
            mha_net = MHANet(encoder_hyps1, latent_hyps1, decoder_hyps1, scalar_hyps1).to(**kwargs)
            t1 = time.perf_counter()
            mha_net = train_mha_net(mha_net, x_std, y_std, s_std, epochs=epochs, learning_rate=lr)
            t2 = time.perf_counter()

            # predicting using trained network
            t3 = time.perf_counter()
            _, sca, cp_preds = mha_net(x_transform.transform(test_x))
            t4 = time.perf_counter()
            cp_preds = y_transform.inverse_transform(cp_preds)
            sca = s_transform.inverse_transform(sca)

            # evaluating network
            vec_nrmse = calculate_nrmse(test_y, cp_preds)
            sca_nrmse = calculate_nrmse(test_s, sca)

            # printing results
            print(f"##### Printing results for {NUM_SAMPLES} and iteration {i+1} #####")
            print(f"Vector NRMSE: {np.mean(vec_nrmse.detach().cpu().numpy())}")
            print(f"Scalar NRMSE: {sca_nrmse.detach().cpu().numpy()}")

            # saving the model
            os.makedirs(f"./saved_models/mha-pressure/wing/models/{j+1}", exist_ok=True)
            torch.save(mha_net.state_dict(), f"./saved_models/mha-pressure/wing/models/{j+1}/mha_model_{j+1}_{NUM_SAMPLES}_{i+1}_{TYPE}.pth")

            vec_error[j,i,:] = vec_nrmse
            vec_preds[j,i,:,:] = cp_preds
            sca_error[j,i,:] = sca_nrmse
            sca_preds[j,i,:,:] = sca
            train_times[j,i,:] = t2-t1
            test_times[j,i,:] = t4-t3

    # saving the results
    results = {"vec_nrmse": vec_error.detach().cpu().numpy(), "sca_nrmse": sca_error.detach().cpu().numpy(),
                "vec_preds": vec_preds.detach().cpu().numpy(), "sca_preds": sca_preds.detach().cpu().numpy(),
                "train_times": train_times, "test_times": test_times}
    os.makedirs("./results/mha-pressure/wing/", exist_ok=True)
    savemat(f"./results/mha-pressure/wing/mha_results_{NUM_SAMPLES}_{TYPE}.mat", results)

elif METHOD == "mha-friction":

    vec_error = torch.zeros(5,10,train_z.shape[-1])
    vec_preds = torch.zeros(5,10,test_x.shape[0],train_z.shape[-1])
    sca_error = torch.zeros(5,10,3)
    sca_preds = torch.zeros(5,10,test_x.shape[0],3)
    train_times = np.zeros((5,10,1))
    test_times = np.zeros((5,10,1))

    for j in range(5):
        # get the hyperparameters
        with open(f"./saved_models/{METHOD}/wing/hyps/{j+1}/hyp_{NUM_SAMPLES}.pkl", "rb") as file:
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

        # mha network
        hidden_layers = [params["n_neurons"]] * params["n_layers"]
        hidden_layers_1 = [params["n_neurons_1"]] * params["n_layers_1"]
        first_layer = params["n_neuro_first"]
        shrinkage = params["shrinkage"]
        n_layers = params["num_layers"]
        layers = []
        layer = first_layer
        for _ in range(n_layers):
            layers.append(layer)
            layer = int(layer*shrinkage)
        encoder_layers = layers
        decoder_layers = layers[::-1]
        latent_dim = params["latent"]
        activation = activation_dict[params["activation"]]
        lr = 1e-3
        epochs = 3000

        encoder_hyps1 = MLPHyperparameters(hidden_layers=encoder_layers, input_dim=y_std.shape[-1], 
                                                    output_dim=latent_dim, activation=activation)
        latent_hyps1 = MLPHyperparameters(hidden_layers=hidden_layers, input_dim=x_std.shape[-1],
                                                    output_dim=latent_dim, activation=activation)
        decoder_hyps1 = MLPHyperparameters(hidden_layers=decoder_layers, input_dim=latent_dim,
                                                    output_dim=y_std.shape[-1], activation=activation)
        scalar_hyps1 = MLPHyperparameters(hidden_layers=hidden_layers_1, input_dim=latent_dim,
                                                        output_dim=s_std.shape[-1], activation=activation)

        for i in range(10):
            set_seed(i)
            mha_net = MHANet(encoder_hyps1, latent_hyps1, decoder_hyps1, scalar_hyps1).to(**kwargs)
            t1 = time.perf_counter()
            mha_net = train_mha_net(mha_net, x_std, z_std, s_std, epochs=epochs, learning_rate=lr)
            t2 = time.perf_counter()

            # predicting using trained network
            t3 = time.perf_counter()
            _, sca, cf_preds = mha_net(x_transform.transform(test_x))
            t4 = time.perf_counter()
            cf_preds = z_transform.inverse_transform(cf_preds)
            sca = s_transform.inverse_transform(sca)

            # evaluating network
            vec_nrmse = calculate_nrmse(test_z, cf_preds)
            sca_nrmse = calculate_nrmse(test_s, sca)

            # printing results
            print(f"##### Printing results for {NUM_SAMPLES} and iteration {i+1} #####")
            print(f"Vector NRMSE: {np.mean(vec_nrmse.detach().cpu().numpy())}")
            print(f"Scalar NRMSE: {sca_nrmse.detach().cpu().numpy()}")

            # saving the model
            os.makedirs(f"./saved_models/mha-friction/wing/models/{j+1}", exist_ok=True)
            torch.save(mha_net.state_dict(), f"./saved_models/mha-friction/wing/models/{j+1}/mha_model_{j+1}_{NUM_SAMPLES}_{i+1}_{TYPE}.pth")

            vec_error[j,i,:] = vec_nrmse
            vec_preds[j,i,:,:] = cf_preds
            sca_error[j,i,:] = sca_nrmse
            sca_preds[j,i,:,:] = sca
            train_times[j,i,:] = t2-t1
            test_times[j,i,:] = t4-t3

    # saving the results
    results = {"vec_nrmse": vec_error.detach().cpu().numpy(), "sca_nrmse": sca_error.detach().cpu().numpy(),
                "vec_preds": vec_preds.detach().cpu().numpy(), "sca_preds": sca_preds.detach().cpu().numpy(),
                "train_times": train_times, "test_times": test_times}
    os.makedirs("./results/mha-friction/wing/", exist_ok=True)
    savemat(f"./results/mha-friction/wing/mha_results_{NUM_SAMPLES}_{TYPE}.mat", results)

elif METHOD == "mtaero":
    
    vec_error = torch.zeros(5,10,train_y.shape[-1])
    vec_preds = torch.zeros(5,10,test_x.shape[0],train_y.shape[-1])
    sca_error = torch.zeros(5,10,3)
    sca_preds = torch.zeros(5,10,test_x.shape[0],3)
    train_times = np.zeros((5,10,1))
    test_times = np.zeros((5,10,1))

    for j in range(5):

        with open(f"./saved_models/{METHOD}/wing/hyps/{j+1}/hyp_{NUM_SAMPLES}.pkl", "rb") as file:
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
        first_layer = params["n_neuro_first"]
        shrinkage = params["shrinkage"]
        n_layers = params["num_layers"]
        layers = []
        layer = first_layer
        for _ in range(n_layers):
            layers.append(layer)
            layer = int(layer*shrinkage)
        encoder_layers = layers
        decoder_layers = layers[::-1]
        latent_dim = params["latent"]
        lr = 1e-3
        epochs = 3000
        activation = activation_dict[params["activation"]]

        cp_decoder_hyps = MLPHyperparameters(hidden_layers=decoder_layers, input_dim=latent_dim,
                                                    output_dim=y_std.shape[-1], activation=activation)
        scalar_hyps = MLPHyperparameters(hidden_layers=hidden_layers, input_dim=x_std.shape[-1],
                                            output_dim=s_std.shape[-1] + latent_dim, activation=activation)

        for i in range(10):
            set_seed(i)
            aero_net = MTAeroNet(cp_decoder_hyps, scalar_hyps, s_std.shape[-1]).to(**kwargs)
            t1 = time.perf_counter()
            aero_net = train_mt_net(aero_net, x_std, y_std, s_std, epochs=epochs, learning_rate=lr)
            t2 = time.perf_counter()

            # predicting using trained network
            t3 = time.perf_counter()
            cp_preds, sca = aero_net(x_transform.transform(test_x))
            t4 = time.perf_counter()
            cp_preds = y_transform.inverse_transform(cp_preds)
            sca = s_transform.inverse_transform(sca)

            # evaluating network
            cp_nrmse = calculate_nrmse(test_y, cp_preds)
            sca_nrmse = calculate_nrmse(test_s, sca)

            # printing results
            print(f"##### Printing results for {NUM_SAMPLES} #####")
            print(f"Vector NRMSE: {np.mean(cp_nrmse.detach().cpu().numpy())}")
            print(f"Scalar NRMSE: {sca_nrmse.detach().cpu().numpy()}")

            # saving the model
            os.makedirs(f"./saved_models/mtaero/wing/models/{j+1}", exist_ok=True)
            torch.save(aero_net.state_dict(), f"./saved_models/mtaero/wing/models/{j+1}/mtaero_model_{j+1}_{NUM_SAMPLES}_{i+1}_{TYPE}.pth")

            vec_error[j,i,:] = cp_nrmse
            vec_preds[j,i,:,:] = cp_preds
            sca_error[j,i,:] = sca_nrmse
            sca_preds[j,i,:,:] = sca
            train_times[j,i,:] = t2-t1
            test_times[j,i,:] = t4-t3

    # saving the results
    results = {"vec_nrmse": vec_error.detach().cpu().numpy(), "sca_nrmse": sca_error.detach().cpu().numpy(),
                "vec_preds": vec_preds.detach().cpu().numpy(), "sca_preds": sca_preds.detach().cpu().numpy(),
                "train_times": train_times, "test_times": test_times}
    os.makedirs("./results/mtaero/wing/", exist_ok=True)
    savemat(f"./results/mtaero/wing/mtaero_results_{NUM_SAMPLES}_{TYPE}.mat", results)

elif METHOD == "mhaero":

    cp_error = torch.zeros(5,10,train_y.shape[-1])
    cp_preds = torch.zeros(5,10,test_x.shape[0], train_y.shape[-1])
    cf_error = torch.zeros(5,10,train_z.shape[-1])
    cf_preds = torch.zeros(5,10,test_x.shape[0], train_z.shape[-1])
    sca_error = torch.zeros(5,10,3)
    sca_preds = torch.zeros(5,10,test_x.shape[0],3)
    train_times = np.zeros((5,10,1))
    test_times = np.zeros((5,10,1))

    for j in range(5):
        with open(f"./saved_models/{METHOD}/wing/hyps/{j+1}/hyp_{NUM_SAMPLES}.pkl", "rb") as file:
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

        for i in range(10):
            set_seed(i)
            aero_net = MHAeroNet(cp_decoder_hyps, scalar_hyps, cf_decoder_hyps, s_std.shape[-1]).to(**kwargs)
            t1 = time.perf_counter()
            aero_net = train_mh_net(aero_net, x_std, y_std, z_std, s_std, epochs=epochs, learning_rate=lr)
            t2 = time.perf_counter()

            # predicting using trained network
            t3 = time.perf_counter()
            cp, cf, sca = aero_net(x_transform.transform(test_x))
            t4 = time.perf_counter()
            cp = y_transform.inverse_transform(cp)
            cf = z_transform.inverse_transform(cf)
            sca = s_transform.inverse_transform(sca)
        
            # evaluating network
            cp_nrmse = calculate_nrmse(test_y, cp)
            cf_nrmse = calculate_nrmse(test_z, cf)
            sca_nrmse = calculate_nrmse(test_s, sca)

            # printing results
            print(f"##### Printing results for {NUM_SAMPLES} #####")
            print(f"CP NRMSE: {np.mean(cp_nrmse.detach().cpu().numpy())}")
            print(f"CF NRMSE: {np.mean(cf_nrmse.detach().cpu().numpy())}")
            print(f"Scalar NRMSE: {sca_nrmse.detach().cpu().numpy()}")

            # saving the model
            os.makedirs(f"./saved_models/mhaero/wing/models/{j+1}", exist_ok=True)
            torch.save(aero_net.state_dict(), f"./saved_models/mhaero/wing/models/{j+1}/mhaero_model_{j+1}_{NUM_SAMPLES}_{i+1}_{TYPE}.pth")

            cp_error[j,i,:] = cp_nrmse
            cp_preds[j,i,:,:] = cp
            cf_error[j,i,:] = cf_nrmse
            cf_preds[j,i,:,:] = cf
            sca_error[j,i,:] = sca_nrmse
            sca_preds[j,i,:,:] = sca
            train_times[j,i,:] = t2-t1
            test_times[j,i,:] = t4-t3

    # saving the results
    results = {"cp_nrmse": cp_error.detach().cpu().numpy(), "sca_nrmse": sca_error.detach().cpu().numpy(), "cf_nrmse": cf_error.detach().cpu().numpy(),
                "cp_preds": cp_preds.detach().cpu().numpy(), "sca_preds": sca_preds.detach().cpu().numpy(), "cf_preds": cf_preds.detach().cpu().numpy(),
                "train_times": train_times, "test_times": test_times}
    os.makedirs("./results/mhaero/wing/", exist_ok=True)
    savemat(f"./results/mhaero/wing/mhaero_results_{NUM_SAMPLES}_{TYPE}.mat", results)


