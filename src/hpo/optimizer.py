import optuna
import numpy as np
import torch
import time
import torch.nn as nn
from ..models.mlp import MLP, train_net, MLPHyperparameters
from ..models.mha_net import MHANet, train_mha_net, train_mha_net_weighted
from ..models.mt_aero_net import MTAeroNet, train_mt_net, train_mt_net_weighted
from ..models.mh_aero_net import MHAeroNet, train_mh_net, train_mh_net_weighted
from ..utils.standardize import Standardize
from ..utils.evaluate_model import calculate_nrmse
from sklearn.model_selection import train_test_split

kwargs = {"dtype": torch.float32, "device": "cuda" if torch.cuda.is_available() else "cpu"}

class HPOptimizer:

    def __init__(self, x_train, y_train, z_train, s_train, split_ratio, 
                _type, method, seed):

        # splitting data into train and validation split
        train_x, val_x, train_y, val_y, train_z, val_z, train_s, val_s = train_test_split(x_train, y_train, z_train, s_train,
                                                                          test_size=split_ratio, random_state=seed)
        self.train_x = torch.from_numpy(train_x).to(**kwargs)
        self.val_x = torch.from_numpy(val_x).to(**kwargs)
        self.train_y = torch.from_numpy(train_y).to(**kwargs)
        self.val_y = torch.from_numpy(val_y).to(**kwargs)
        self.train_z = torch.from_numpy(train_z).to(**kwargs)
        self.val_z = torch.from_numpy(val_z).to(**kwargs)
        self.train_s = torch.from_numpy(train_s).to(**kwargs)
        self.val_s = torch.from_numpy(val_s).to(**kwargs)
        self.method = method
        self.seed = seed
        self._type = _type
        self.activation_dict = {
            1: nn.SiLU(),
            2: nn.Tanh(),
            3: nn.ReLU(),
            4: nn.LeakyReLU()
        }

    def obj_func(self, trial):

        # transformed data
        x_transform = Standardize(self.train_x)
        x_std = x_transform.transform(self.train_x)
        y_transform = Standardize(self.train_y)
        y_std = y_transform.transform(self.train_y)
        z_transform = Standardize(self.train_z)
        z_std = z_transform.transform(self.train_z)
        s_transform = Standardize(self.train_s)
        s_std = s_transform.transform(self.train_s)

        if self.method == "coeff":
            hidden_layers = [trial.suggest_int("n_neurons", 8, 64)] * trial.suggest_int("n_layers", 2, 8)
            lr = 1e-3
            epochs = 3000
            activation = self.activation_dict[trial.suggest_int("activation", 1, 4)]

            hyps = MLPHyperparameters(hidden_layers=hidden_layers, input_dim=x_std.shape[-1],
                                                output_dim=s_std.shape[-1], activation=activation)
            coeff_network = MLP(hyps).to(**kwargs)
            coeff_network = train_net(coeff_network, x_std, s_std, epochs=epochs, learning_rate=lr)
        
            # predicting using trained network
            coeff_network_preds = coeff_network(x_transform.transform(self.val_x))
            coeff_network_preds = s_transform.inverse_transform(coeff_network_preds)
        
            # evaluating coeff network
            coeff_nrmse = calculate_nrmse(self.val_s, coeff_network_preds)
            return np.mean(coeff_nrmse.detach().cpu().numpy())

        elif self.method == "pressure":
            first_layer = trial.suggest_int(f"n_neuro_first", 16, 2048)
            shrinkage = trial.suggest_float(f"shrinkage", 0.2, 0.8)
            n_layers = trial.suggest_int(f"num_layers", 2, 6)
            layers = []
            layer = first_layer
            for _ in range(n_layers):
                layers.append(layer)
                layer = int(layer*shrinkage)
            hidden_layers = layers[::-1]
            lr = 1e-3
            epochs = 3000
            activation = self.activation_dict[trial.suggest_int("activation", 1, 4)]

            hyps = MLPHyperparameters(hidden_layers=hidden_layers, input_dim=x_std.shape[-1],
                                        output_dim=y_std.shape[-1], activation=activation)
            cp_network = MLP(hyps).to(**kwargs)
            cp_network = train_net(cp_network, x_std, y_std, epochs=epochs, learning_rate=lr)
        
            # predicting using trained network
            cp_network_preds = cp_network(x_transform.transform(self.val_x))
            cp_network_preds = y_transform.inverse_transform(cp_network_preds)
        
            # evaluating cp network
            cp_nrmse = calculate_nrmse(self.val_y, cp_network_preds)
            return np.mean(cp_nrmse.detach().cpu().numpy())

        elif self.method == "friction":
            first_layer = trial.suggest_int(f"n_neuro_first", 16, 2048)
            shrinkage = trial.suggest_float(f"shrinkage", 0.2, 0.8)
            n_layers = trial.suggest_int(f"num_layers", 2, 6)
            layers = []
            layer = first_layer
            for _ in range(n_layers):
                layers.append(layer)
                layer = int(layer*shrinkage)
            hidden_layers = layers[::-1]
            lr = 1e-3
            epochs = 3000
            activation = self.activation_dict[trial.suggest_int("activation", 1, 4)]

            hyps = MLPHyperparameters(hidden_layers=hidden_layers, input_dim=x_std.shape[-1],
                                        output_dim=z_std.shape[-1], activation=activation)
            cf_network = MLP(hyps).to(**kwargs)
            cf_network = train_net(cf_network, x_std, z_std, epochs=epochs, learning_rate=lr)
        
            # predicting using trained network
            cf_network_preds = cf_network(x_transform.transform(self.val_x))
            cf_network_preds = z_transform.inverse_transform(cf_network_preds)
        
            # evaluating cp network
            cf_nrmse = calculate_nrmse(self.val_z, cf_network_preds)
            return np.mean(cf_nrmse.detach().cpu().numpy())

        elif self.method == "mtaero":
            hidden_layers = [trial.suggest_int("n_neurons", 8, 64)] * trial.suggest_int("n_layers", 2, 6)
            
            first_layer = trial.suggest_int(f"n_neuro_first", 16, 2048)
            shrinkage = trial.suggest_float(f"shrinkage", 0.2, 0.8)
            n_layers = trial.suggest_int(f"num_layers", 2, 6)
            layers = []
            layer = first_layer
            for _ in range(n_layers):
                layers.append(layer)
                layer = int(layer*shrinkage)
            decoder_layers = layers[::-1]
            
            latent_dim = trial.suggest_int("latent", 16, 128)
            lr = 1e-3
            epochs = 3000
            activation = self.activation_dict[trial.suggest_int("activation", 1, 4)]
            weight = trial.suggest_float("weight", 0.1, 0.9)

            cp_decoder_hyps = MLPHyperparameters(hidden_layers=decoder_layers, input_dim=latent_dim,
                                                output_dim=y_std.shape[-1], activation=activation)
            scalar_hyps = MLPHyperparameters(hidden_layers=hidden_layers, input_dim=x_std.shape[-1],
                                                output_dim=s_std.shape[-1] + latent_dim, activation=activation)

            aero_net = MTAeroNet(cp_decoder_hyps, scalar_hyps, s_std.shape[-1]).to(**kwargs)
            aero_net = train_mt_net_weighted(aero_net, x_std, y_std, s_std, epochs=epochs, learning_rate=lr, weight=weight)
        
            # predicting using trained network
            cp_preds, sca_preds = aero_net(x_transform.transform(self.val_x))
            cp_preds = y_transform.inverse_transform(cp_preds)
            sca_preds = s_transform.inverse_transform(sca_preds)
        
            # evaluating network
            cp_nrmse = calculate_nrmse(self.val_y, cp_preds)
            sca_nrmse = calculate_nrmse(self.val_s, sca_preds)
            
            return np.mean(cp_nrmse.detach().cpu().numpy()), np.mean(sca_nrmse.detach().cpu().numpy())

        elif self.method == "mhaero":
            hidden_layers = [trial.suggest_int("n_neurons", 8, 64)] * trial.suggest_int("n_layers", 2, 6)
            
            dec_layers = []
            for i in range(2):
                first_layer = trial.suggest_int(f"n_neuro_first_{i+1}", 16, 2048)
                shrinkage = trial.suggest_float(f"shrinkage_{i+1}", 0.2, 0.8)
                n_layers = trial.suggest_int(f"num_layers_{i+1}", 2, 6)
                layers = []
                layer = first_layer
                for _ in range(n_layers):
                    layers.append(layer)
                    layer = int(layer*shrinkage)
                decoder_layers = layers[::-1]
                dec_layers.append(decoder_layers)
            
            latent_dim = trial.suggest_int("latent", 16, 128)
            lr = 1e-3
            epochs = 3000
            activation = self.activation_dict[trial.suggest_int("activation", 1, 4)]
            weight = trial.suggest_float("weight", 0.1, 0.9)

            cp_decoder_hyps = MLPHyperparameters(hidden_layers=dec_layers[0], input_dim=latent_dim,
                                                output_dim=y_std.shape[-1], activation=activation)
            cf_decoder_hyps = MLPHyperparameters(hidden_layers=dec_layers[1], input_dim=latent_dim,
                                                output_dim=z_std.shape[-1], activation=activation)
            scalar_hyps = MLPHyperparameters(hidden_layers=hidden_layers, input_dim=x_std.shape[-1],
                                                output_dim=s_std.shape[-1] + latent_dim, activation=activation)

            aero_net = MHAeroNet(cp_decoder_hyps, scalar_hyps, cf_decoder_hyps, s_std.shape[-1]).to(**kwargs)
            aero_net = train_mh_net_weighted(aero_net, x_std, y_std, z_std, s_std, epochs=epochs, learning_rate=lr, weight=weight)
        
            # predicting using trained network
            cp_preds, cf_preds, sca_preds = aero_net(x_transform.transform(self.val_x))
            cp_preds = y_transform.inverse_transform(cp_preds)
            cf_preds = z_transform.inverse_transform(cf_preds)
            sca_preds = s_transform.inverse_transform(sca_preds)
        
            # evaluating network
            cp_nrmse = calculate_nrmse(self.val_y, cp_preds)
            cf_nrmse = calculate_nrmse(self.val_z, cf_preds)
            sca_nrmse = calculate_nrmse(self.val_s, sca_preds)
            
            return np.mean(cp_nrmse.detach().cpu().numpy()) + np.mean(cf_nrmse.detach().cpu().numpy()), np.mean(sca_nrmse.detach().cpu().numpy())

        elif self.method == "mha-pressure":

            # latent and scalar networks
            hidden_layers_1 = [trial.suggest_int("n_neurons_1", 8, 64)] * trial.suggest_int("n_layers_1", 2, 6)
            hidden_layers = [trial.suggest_int("n_neurons", 8, 64)] * trial.suggest_int("n_layers", 2, 6)
            activation = self.activation_dict[trial.suggest_int("activation", 1, 4)]
            weight = trial.suggest_float("weight", 0.1, 0.9)

            # setting the autoencoder hyperparameters
            first_layer = trial.suggest_int("n_neuro_first", 16, 2048)
            shrinkage = trial.suggest_float("shrinkage", 0.2, 0.8)
            n_layers = trial.suggest_int("num_layers", 2, 6)
            layers = []
            layer = first_layer
            for _ in range(n_layers):
                layers.append(layer)
                layer = int(layer*shrinkage)
            encoder_layers = layers
            decoder_layers = layers[::-1]
            latent_dim = trial.suggest_int("latent", 16, 128)
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
            mha_net = MHANet(encoder_hyps1, latent_hyps1, decoder_hyps1, scalar_hyps1).to(**kwargs)
            mha_net = train_mha_net_weighted(mha_net, x_std, y_std, s_std, epochs=epochs, learning_rate=lr, weight=weight)

            # predicting using trained network
            _, sca_preds, vec_preds = mha_net(x_transform.transform(self.val_x))
            vec_preds = y_transform.inverse_transform(vec_preds)
            sca_preds = s_transform.inverse_transform(sca_preds)

            # evaluating network
            vec_nrmse = calculate_nrmse(self.val_y, vec_preds)
            sca_nrmse = calculate_nrmse(self.val_s, sca_preds)

            return np.mean(vec_nrmse.detach().cpu().numpy()), np.mean(sca_nrmse.detach().cpu().numpy())

        elif self.method == "mha-friction":

            # latent and scalar networks
            hidden_layers_1 = [trial.suggest_int("n_neurons_1", 8, 64)] * trial.suggest_int("n_layers_1", 2, 6)
            hidden_layers = [trial.suggest_int("n_neurons", 8, 64)] * trial.suggest_int("n_layers", 2, 6)
            activation = self.activation_dict[trial.suggest_int("activation", 1, 4)]
            weight = trial.suggest_float("weight", 0.1, 0.9)

            # setting the autoencoder hyperparameters
            first_layer = trial.suggest_int("n_neuro_first", 16, 2048)
            shrinkage = trial.suggest_float("shrinkage", 0.2, 0.8)
            n_layers = trial.suggest_int("num_layers", 2, 6)
            layers = []
            layer = first_layer
            for _ in range(n_layers):
                layers.append(layer)
                layer = int(layer*shrinkage)
            encoder_layers = layers
            decoder_layers = layers[::-1]
            latent_dim = trial.suggest_int("latent", 16, 128)
            lr = 1e-3
            epochs = 3000

            encoder_hyps1 = MLPHyperparameters(hidden_layers=encoder_layers, input_dim=z_std.shape[-1], 
                                                output_dim=latent_dim, activation=activation)
            latent_hyps1 = MLPHyperparameters(hidden_layers=hidden_layers, input_dim=x_std.shape[-1],
                                                        output_dim=latent_dim, activation=activation)
            decoder_hyps1 = MLPHyperparameters(hidden_layers=decoder_layers, input_dim=latent_dim,
                                                        output_dim=z_std.shape[-1], activation=activation)
            scalar_hyps1 = MLPHyperparameters(hidden_layers=hidden_layers_1, input_dim=latent_dim,
                                                            output_dim=s_std.shape[-1], activation=activation)
            mha_net = MHANet(encoder_hyps1, latent_hyps1, decoder_hyps1, scalar_hyps1).to(**kwargs)
            mha_net = train_mha_net_weighted(mha_net, x_std, z_std, s_std, epochs=epochs, learning_rate=lr, weight=weight)

            # predicting using trained network
            _, sca_preds, vec_preds = mha_net(x_transform.transform(self.val_x))
            vec_preds = z_transform.inverse_transform(vec_preds)
            sca_preds = s_transform.inverse_transform(sca_preds)

            # evaluating network
            vec_nrmse = calculate_nrmse(self.val_z, vec_preds)
            sca_nrmse = calculate_nrmse(self.val_s, sca_preds)

            return np.mean(vec_nrmse.detach().cpu().numpy()), np.mean(sca_nrmse.detach().cpu().numpy())

    def optimize(self, num_trials): 
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        if self._type == "single":
            def objective(trial):
                o1 = self.obj_func(trial)
                return o1
            opt_study = optuna.create_study(sampler=sampler,directions=["minimize"])
        else:
            def objective(trial):
                o1, o2 = self.obj_func(trial)
                return o1, o2  
            opt_study = optuna.create_study(sampler=sampler,directions=["minimize","minimize"])
        
        t_start = time.perf_counter()
        opt_study.optimize(objective, n_trials=num_trials)
        t_end = time.perf_counter()
        time_elapsed = t_end - t_start

        best_hyps = opt_study.best_trials
        values = []
        parameters = []
        for hyp in best_hyps:
            values.append(hyp.values)
            parameters.append(hyp.params)

        return values, parameters, time_elapsed




