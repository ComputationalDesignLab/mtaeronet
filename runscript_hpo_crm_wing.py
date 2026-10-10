import os
import argparse
import numpy as np
from scipy.io import loadmat, savemat
from src.hpo.optimizer import HPOptimizer
import pickle
from src.utils.set_seed import set_seed

# parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument("--samples", help="number of samples")
parser.add_argument("--method", help="modeling method to use")
parser.add_argument("--type", help="type of optimization")
parser.add_argument("--trials", help="number of trials")
parser.add_argument("--seed", help="number of trials")
args = parser.parse_args()

# constants
NUM_SAMPLES = int(args.samples)
SEED = int(args.seed)
set_seed(SEED)
METHOD = args.method
TYPE = args.type
N = int(NUM_SAMPLES/50)
TRIALS = int(args.trials)

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

train_x = np.vstack(X)
train_s = np.vstack(S)
train_y = np.vstack(Y)
train_z = np.vstack(Z)

# defining optimizer
optimizer = HPOptimizer(train_x, train_y, train_z, train_s, _type=TYPE,
                        split_ratio=0.20, method=METHOD, seed=SEED)
best_vals, best_params, time = optimizer.optimize(num_trials=TRIALS)

# saving the results
results = {"vals": best_vals, "params": best_params, "time": time}
os.makedirs(f"./saved_models/{METHOD}/wing/hyps/{SEED}", exist_ok=True)
with open(f"./saved_models/{METHOD}/wing/hyps/{SEED}/hyp_{NUM_SAMPLES}.pkl", "wb") as file:
    pickle.dump(results, file)