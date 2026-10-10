import torch
import numpy as np
from scipy.stats.qmc import Sobol, scale
from dataclasses import dataclass

kwargs = {"dtype": torch.float32, "device": "cuda" if torch.cuda.is_available() else "cpu"}

@dataclass
class QMCResults:
    predictions: np.ndarray
    mean_predictions: np.ndarray
    std_predictions: np.ndarray

class QMCRunner:

    def __init__(self, model, dim, distribution_list, l_bounds, u_bounds):
        self.model = model
        self.dim = dim
        self.distribution_list = distribution_list
        self.l_bounds = l_bounds
        self.u_bounds = u_bounds

    def generate_samples(self, po2):
        sampler = Sobol(d=self.dim, scramble=False)
        samples = sampler.random_base2(po2)
        samples = samples[1:,:]
        scaled_samples = [self.distribution_list[k].ppf(samples[:,k]) for k in range(len(self.distribution_list))]
        torch_samples = torch.transpose(torch.from_numpy(np.column_stack([scaled_samples])), 0, 1).to(**kwargs)
        return torch_samples

    def run(self, po2, in_transform, cp_transform, cf_transform, sca_transform):
        samples = self.generate_samples(po2)
        cp_predictions, cf_predictions, sca_predictions = self.model(in_transform.transform(samples))
        cp_predictions = cp_transform.inverse_transform(cp_predictions)
        sca_predictions = sca_transform.inverse_transform(sca_predictions)
        cf_predictions = cf_transform.inverse_transform(cf_predictions)

        mean_cp_predictions = torch.mean(cp_predictions, axis=0)
        std_cp_predictions = torch.std(cp_predictions, axis=0)
        cp_results = QMCResults(predictions=cp_predictions.detach().cpu().numpy(), mean_predictions=mean_cp_predictions.detach().cpu().numpy(),
                             std_predictions=std_cp_predictions.detach().cpu().numpy())

        mean_cf_predictions = torch.mean(cf_predictions, axis=0)
        std_cf_predictions = torch.std(cf_predictions, axis=0)
        cf_results = QMCResults(predictions=cf_predictions.detach().cpu().numpy(), mean_predictions=mean_cf_predictions.detach().cpu().numpy(),
                             std_predictions=std_cf_predictions.detach().cpu().numpy())

        mean_sca_predictions = torch.mean(sca_predictions, axis=0)
        std_sca_predictions = torch.std(sca_predictions, axis=0)
        sca_results = QMCResults(predictions=sca_predictions.detach().cpu().numpy(), mean_predictions=mean_sca_predictions.detach().cpu().numpy(),
                                std_predictions=std_sca_predictions.detach().cpu().numpy())
        
        return cp_results, cf_results, sca_results



