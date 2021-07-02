import torch
from torch.utils.data import Sampler
import numpy as np


class WeightedSampler(Sampler):
    """
    PyTorch implementation for Weighted Sampler.
    """
    def __init__(self, weights=1, num_samples=-1):
        """
        :param weights: int, float or 1d tensor;
        weights of the sampling.
        :param num_samples: int; number of samples in the dataset.
        num_samples must be specified if weights_init is a constant value.
        When weights_init is a 1d tensor, the number of samples is inferred
        automatically from the size of weights_init.
        """
        self.num_samples = num_samples

        if isinstance(weights, float) or isinstance(weights, int):
            assert num_samples > 0, \
                "The number of samples should be specified if a constant weights_init is used"
            print('Initialize the weights of the hardness weighted sampler to the value', weights)
            self.weights = torch.tensor(np.random.normal(loc=weights, scale=0., size=num_samples))
        # Initialization with a given vector of per-example loss values
        else:
            assert len(weights.shape) == 1, "initial weights should be a 1d tensor"
            self.weights = weights.float()
            if self.num_samples <= 0:
                self.num_samples = weights.shape[0]
            else:
                assert self.num_samples == weights.shape[0], \
                    "weights_init should have a size equal to num_samples"

    def get_distribution(self):
        return self.weights

    def draw_samples(self, n):
        """
        Draw n sample indices using the hardness weighting sampling method.
        """
        eps = 0.0001 / self.num_samples
        # Get the distribution (softmax)
        distribution = self.get_distribution()
        p = distribution.numpy()
        # Set min proba to epsilon for stability
        p[p <= eps] = eps
        p /= p.sum()
        # Use numpy implementation of multinomial sampling because it is much faster
        # than the one in PyTorch
        sample_list = np.random.choice(
            self.num_samples,
            n,
            p=p,
            replace=False,
        ).tolist()
        return sample_list

    def save_weights(self, save_path):
        torch.save(self.weights, save_path)

    def load_weights(self, weights_path):
        print('Load the sampling weights from %s' % weights_path)
        weights = torch.load(weights_path)
        self.weights = weights
        self.num_samples = self.weights.size()[0]
