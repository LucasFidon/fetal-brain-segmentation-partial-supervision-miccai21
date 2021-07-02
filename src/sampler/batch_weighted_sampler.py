from torch.utils.data import BatchSampler
from src.sampler.weighted_sampler import WeightedSampler


class BatchWeightedSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last=False):
        """
        Custom Batch Sampler that calls the sampler once per iteration
        instead of once per epoch.
        An epoch consists in n iterations, where n is equal
        to the number of examples in dataset.
        :param sampler: WeightedSampler; a PyTorch sampler
        :param batch_size: int; number of samples per batch.
        :param drop_last: bool; if True, incomplete batch at the end
        of an epoch are dropped.
        """
        assert isinstance(sampler, WeightedSampler), \
            "The sampler used in the BatchWeightedSampler must be a WeightedSampler"
        super(BatchWeightedSampler, self).__init__(
            sampler,
            batch_size,
            drop_last
        )

    @property
    def num_samples(self):
        return self.sampler.num_samples

    def __iter__(self):
        for _ in range(len(self)):
            batch = self.sampler.draw_samples(self.batch_size)
            self.batch = [x for x in batch]
            yield batch

    def __len__(self):
        """
        :return: int; number of batches per epoch.
        """
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
