from nvidia.dali.plugin.pytorch import DALIGenericIterator


class PyTorchIterator(DALIGenericIterator):
    def __init__(self,
                 pipelines,
                 output_map,
                 size,
                 auto_reset=False,
                 fill_last_batch=True,
                 dynamic_shape=False,
                 last_batch_padded = False):
        super(PyTorchIterator, self).__init__(
            pipelines=pipelines,
            output_map=output_map,
            size=size,
            auto_reset=auto_reset,
            fill_last_batch=fill_last_batch,
            dynamic_shape=dynamic_shape,
            last_batch_padded=last_batch_padded,
        )

    @property
    def batch_sampler(self):
        """
        Expose the batch sampler to keep the same API as PyTorch Dataloader.
        Not tested with several pipelines.
        """
        return self._pipes[0].batch_sampler

    @property
    def dataset(self):
        """
        Expose the dataset to keep the same API as PyTorch Dataloader.
        Not tested with several pipelines.
        """
        return self._pipes[0].loader
