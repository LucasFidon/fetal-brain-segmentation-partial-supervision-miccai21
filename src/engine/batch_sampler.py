from torch.utils.data import BatchSampler


class BatchSamplerFillLastBatch(BatchSampler):
    """
    PyTorch BatchSampler with an additional option.
    """
    def __init__(self, sampler, batch_size, drop_last=False, fill_last_batch=True):
        super(
            BatchSamplerFillLastBatch,
            self).__init__(
            sampler,
            batch_size,
            drop_last)
        self.fill_last_batch = fill_last_batch

    def __iter__(self):
        batch = []
        batch_needs_to_be_emptied = False
        for idx in self.sampler:
            # Empty the batch if it was full and has been delivered
            if batch_needs_to_be_emptied:
                batch = []
                batch_needs_to_be_emptied = False
            # Add new index to the batch
            batch.append(idx)
            # Deliver the batch if it is full after adding a new index
            if len(batch) == self.batch_size:
                batch_needs_to_be_emptied = True
                yield batch
        # Last batch: optionally fill it and return it
        if len(batch) > 0 and not self.drop_last:
            if self.fill_last_batch:
                # Complete the last batch
                for idx in self.sampler:
                    if len(batch) ==  self.batch_size:
                        break
                    batch.append(idx)
            yield batch