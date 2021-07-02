"""
@brief  The engine contains the training loop.
        It allows to easily customized the training loop
        using hook functions.

        Compared to the Engine class from torchnet,
        I have added a hook function before the backward pass.

@author Lucas Fidon (lucas.fidon@kcl.ac.uk)
@date   July 2021
"""

from torchnet.engine import Engine
from torch.cuda.amp import autocast, GradScaler


class Engine(Engine):

    def train(self, network, iterator, maxepoch,
              optimizer, every_n_iter, fp16=False):
        """
        Training loop.
        The number of epochs consists of n training iterations,
        where n is equal to the number of examples in the training dataset.
        :param network: network architecture.
        :param iterator: Sampler. Used to load data to be given
        to the network at each iteration.
        :param maxepoch: int; maximum number of epochs.
        :param optimizer: optimizer to use (e.g. SGD, Adam, ...etc).
        """
        state = {
            'network': network,
            'iterator': iterator,
            'maxepoch': maxepoch,
            'optimizer': optimizer,
            'epoch': 1,
            't': 1,  # iteration number
            'every_n_iter': every_n_iter,
            'train': True,
        }
        scaler = None
        if fp16:
            # The gradient scaler is here to help avoiding gradient underflow
            # when training with mixed precision.
            scaler = GradScaler()
        # Call hook function to run before the beginning of training.
        self.hook('on_start', state)
        while state['epoch'] < state['maxepoch']:
            # Call hook function to run at the beginning of each epoch.
            self.hook('on_start_epoch', state)
            for sample in state['iterator']:
                state['sample'] = sample
                # call hook function to run after each sampling.
                self.hook('on_sample', state)

                def closure():
                    if fp16:  # use mixed precision
                        with autocast():
                            loss = state['network'](state['sample'])  # float16
                    else:
                        loss = state['network'](state['sample'])
                    state['loss'] = loss  # used for logs only
                    del state['sample']

                    # Add a hook before backward pass to allow the user
                    # to modify the loss.
                    self.hook('on_start_backward', state)

                    # Create the gradient.
                    if fp16:
                        # Scale the loss before computing the backward pass
                        # to avoid gradient underflow.
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    # Call hook function to run after each forward pass
                    # can be used to modify the output and the loss after
                    # the backward pass.
                    self.hook('on_forward', state)

                    return loss

                state['optimizer'].zero_grad()
                if fp16:
                    # scaler.step() first unscales the gradients of the optimizer's assigned params.
                    # If these gradients don't contain infs or NaNs, optimizer.step() is then called,
                    # otherwise, optimizer.step() is skipped.
                    loss = closure()
                    scaler.step(state['optimizer'])
                    #scaler.step(state['optimizer'], closure=closure)  # closure not supported with fp16 yet
                    # Updates the scale for next iteration.
                    scaler.update()
                else:
                    # closure is usefull for some optimizer that requires the value of the loss.
                    state['optimizer'].step(closure)

                # Call hook function to run after each update of
                # the network parameters.
                self.hook('on_update', state)

                if state['t'] % state['every_n_iter'] == 0:
                    # Save model and logs
                    self.hook('on_every_n_iter', state)
                state['t'] += 1

            state['epoch'] += 1
            # Call hook function to run at the end of each epoch.
            self.hook('on_end_epoch', state)
        self.hook('on_end', state)
        return state
