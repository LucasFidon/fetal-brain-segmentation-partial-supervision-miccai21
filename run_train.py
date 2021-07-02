"""
@brief  PyTorch training code.

@author Lucas Fidon (lucas.fidon@kcl.ac.uk)
@date   July 2021
"""

import argparse
from decimal import Decimal
import json
from tqdm import tqdm
import numpy as np
import torch
from torch.optim import Adam
import torchnet as tnt
from src.network_architectures.custom_3dunet.unet import UNet3D
from src.engine.engine import Engine
from src.dataset.dataset import Fetal3DSegDataset, Fetal3DSegPipeline
from src.sampler.weighted_sampler import WeightedSampler
from src.sampler.batch_weighted_sampler import BatchWeightedSampler
from src.utils.definitions import *
from src.engine.dali_iterator import PyTorchIterator
# You need to install my python package for the label-set loss function
# https://github.com/LucasFidon/label-set-loss-functions
from label_set_loss_functions.loss import LeafDiceLoss, MarginalizedDiceLoss, MeanDiceLoss


SUPPORTED_LOSS = ['mean_dice', 'leaf_dice', 'marginalized_dice']
SUPPORTED_MODEL = ['unet']
SUPPORTED_OPTIMIZER = ['adam']

# Comment this for deterministic behaviour or when using variable patch size
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

parser = argparse.ArgumentParser(
    description='Segmentation training script')

# Model options
parser.add_argument('--model', default='unet', type=str)
parser.add_argument('--norm', default='instance', type=str, help='instance or batch')
parser.add_argument('--num_classes', default=NUM_CLASS, type=int)
parser.add_argument('--save', default='./logs/test_fetal3d_seg', type=str,
                    help='save parameters and logs in this folder')
parser.add_argument('--resume', default='', type=str,
    help='(optional) resume training from the checkpoint indicated; ex: ./logs/model_iter12000.pt7')
parser.add_argument('--dtype', default='float', type=str)
parser.add_argument('--nthread', default=4, type=int)


# Training options
parser.add_argument('--loss', default='mean_dice', type=str,
                    help='Available options are %s' % SUPPORTED_LOSS)
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--save_every_n_iter', default=100, type=int)  # 100
parser.add_argument('--weight_decay', default=0., type=float)

# Misc optimization options
parser.add_argument('--optim', default='adam')
parser.add_argument('--batch_size', default=3, type=int)
parser.add_argument('--patch_size', default='[144,160,144]', type=str)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value to use for the optimizer.')

# Options for data augmentation
parser.add_argument('--no_flip', action='store_true',
                    help='Do not use right-left flip augmentation during training with proba 0.5')
parser.add_argument('--flip_all', action='store_true',
                    help='Use flip augmentation (all axis) during training with proba 0.5')
parser.add_argument('--no_data_augmentation', action='store_true')
parser.add_argument('--gaussian_blur', action='store_true')
parser.add_argument('--no_zoom', action='store_true')

# Options to save GPU memory
parser.add_argument('--grad_check', action='store_true',
                    help="(optional; recommended) It activates gradient checkpointing."
                         " Cannot be combined with --fp16")
parser.add_argument('--fp16', action='store_true',
                    help='(optional) It activates mixed precision. Cannot be combined with --grad_check.')

# Data
parser.add_argument('--data_csv',
                    default = os.path.join(
                        '/data',
                        'fetal_brain_srr_parcellation/training.csv'
                    ),
                    type=str,
                    help='path to a csv file that maps sample id '
                         'to image path and segmentation path')


def get_network(model, num_channels, num_classes, norm='instance', grad_check=False):
    """
    Return the deep neural network architecture.
    :param num_channels: number of input channels.
    :return: neural network model.
    """
    assert model in SUPPORTED_MODEL, 'Model %s not supported' % model
    print('A 3D U-Net is used')
    network = UNet3D(
        in_channels=num_channels,
        out_classes=num_classes,
        out_channels_first_layer=30,
        num_encoding_blocks=5,
        residual=False,
        normalization=norm,
        padding=True,
        activation='LeakyReLU',
        upsampling_type='trilinear',
        gradient_checkpointing=grad_check,
    )
    # Put the model on the gpu.
    network.cuda()
    # Set the network in training mode
    network.train()
    return network


def get_loss(loss_name='mean_dice'):
    if loss_name == 'mean_dice':
        print('Use the mean Dice loss (denominator squared)')
        loss = MeanDiceLoss(squared=True)
        return loss
    elif loss_name == 'mean_dice_partial':
        print('Use the marginalized Dice loss')
        print('Labels supersets:')
        print(LABELSET_MAP)
        loss = MarginalizedDiceLoss(LABELSET_MAP)
        return loss
    elif loss_name == 'leaf_dice':
        print('Use the mean Dice loss with soft labels ground-truth (denominator squared)')
        print('Labels supersets:')
        print(LABELSET_MAP)
        loss = LeafDiceLoss(LABELSET_MAP)
        return loss
    else:
        raise NotImplementedError('Loss function %s not defined' % loss_name)


def main(opt):
    """
    Train a network for segmentation.
    :param opt: parsed command line arguments.
    """
    def create_iterator(opt):
        """
        Return a PyTorch data loader.
        This is a pipeline that includes:
        - data loading
        - random indices sampling
        - data normalization / augmentation
        :return: PyTorch data loader
        """
        # Create the dataset (data reader)
        dataset = Fetal3DSegDataset(
            data_csv=opt.data_csv,
            use_data_augmentation=not opt.no_data_augmentation,
            use_zoom=not opt.no_zoom,
        )

        # Create the index batch sampler
        idx_sampler = WeightedSampler(
            num_samples=len(dataset),
            weights=1,
        )
        batch_idx_sampler = BatchWeightedSampler(
            sampler=idx_sampler,
            batch_size=opt.batch_size,
            drop_last=False,
        )

        # Create the data normalization/augmentation pipeline
        dali_pipeline = Fetal3DSegPipeline(
                    dataset,
                    batch_index_sampler=batch_idx_sampler,
                    patch_size=json.loads(opt.patch_size),
                    num_threads=opt.nthread,
                    do_flip=not opt.no_flip,
                    do_flip_all=opt.flip_all,
                    do_gauss_blur=opt.gaussian_blur,
                    do_zoom=False,  # todo
                )
        # Create the DALI PyTorch dataloader
        data_loader = PyTorchIterator(
            pipelines=dali_pipeline,
            size=len(dataset),
            output_map=['img', 'seg', 'idx'],
            # if True the last batch is completed to have a length equal to batch_size.
            # However, DALI is not using the batch sampler to select the indices
            # used to fill the last batch...
            fill_last_batch=True,
            # if False samples used to complete the previous last batch
            # are removes from the next epoch.
            last_batch_padded=True,
            auto_reset=True,
        )
        return data_loader

    def create_optimizer(opt, lr):
        # Create the optimizer
        if opt.optim == 'adam':
            print('Create Adam optimizer with lr=%f, momentum=%f' % (lr, opt.momentum))
            optim = Adam(network.parameters(),
                        lr,
                        betas=(opt.momentum, 0.999),
                        weight_decay=opt.weight_decay,
                        amsgrad=False
                    )
        else:
            return ValueError('Optimizer %s not supported' % opt.optim)

        return optim


    def infer_and_eval(sample):
        """
        :param sample: couple of tensors; input batch and
        corresponding ground-truth segmentations.
        :return: float, 1d tensor; mean loss for the input batch
        and None or batch of predicted segmentations.
        """
        # DALI data loading pipeline is used
        inputs = sample[0]['img']
        targets = torch.squeeze(sample[0]['seg'], dim=1)

        y = network(inputs)
        loss = loss_func(y, targets)  # float; mean batch loss
        del y
        return loss

    def log(t, state):
        """
        Save the network parameters, the weights, and the logs.
        :param t: dict; Contains info about current hyperparameters value.
        :param state: dict; Contains info about current hyperparameters value.
        """
        torch.save(
            dict(params=network.state_dict(),
                 epoch=t['epoch'],
                 iter=t['iter'],
                 optimizer=state['optimizer'].state_dict()),
            os.path.join(opt.save, 'model_iter%d.pt7' % t['iter'])
        )
        z = {**vars(opt), **t}
        # Write the logs for epoch t.
        with open(os.path.join(opt.save, 'log.txt'), 'a') as flog:
            flog.write('json_stats: ' + json.dumps(z) + '\n')
        print(z)

    def on_sample(state):
        """
        Called after a batch is drawn from the training dataset.
        :param state: dict.
        """
        pass

    def on_forward(state):
        """
        Called at the beginning of each forward pass.
        :param state: dict.
        """
        loss = float(state['loss'])
        # Update running average and std for the loss value.
        meter_loss.add(loss)
        if state['train']:
            state['iterator'].set_postfix(loss=loss)

    def on_start(state):
        """
        Called only once, at the beginning of the training.
        :param state: dict.
        """
        state['epoch'] = epoch
        state['t'] = iter

    def on_start_epoch(state):
        """
        Called at the beginning of each epoch.
        :param state: dict.
        """
        meter_loss.reset()
        timer_train.reset()
        epoch = state['epoch'] + 1
        if epoch in epoch_step and opt.lr_decay_ratio != 1:
            # learning rate decay
            lr = state['optimizer'].param_groups[0]['lr']
            state['optimizer'] = create_optimizer(opt, lr * opt.lr_decay_ratio)
        # reinitialize the progress bar for the next epoch
        state['iterator'] = tqdm(train_loader, dynamic_ncols=True)

    def save_model_and_logs(state):
        """
        Called at the end of each epoch.
        Aggregate logs at the end of each epoch, and print them.
        :param state: dict.
        """
        train_loss = meter_loss.value()  # mean and std
        train_time = timer_train.value()
        meter_loss.reset()
        timer_test.reset()
        logs_dict = {
            "train_loss": train_loss[0],
            "train_loss_std": train_loss[1],
            "epoch": state['epoch'],
            "iter": state['t'],
            "num_classes": num_classes,
            "n_parameters": n_parameters,
            "train_time": train_time,
        }
        print(log(logs_dict, state))
        print('==> id: %s, epoch (%d/%d), iter %d, training loss: \33[91m%.3E\033[0m' %
              (opt.save, state['epoch'], opt.epochs, state['t'], Decimal(train_loss[0])))

    epoch_step = json.loads(opt.epoch_step)
    num_classes = opt.num_classes
    num_channels = 1

    # Create the dataloader.
    # It loads the data, pre-process them, and give them to the network.
    train_loader = create_iterator(opt)

    # Create the network that will be trained.
    network = get_network(
        opt.model, num_channels, num_classes, norm=opt.norm, grad_check=opt.grad_check)
    # print(network)

    # Create the loss function to use for training the network.
    loss_func = get_loss(opt.loss)

    optimizer = create_optimizer(opt, opt.lr)

    epoch = 1
    iter = 1
    # (optional) Restore the parameters of the network to resume training
    # from a previous session.
    if opt.resume != '':
        state_dict = torch.load(opt.resume)
        epoch = state_dict['epoch']
        iter = state_dict['iter']
        print("Resume training from iter %d" % iter)

        # Restore the model parameters
        network.load_state_dict(state_dict['params'])
        optimizer.load_state_dict(state_dict['optimizer'])

        if opt.dro:
            # Restore the weights of the sampler (if applicable)
            weights_path = os.path.join(os.path.dirname(opt.resume),
                                        'weights_iter%d.pt7' % iter)
            train_loader.batch_sampler.sampler.load_weights(weights_path)

    # Print the number of parameters
    trainable_model_parameters = filter(
        lambda p: p.requires_grad, network.parameters())
    n_parameters = int(sum([np.prod(p.size()) for p in trainable_model_parameters]))
    print('\nTotal number of parameters:', n_parameters)

    # Maintain average and std of the loss for logs.
    meter_loss = tnt.meter.AverageValueMeter()
    timer_train = tnt.meter.TimeMeter('s')
    timer_test = tnt.meter.TimeMeter('s')

    if not os.path.exists(opt.save):
        os.mkdir(opt.save)

    engine = Engine()
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    # engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_every_n_iter'] = save_model_and_logs
    engine.hooks['on_start'] = on_start

    # Train the network.
    engine.train(infer_and_eval, train_loader, opt.epochs, optimizer,
                 every_n_iter=opt.save_every_n_iter, fp16=opt.fp16)


if __name__ == '__main__':
    opt = parser.parse_args()
    print('parsed options:', vars(opt))
    if not os.path.exists('logs'):
        os.mkdir('logs')
    if not os.path.exists(opt.save):
        os.mkdir(opt.save)
    main(opt)
