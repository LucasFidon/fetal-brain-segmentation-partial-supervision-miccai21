"""
@brief  PyTorch inference code for segmentation.

        A 3D fetal brain MRI is segmented using a pre-trained CNN.

@author Lucas Fidon (lucas.fidon@kcl.ac.uk)
@date   July 2021.
"""

import argparse
import json
import numpy as np
import torch
import torch.utils.data
from time import time
from torch.utils.data import DataLoader
from src.utils.definitions import *
from src.utils.misc import cast
from src.network_architectures.custom_3dunet.unet import UNet3D
from src.image.inference_image import InferenceImage
from src.dataset.inference_single_volume_dataset import SingleVolumeDataset

parser = argparse.ArgumentParser(description='Run inference for segmentation')
# Data options
parser.add_argument('--input', default='.', type=str, help='Path of the 3D MRI to segment. '
                    'We assume that the brain mask is in the same folder.')
parser.add_argument('--output_folder', default='', type=str,
    help='(optional) Path of the folder where to store the segmentations.'
    'By default the folder containing the input image is used.'
)
# Model options
parser.add_argument(
    '--model',
    nargs='+',
    type=str,
    help='Path(s) to the parameters of the pre-trained model(s). '
         'It also accepts multiple models for ensembling.',
)
parser.add_argument('--patch_size', default='[144,160,144]', type=str)
parser.add_argument('--num_classes', default=NUM_CLASS, type=int,
                    help='Number of classes to predict.'
                         ' It must be compatible with the deep neural network in --save')
# Pre and post-processing option
parser.add_argument('--save_proba', action='store_true')
parser.add_argument('--mask_margin', default=3, type=int,
                    help='(optional) Margin to use to mask the input image.')

# Hyperparameters
VERSION='0.1.0'
NUM_CHANNELS = 1


def create_image_loader(img_path, mask_path=None,
                        patch_size=[144,160,144], mask_margin=3):
    """
    Create the image loader.
    In PyTorch the dataset is responsible for loading the data,
    pre-processing them, and apply data augmentation (optional)
    """
    # Note that the data are already normalised for segmentation.
    # Create the dataset.
    dataset = SingleVolumeDataset(
        img_path=img_path,
        mask_path=mask_path,
        patch_size=patch_size,
        mask_margin=mask_margin,
    )
    img_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    return img_loader


def get_network(num_channels, num_classes, norm='instance'):
    """
    Return the deep neural network architecture.
    :param num_channels: number of input channels.
    :return: neural network model.
    """
    # Typical Unet architecture used in SoA pipeline
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
    )
    # Put the model on the gpu.
    network.cuda()
    # Set the network in evaluation mode.
    network.eval()
    return network


def segment(img_loader, network, num_class=NUM_CLASS, return_proba=True):
    # Rk: ensembling is done before the softmax

    # Return the pre-softmax full-size output of the model
    score_map_shape = np.concatenate((
        [num_class], img_loader.dataset.img.shape))
    output_full_size = InferenceImage(
        shape=score_map_shape,
        patch_size=img_loader.dataset.patch_shape,
        fusion='gaussian',
        margin=0,
    )
    for sample in img_loader:
        patch = cast(sample[0], 'float')
        coords = sample[2]
        # Segment the patches
        with torch.no_grad():  # Tell PyTorch to not store data for backpropagation
            # Normal inference
            out = network(patch)
            num_inputs = 1
            print('Perform test-time right-left flipping and ensembling')
            flip_inputs = torch.flip(patch, (2,))
            num_inputs += 1
            out += torch.flip(network(flip_inputs), (2,))
            # Normalize the output
            print('%s inferences have been done' % num_inputs)
            out /= num_inputs
        # Aggregate the patch prediction in the full size prediction
        output_full_size.add_patch(out, coords)
    if return_proba:
        return output_full_size.probability_maps
    else:
        return output_full_size.score_maps


def main(opt):
    """
    Run inference for a network that have been trained with main_seg_fetal3d.py.
    :param opt: parsed command line arguments.
    """
    def postprocess_and_save(output, save_proba=False):
        # Define the folder where to save the segmentation
        if opt.output_folder == '':
            # By default the segmentations are saved in the folder of the input.
            save_folder = os.path.split(opt.input)[0]
        else:
            save_folder = opt.output_folder
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
        print("Save segmentations in %s" % save_folder)

        # Get the name of the input slice
        name = img_loader.dataset.img_name

        # Predicted segmentation
        pred_cnn = np.argmax(output, axis=0)

        # Save the predicted parcellation from the CNN only
        save_path = os.path.join(
            save_folder, '%s_parcellation_cnn_autoseg.nii.gz' % name)
        img_loader.dataset.save(pred_cnn, save_path)  # use pred_cnn here

        if save_proba:
            proba = output
            save_path_proba = os.path.join(
                save_folder, '%s_parcellation_softmax_autoseg.nii.gz' % name)
            img_loader.dataset.save(proba, save_path_proba)


    def restore(model_path):
        assert os.path.exists(model_path), "Cannot find the model %s" % model_path
        state_dict = torch.load(model_path)
        network.load_state_dict(state_dict['params'])

    # Create the image loader
    img_loader = create_image_loader(
        opt.input,
        patch_size=json.loads(opt.patch_size),
        mask_margin=opt.mask_margin,
    )

    # Create the network
    network = get_network(NUM_CHANNELS, opt.num_classes)
    trainable_model_parameters = filter(
        lambda p: p.requires_grad, network.parameters())
    n_parameters = sum([np.prod(p.size()) for p in trainable_model_parameters])
    print('\nTotal number of parameters:', n_parameters)

    # Run inference for all samples/patches/subwindows and save the output seg
    pred_out_full = 0
    model_paths = opt.model
    print('Do ensembling with %d models' % len(model_paths))
    for model_path in model_paths:
        restore(model_path)
        pred_out_full += segment(
                img_loader,
                network,
                num_class=opt.num_classes,
                return_proba=True,  # if False, return the score map
        )
    pred_out_full /= len(model_paths)
    postprocess_and_save(pred_out_full, save_proba=opt.save_proba)


if __name__ == '__main__':
    opt = parser.parse_args()
    print('parsed options:', vars(opt))
    t_0 = time()
    main(opt)
    print('Inference performed in %.2f sec\n' % (time() - t_0))
