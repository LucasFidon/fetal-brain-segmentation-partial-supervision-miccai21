"""
@brief  Pytorch Dataset class used for inference with 3d image segmentation.
        It is designed for running inference on ONLY ONE image.
        The Volume is normalized to zeros mean and unit variance when loaded.

@author Lucas Fidon (lucas.fidon@kcl.ac.uk)
@date   july 2021.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from scipy.ndimage.morphology import binary_dilation
from src.utils.misc import pad_if_needed, MIN_SIZE


def _check_image_shape(img_shape):
    """
    Raise an error if the input image is not a 3D array
    or if no patch can fit in the input image.
    :param img_shape: shape of the input image.
    :param patch_size: int; size of the patch.
    """
    num_dim = len(img_shape)
    assert num_dim == 3, \
        "\nWrong number of dimensions, should be 3 but found %d" % num_dim


class SingleVolumeDataset(Dataset):
    def __init__(self, img_path, mask_path=None, patch_size=MIN_SIZE, mask_margin=3):
        """
        Dataset class used for inference with 3d image segmentation.
        It is designed for running inference on ONLY ONE image.
        The Volume is normalized to zeros mean and unit variance when loaded.
        :param img_path: str; path to the image.
        :param mask_path: str; path to the mask.
        :param patch_size: int or int list; size of the patch to use.
        :param mask_margin: int; number of dilation iteration to apply to the mask.
        """
        super(SingleVolumeDataset, self).__init__()
        if isinstance(patch_size, int):
            self.patch_shape = [patch_size] * 3
        else:
            self.patch_shape = patch_size
        self.dilation_num_iter = mask_margin
        self._preprocessing_pipeline(img_path, mask_path)
        self._set_patch_coordinates_list()

    def pad_if_needed(self, img, return_padding_values=False):
        return pad_if_needed(
            img,
            min_size=self.patch_shape,
            return_padding_values=return_padding_values
        )

    def _preprocessing_pipeline(self, img_path, mask_path):
        # Load the image and get image info
        self.img_path = img_path
        img_nii = nib.load(self.img_path)
        self.img = img_nii.get_data().astype(np.float32)
        if self.img.ndim == 4:
            self.img = self.img[:,:,:,0]
        self.ori_img_shape = np.copy(self.img.shape)
        self.affine = img_nii.affine
        self.header = img_nii.header
        self.img_name = os.path.split(img_path)[-1].split('.')[0]

        # Load the mask
        if mask_path is None:
            mask_path = img_path.replace('.nii', '_mask.nii')
            if not os.path.exists(mask_path):
                # Try with the filename 'mask.nii.gz'
                mask_path = os.path.join(
                    os.path.split(img_path)[0],
                    'mask.nii.gz',
                )
        self.mask_path = mask_path
        mask_nii = nib.load(self.mask_path)
        self.mask = mask_nii.get_data().astype(np.uint8)

        # Skull stripping
        if self.dilation_num_iter > 0:
            self.mask = binary_dilation(
                self.mask, iterations=self.dilation_num_iter)
        self.img[self.mask == 0] = 0.

        # Set Nan values to mean intensity value
        num_nans = np.count_nonzero(np.isnan(self.img))
        if num_nans > 0:
            print('\nWarning! %d NaN values were found in the image %s'
                  % (num_nans, self.img_path))
            print('Replace nan values with the mean value of the image.')
            self.img[np.isnan(self.img)] = np.nanmean(self.img)

        # Pad the image and the mask with zeros if needed.
        # Padding values will be used to crop the volumes to save.
        self.img, self.padding_values = self.pad_if_needed(self.img, True)
        self.mask = self.pad_if_needed(self.mask, False)

        # Clip last percentile of the image intensity
        p999 = np.percentile(self.img, 99.9)
        self.img[self.img > p999] = p999

    def _set_patch_coordinates_list(self):
        """
        Create the list of the starting and ending coordinates
        of all patches/subwindows of the input image to process.
        """
        def normalize_coordinates(coordinates):
            for dim in range(3):
                # Make sure that the coordinates are positive
                if coordinates[dim, 0] < 0:
                    coordinates[dim, 0] = 0
                    coordinates[dim, 1] = min(
                        self.patch_shape[dim], self.img.shape[dim])
                # Make sure the coordinates do not exceed the image size
                if coordinates[dim, 1] > self.img.shape[dim]:
                    coordinates[dim, 1] = self.img.shape[dim]
                    coordinates[dim, 0] = max(
                        0, coordinates[dim, 1] - self.patch_shape[dim])
            return coordinates
        _check_image_shape(self.img.shape)
        self.patch_coordinates_list = []
        # Get the extremal coordinates of the mask
        num_fg = np.sum(self.mask)
        assert num_fg > 0, \
            "\nWARNING! The segmentation contains only background according to the mask."
        x_fg, y_fg, z_fg = np.where(self.mask >= 1)
        x_fg_min = np.min(x_fg)
        x_fg_max = np.max(x_fg)
        y_fg_min = np.min(y_fg)
        y_fg_max = np.max(y_fg)
        z_fg_min = np.min(z_fg)
        z_fg_max = np.max(z_fg)
        # Coordinates of a window containing the fg mask
        mask_coords = np.array([
            [x_fg_min, x_fg_max],
            [y_fg_min, y_fg_max],
            [z_fg_min, z_fg_max],
        ])

        # Get the coordinates of the central patch
        x_center = int(0.5 * (int(np.max(x_fg)) + int(np.min(x_fg))))
        x_min = max(x_center - self.patch_shape[0] // 2, 0)
        x_max = x_min + self.patch_shape[0]
        y_center = int(0.5 * (int(np.max(y_fg)) + int(np.min(y_fg))))
        y_min = max(y_center - self.patch_shape[1] // 2, 0)
        y_max = y_min + self.patch_shape[1]
        z_center = int(0.5 * (int(np.max(z_fg)) + int(np.min(z_fg))))
        z_min = max(z_center - self.patch_shape[2] // 2, 0)
        z_max = z_min + self.patch_shape[2]
        patch_coords_center = np.array([
            [x_min, x_max],
            [y_min, y_max],
            [z_min, z_max],
        ])
        patch_coords_center = normalize_coordinates(patch_coords_center)
        self.patch_coordinates_list.append(patch_coords_center)

        # Add other patches around the central patch if needed
        for dim in range(3):
            new_patches_to_add = []
            # Check if we need to add patches with a shift on the left along this dimension
            if mask_coords[dim, 0] < patch_coords_center[dim, 0]:
                # We loop over all existing patches
                # this allows to create all the combination of shifts
                # necessary to cover all the mask
                for existing_patch in self.patch_coordinates_list:
                    new_patch_coords = np.copy(existing_patch)
                    new_patch_coords[dim, 0] -= self.patch_shape[dim] // 2
                    new_patch_coords[dim, 1] = new_patch_coords[dim, 0] + self.patch_shape[dim]
                    new_patch_coords = normalize_coordinates(new_patch_coords)
                    new_patches_to_add.append(new_patch_coords)
            # Check if we need to add patches with a shift on the right along this dimension
            if mask_coords[dim, 1] > patch_coords_center[dim, 1]:
                for existing_patch in self.patch_coordinates_list:
                    new_patch_coords = np.copy(existing_patch)
                    new_patch_coords[dim, 0] += self.patch_shape[dim] // 2
                    new_patch_coords[dim, 1] = new_patch_coords[dim, 0] + self.patch_shape[dim]
                    new_patch_coords = normalize_coordinates(new_patch_coords)
                    new_patches_to_add.append(new_patch_coords)
            for patch in new_patches_to_add:
                self.patch_coordinates_list.append(patch)

        print('Found %d patches to process' % len(self.patch_coordinates_list))

    def __getitem__(self, index=0):
        """
        :param index: int; index of the patch to return.
        :return: patch/subwindow to segment.
        """
        coords = self.patch_coordinates_list[index]
        img = self.img[coords[0,0]:coords[0,1],coords[1,0]:coords[1,1],coords[2,0]:coords[2,1]]
        mask = self.mask[coords[0,0]:coords[0,1],coords[1,0]:coords[1,1],coords[2,0]:coords[2,1]]
        global_coords = np.copy(coords)

        # Whitening of the patch image
        assert np.abs(img).sum() != 0, "The image is black. " \
                                       "There must be a problem with " \
                                       "your image %s." % self.img_path
        img = img - np.mean(img)
        img = img / np.std(img)

        # Convert to torch tensor
        img = np.expand_dims(img, axis=0)  # Add chanel dimension
        # img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = torch.from_numpy(img)

        return img, mask, global_coords

    def __len__(self):
        """
        Number of patches for the input volume.
        """
        return len(self.patch_coordinates_list)

    def _put_score_map_in_image_space(self, pred_score_map):
        assert pred_score_map.ndim == 4, \
            "Only 4D volumes (n_chan, x_dim, y_dim, z_dim) supported here."
        # Crop the volume to be saved if the image was padded.
        # if the image was padded there is a shift between
        # the predicted volume and the original image
        crop = np.copy(self.padding_values)
        crop[:, 1] = np.array(pred_score_map.shape[1:]) - self.padding_values[:, 1]
        return pred_score_map[:,crop[0,0]:crop[0,1],crop[1,0]:crop[1,1],crop[2,0]:crop[2,1]]


    def put_in_image_space(self, pred_volume):
        volume = np.squeeze(pred_volume)
        if volume.ndim == 4:  # score map
            return self._put_score_map_in_image_space(volume)
        else:
            # Crop the volume to be saved if the image was padded.
            # if the image was padded there is a shift between
            # the predicted volume and the original image
            crop = np.copy(self.padding_values)
            crop[:, 1] = np.array(volume.shape) - self.padding_values[:, 1]
            volume_ori_img_space = volume[crop[0,0]:crop[0,1], crop[1,0]:crop[1,1], crop[2,0]:crop[2,1]]
            return volume_ori_img_space

    def save(self, pred_volume, save_path):
        """
        Put in image space and save
        :param pred_volume:
        :param save_path:
        :return:
        """
        volume_ori_img_space = self.put_in_image_space(pred_volume)
        if volume_ori_img_space.ndim == 4:  # for score or proba maps
            shape = volume_ori_img_space.shape
            if shape[0] == np.min(shape):
                # Put the channel dimension at the end
                volume_ori_img_space = np.transpose(volume_ori_img_space, axes=(1, 2, 3, 0))
        # Create the nifti image and save
        pred_volume_nii = nib.Nifti1Image(volume_ori_img_space, self.affine, self.header)
        nib.save(pred_volume_nii, save_path)
