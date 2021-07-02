from __future__ import division
import csv
import os
import numpy as np
import nibabel as nib
import torch
from scipy.ndimage.morphology import binary_dilation
from src.utils.misc import pad_if_needed


class Fetal3DSegDataPathDataset(object):
    def __init__(self, data_csv, patch_size):
        self.samples_id_list = []  # sample ids to use
        self.samples_info_dict = {}  # info about img and seg path
        self.current_img_name = None
        if isinstance(patch_size, int):
            self.patch_size = [patch_size] * 3
        else:
            self.patch_size = patch_size
        # Get the samples info
        with open(data_csv, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            for row in csv_reader:
                id = row[0]
                self.samples_id_list.append(id)
                img_path = row[1]
                seg_path = row[2]
                mask_path = os.path.join(
                    os.path.split(img_path)[0],
                    'mask.nii.gz'
                )
                self.samples_info_dict[id] = {
                    'image_path': img_path,
                    'mask_path': mask_path,
                    'label_path': seg_path,
                }
        print("Found %d samples in %s" % (len(self.samples_id_list), data_csv))

    def __getitem__(self, index):
        """
        :param index: index (int): Index
        """
        sample_id = self.samples_id_list[index]
        img_path = self.samples_info_dict[sample_id]['image_path']
        seg_path = self.samples_info_dict[sample_id]['label_path']
        mask_path = self.samples_info_dict[sample_id]['mask_path']
        return img_path, seg_path, mask_path, sample_id


# Usual PyTorch dataset (this is a data reader)
class Fetal3DSegEvaluationDataset(object):
    def __init__(self, data_csv, patch_size):
        self.samples_id_list = []  # sample ids to use
        self.samples_info_dict = {}  # info about img and seg path
        self.current_img_name = None
        self.patch_size = patch_size
        # Get the samples info
        with open(data_csv, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            for row in csv_reader:
                id = row[0]
                self.samples_id_list.append(id)
                img_path = row[1]
                seg_path = row[2]
                mask_path = os.path.join(
                    os.path.split(img_path)[0],
                    'mask.nii.gz'
                )
                self.samples_info_dict[id] = {
                    'image_path': img_path,
                    'mask_path': mask_path,
                    'label_path': seg_path,
                }
        print("Found %d samples in %s" % (len(self.samples_id_list), data_csv))

    def cast(self, params, dtype='float'):
        # Change tensor type and put them on the appropriate device
        if isinstance(params, dict):
            return {k: cast(v, dtype) for k, v in params.items()}
        else:
            return getattr(
                params.cuda() if torch.cuda.is_available() else params,
                dtype)()

    def __getitem__(self, index):
        """
        :param index: index (int): Index
        """
        self.current_sample_id = self.samples_id_list[index]
        # Load the image as a numpy array and match PyTorch convention 'CHWD'
        img_path = self.samples_info_dict[self.current_sample_id]['image_path']
        img = nib.load(img_path).get_data()
        if img.ndim == 4:
            img = img[:,:,:,0]
        img = pad_if_needed(img)
        # deep copy as a unit-strided data
        img = np.copy(img, order='C').astype(np.float32)
        # add channel and batch dimension
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=0)

        # Load the segmentation as a numpy array
        seg_path = self.samples_info_dict[self.current_sample_id]['label_path']
        seg = nib.load(seg_path).get_data()
        seg = pad_if_needed(seg)
        # Deep copy as a unit-strided data
        seg = np.copy(seg, order='C').astype(np.uint8)
        # Add channel and batch dimension
        seg = np.expand_dims(seg, axis=0)
        seg = np.expand_dims(seg, axis=0)

        # Load the mask
        mask_path = self.samples_info_dict[self.current_sample_id]['mask_path']
        mask = nib.load(mask_path).get_data()
        mask = pad_if_needed(mask)
        mask = np.copy(mask, order='C').astype(np.uint8)

        # Dilate the mask
        mask = binary_dilation(mask, iterations=3)
        mask = np.expand_dims(mask, axis=0)
        mask = np.expand_dims(mask, axis=0)
        # skull stripping
        img[mask == 0] = 0.

        # Clip the last percentile
        p999 = np.percentile(img, 99.9)
        img[img > p999] = p999

        # Extract the central patch
        ori_shape = np.array(img.shape[2:])

        def normalize_coordinates(coordinates):
            for dim in range(3):
                # Make sure that the coordinates are positive
                if coordinates[dim, 0] < 0:
                    coordinates[dim, 0] = 0
                    coordinates[dim, 1] = min(
                        self.patch_size[dim], ori_shape[dim])
                # Make sure the coordinates do not exceed the image size
                if coordinates[dim, 1] > ori_shape[dim]:
                    coordinates[dim, 1] = ori_shape[dim]
                    coordinates[dim, 0] = max(
                        0, coordinates[dim, 1] - self.patch_size[dim])
            return coordinates

        # Get the extremal coordinates of the mask
        num_fg = np.sum(mask)
        assert num_fg > 0, "The segmentation contains only background."
        x_fg, y_fg, z_fg = np.where(mask[0,0,...] >= 1)
        x_center = int(0.5 * (int(np.max(x_fg)) + int(np.min(x_fg))))
        x_min = max(x_center - self.patch_size[0] // 2, 0)
        x_max = x_min + self.patch_size[0]
        y_center = int(0.5 * (int(np.max(y_fg)) + int(np.min(y_fg))))
        y_min = max(y_center - self.patch_size[1] // 2, 0)
        y_max = y_min + self.patch_size[1]
        z_center = int(0.5 * (int(np.max(z_fg)) + int(np.min(z_fg))))
        z_min = max(z_center - self.patch_size[2] // 2, 0)
        z_max = z_min + self.patch_size[2]
        patch_coords = np.array([
            [x_min, x_max],
            [y_min, y_max],
            [z_min, z_max],
        ])
        normalize_coordinates(patch_coords)
        coord_min = patch_coords[:, 0]
        coord_max = patch_coords[:, 1]

        # Extract patch
        img_patch = img[:,:,coord_min[0]:coord_max[0],coord_min[1]:coord_max[1],coord_min[2]:coord_max[2]]
        seg_patch = seg[:,:,coord_min[0]:coord_max[0],coord_min[1]:coord_max[1],coord_min[2]:coord_max[2]]
        mask_patch = mask[:,:,coord_min[0]:coord_max[0],coord_min[1]:coord_max[1],coord_min[2]:coord_max[2]]

        # Normalize the image
        img_patch -= img_patch.mean()
        img_patch /= img_patch.std()

        # Convert to pytorch tensor
        img_torch = self.cast(torch.from_numpy(img_patch), 'float')
        seg_torch = self.cast(torch.from_numpy(seg_patch), 'long')
        mask_torch = self.cast(torch.from_numpy(mask_patch), 'long')

        return img_torch, seg_torch, mask_torch, coord_min, coord_max, index

    def get_sample_id(self, index):
        return self.samples_id_list[index]

    def get_sample_img_path(self, index):
        id = self.get_sample_id(index)
        img_path = self.samples_info_dict[id]['image_path']
        return img_path

    def __len__(self):
        return len(self.samples_id_list)


class Fetal3dSegEvaluationIterator:
    def __init__(
            self,
            dataset,
            batch_idx_sampler,
            output_map=['img', 'seg', 'mask', 'coord_min', 'coord_max', 'idx']
    ):
        self.dataset = dataset
        self.batch_idx_sampler = batch_idx_sampler
        self.output_map = output_map

    def __iter__(self):
        for id_list in self.batch_idx_sampler:
            # id_list = self._next_index()  # may raise StopIteration
            img_list = []
            seg_list = []
            mask_list = []
            coord_min_list = []
            coord_max_list = []
            for id in id_list:
                img, seg, mask, coord_min, coord_max, index = self.dataset[id]
                img_list.append(img)
                seg_list.append(seg)
                mask_list.append(mask)
                coord_min_list.append(coord_min)
                coord_max_list.append(coord_max)
            # Concatenate images, segmentations and masks
            img_batch = torch.cat(img_list)  # miaou
            seg_batch = torch.cat(seg_list)
            mask_batch = torch.cat(mask_list)
            # Create the output dictionary
            output = {
                self.output_map[0]: img_batch,
                self.output_map[1]: seg_batch,
                self.output_map[2]: mask_batch,
                self.output_map[3]: coord_min_list,
                self.output_map[4]: coord_max_list,
                self.output_map[5]: id_list,
            }
            yield [output]
