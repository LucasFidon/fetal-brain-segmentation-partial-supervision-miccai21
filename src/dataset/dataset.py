from __future__ import division
import csv
import random
import numpy as np
import nibabel as nib
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from src.dataset.numpy_data_augmentation import *


# Usual PyTorch dataset (this is a data reader)
class Fetal3DSegDataset(object):
    def __init__(self, data_csv, use_data_augmentation=False, use_zoom=False):
        self.samples_id_list = []  # sample ids to use
        self.samples_info_dict = {}  # info about img and seg path
        self.current_img_name = None
        # Get the samples info
        with open(data_csv, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            for row in csv_reader:
                id = row[0]
                self.samples_id_list.append(id)
                img_path = row[1]
                seg_path = row[2]
                mask_path = row[3]
                self.samples_info_dict[id] = {
                    'image_path': img_path,
                    'seg_path': seg_path,
                    'mask_path': mask_path,
                }
        print("Found %d samples in %s" % (len(self.samples_id_list), data_csv))
        # Data augmentation
        self.use_data_augmentation = use_data_augmentation
        self.data_aug_ops = []
        if use_data_augmentation:
            print('Data intensity augmentations are used')
            gauss_noise = AdditiveGaussianNoise(proba=0.15)
            self.data_aug_ops.append(gauss_noise)
            contrast = Contrast(proba=0.3)
            self.data_aug_ops.append(contrast)
            inv_gamma = Gamma(invert_intensities=True, proba=0.1)
            self.data_aug_ops.append(inv_gamma)
            gamma = Gamma(proba=0.3)
            self.data_aug_ops.append(gamma)
        # Data augmentation for image and seg together
        self.joint_data_aug_ops = []
        self.use_join_data_augmentation = use_zoom
        if use_zoom:
            print('Random zoom is used')
            zoom_aug = RandomZoom(scale_range=(0.9, 1.1), proba=0.5)
            self.joint_data_aug_ops.append(zoom_aug)

    def __getitem__(self, index):
        """
        :param index: index (int): Index
        :return: numpy array tuple: (img, seg) where seg is the segmentation
        of the fetal brain in img.
        """
        epsilon = 1e-4
        self.current_sample_id = self.samples_id_list[index]

        # Load the image as a numpy array and match PyTorch convention 'CHWD'
        img_path = self.samples_info_dict[self.current_sample_id]['image_path']
        img = nib.load(img_path).get_data()
        if img.ndim == 4:
            img = img[:,:,:,0]
        # Deep copy as a unit-strided data (required for NVIDIA DALI)
        img = np.copy(img, order='C').astype(np.float32)

        # Load the mask
        mask_path = self.samples_info_dict[self.current_sample_id]['mask_path']
        mask = nib.load(mask_path).get_data()

        # Normalize the image to zero mean and unit variance
        img = (img - img.mean()) / (img.std() + epsilon)

        # Load the segmentation as a numpy array
        seg_path = self.samples_info_dict[self.current_sample_id]['seg_path']
        seg = nib.load(seg_path).get_data()
        # Deep copy as a unit-strided data
        seg = np.copy(seg, order='C').astype(np.uint8)

        # Add the channel dimension for img and seg
        img = np.expand_dims(img, axis=0)
        seg = np.expand_dims(seg, axis=0)

        if self.use_data_augmentation:
            for t_op in self.data_aug_ops:
                img = t_op(img, mask)

        if self.use_join_data_augmentation:
            for t_op in self.joint_data_aug_ops:
                img, seg = t_op(img, seg)

        return img, seg

    def get_sample_id(self, index):
        return self.samples_id_list[index]

    def get_sample_img_path(self, index):
        id = self.get_sample_id(index)
        img_path = self.samples_info_dict[id]['image_path']
        return img_path

    def __len__(self):
        return len(self.samples_id_list)


# Here is the DALI pre-processing+data augmentation layer.
# This can be seen as a replacement of the torch.utils.data.Dataloader class.
# This should inherit from the DALI Pipeline class.
class Fetal3DSegPipeline(Pipeline):
    """
    First version. Will work only with a batch sampler of indices.
    """
    def __init__(self, dataloader, batch_index_sampler, patch_size,
                 num_threads=1, device_id=0, do_flip=False, do_flip_all=False,
                 do_gauss_blur=False, do_zoom=False):
        super(Fetal3DSegPipeline, self).__init__(
            batch_index_sampler.batch_size,
            num_threads,
            device_id,
        )
        if isinstance(patch_size, int):
            self.patch_size = [patch_size] * 3
        else:
            self.patch_size = patch_size
        self.layout = "CDHW"
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.input_indices = ops.ExternalSource()
        # Crop operation to randomly extract a patch from the image
        self.crop = ops.Crop(
            device='cpu',  # gpu supported but not for gauss blurring...
            crop=self.patch_size,  # spatial dimension of the output
        )
        self.crop_x = ops.Uniform(device="cpu", range=[0., 1.])
        self.crop_y = ops.Uniform(device="cpu", range=[0., 1.])
        self.crop_z = ops.Uniform(device="cpu", range=[0., 1.])
        # Zoom
        self.do_zoom = do_zoom
        if do_zoom: #TODO: it does not work. One image dimension is duplicated...
            print('Random zoom is used (p=0.3)')
            print('Random zoom not supported yet with NVIDIA DALI. It will be switched off')
            self.zoom_transform = ops.ExternalSource()
            self.do_zoom = False
        self.change_layout = ops.Reinterpret(layout="DHWC", device="gpu")
        self.zoom_img = ops.WarpAffine(
            device="gpu",
            interp_type=types.DALIInterpType.INTERP_LINEAR,
            size=self.patch_size,
        )
        self.zoom_seg = ops.WarpAffine(
            device="gpu",
            interp_type=types.DALIInterpType.INTERP_NN,
            size=self.patch_size,
        )
        self.restore_layout = ops.Reinterpret(layout=self.layout, device="gpu")
        # Gaussian blur
        if do_gauss_blur:
            print('Random Gaussian blurring is used (p=0.2)')
        self.do_gauss_blur = do_gauss_blur
        self.do_gauss_blur_sample = ops.CoinFlip(probability=0.2)
        self.gauss_blur_sigma = ops.Uniform(device="cpu", range=[0.5, 1.])
        self.gauss_blur = ops.GaussianBlur(
            device="cpu",  # only supported on cpu (Sept 2020)
        )
        # Flip operations
        if do_flip:
            print('Random right-left flipping is used')
        if do_flip_all:
            print('Random flipping for all axis are used')
        self.do_flip = do_flip or do_flip_all
        self.flip = ops.Flip(device="gpu")
        self.do_flip_depth = ops.CoinFlip(probability=0.5)  # right-left flip
        if do_flip_all:  # also flipping wrt other axis
            self.do_flip_horizontal = ops.CoinFlip(probability=0.5)
            self.do_flip_vertical = ops.CoinFlip(probability=0.5)
        else:  # do not flip wrt other axis
            self.do_flip_horizontal = ops.CoinFlip(probability=0.)
            self.do_flip_vertical = ops.CoinFlip(probability=0.)
        # Cast
        self.cast_seg = ops.Cast(device="gpu", dtype=types.INT64)
        self.cast_indices = ops.Cast(device="gpu", dtype=types.INT64)
        self.batch_sampler = batch_index_sampler
        # PyTorch Sampler are iterable
        self.batch_idx_iterator = iter(self.batch_sampler)
        self.loader = dataloader

    def generate_zoom_matrix(self):
        proba_zoom = 0.3
        low_scale = 0.9
        high_scale = 1.1
        out = []
        for _ in range(self.batch_size):
            mat = np.zeros((3, 4))
            do_zoom_sample = random.random() <= proba_zoom
            if do_zoom_sample:
                scale = np.random.uniform(low=low_scale, high=high_scale, size=1)
            else:  # no augmentation
                scale = 1.
            for dim in range(3):
                mat[dim, dim] = scale
            out.append(mat.astype(np.float32))
        return out

    def define_graph(self):
        """
        This is where to define the data pre-processing / augmentation pipeline
        """
        self.imgs = self.input()
        self.segs = self.input_label()
        self.indices = self.input_indices()

        # CPU operations
        images = self.imgs
        labels = self.segs
        output_img, output_seg = self.crop(
            [images, labels],
            crop_pos_x=self.crop_x(),
            crop_pos_y=self.crop_y(),
            crop_pos_z=self.crop_z(),
        )
        if self.do_gauss_blur:
            do_gauss_blur = self.do_gauss_blur_sample()  # 0 or 1
            sigma = self.gauss_blur_sigma()
            output_img_blur = self.gauss_blur(
                output_img,
                sigma=sigma,
            )
            # dummy multiplexing because there is no condition option for GaussianBlur
            output_img = do_gauss_blur * output_img_blur + (1 - do_gauss_blur) * output_img

        # Move data to the GPU
        output_img = output_img.gpu()
        output_seg = output_seg.gpu()
        indices = self.indices.gpu()

        # GPU operations
        if self.do_zoom:
            self.matrix = self.zoom_transform()
            matrix = self.matrix.gpu()
            output_img = self.change_layout(output_img)
            output_seg = self.change_layout(output_seg)
            output_img = self.zoom_img(output_img, matrix=matrix)
            output_seg = self.zoom_seg(output_seg, matrix=matrix)
            output_img = self.restore_layout(output_img)
            output_seg = self.restore_layout(output_seg)
        if self.do_flip:
            do_flip_horizontal = self.do_flip_horizontal()
            do_flip_vertical = self.do_flip_vertical()
            do_flip_depth = self.do_flip_depth()
            output_img = self.flip(
                output_img,
                horizontal=do_flip_horizontal,
                vertical=do_flip_vertical,
                depthwise=do_flip_depth
            )
            output_seg = self.flip(
                output_seg,
                horizontal=do_flip_horizontal,
                vertical=do_flip_vertical,
                depthwise=do_flip_depth
            )
        output_seg = self.cast_seg(output_seg)
        output_indices = self.cast_indices(indices)
        return output_img, output_seg, output_indices

    def _next_index(self):
        """
        :return: int list; next batch of sample indices.
        """
        return next(self.batch_idx_iterator)  # may raise StopIteration

    def _next_batch(self):
        """
        :param idx: int list
        :return: numpy array tuple list; next batch of paired (img, seg)
        """
        id_batch = []
        img_batch = []
        seg_batch = []
        id_list = self._next_index()
        for id in id_list:
            img, seg = self.loader[id]  # numpy arrays after normalization
            img_batch.append(img)
            seg_batch.append(seg)
            id_batch.append(np.array([id], dtype=np.int64))
        return img_batch, seg_batch, id_batch

    def iter_setup(self):
        try:
            # Get a new batch
            (images, labels, indices) = self._next_batch()  # may raise StopIteration
            # Feed the batch to the DALI pipeline
            self.feed_input(self.imgs, images, layout=self.layout)
            self.feed_input(self.segs, labels, layout=self.layout)
            self.feed_input(self.indices, indices)
            if self.do_zoom:
                # Since we are using ExternalSource, we need to feed the
                # externally provided zoom transformation matrix to the pipeline
                self.feed_input(self.matrix, self.generate_zoom_matrix())
        except StopIteration:
            # Just reset the iterator
            self.batch_idx_iterator = iter(self.batch_sampler)
            # Get a new batch
            (images, labels, indices) = self._next_batch()  # may raise StopIteration
            # Feed the batch to the DALI pipeline
            self.feed_input(self.imgs, images, layout=self.layout)
            self.feed_input(self.segs, labels, layout=self.layout)
            self.feed_input(self.indices, indices)
            if self.do_zoom:
                self.feed_input(self.matrix, self.generate_zoom_matrix())
