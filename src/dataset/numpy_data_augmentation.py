import numpy as np
import torch
import torch.nn as nn


class NumpyDataAugmentation:
    """
    Base class for the data augmentation operations that transform
    the intensity of a single image.
    Transformation are applied to each channel independently.
    The function per_channel_transform must be implemented.
    """
    def __init__(self, proba):
        # Proba to apply the transformation to a given channel of a sample
        self.proba = proba

    def check_data(self, array):
        # The input array must be a 5D numpy array
        # num channels x x_dim x y_dim x z_dim
        assert len(array.shape) == 4, "Need a 4D numpy array."

    def draw_parameter_in_range(self, range=(0,1)):
        if range[0] == range[1]:
            param = range[0]
        else:
            param = np.random.uniform(range[0], range[1])
        return param

    def per_channel_transform(self, chan_img):
        # in-place transformation
        raise NotImplementedError

    def __call__(self, img, mask=None):
        self.check_data(img)
        for chan_idx in range(img.shape[0]):
            # Randomly apply the transformation to each channel independently
            if np.random.uniform() <= self.proba:
                if mask is not None:
                    img[chan_idx, mask > 0] = self.per_channel_transform(
                        img[chan_idx, mask > 0])
                else:
                    img[chan_idx, ...] = self.per_channel_transform(
                        img[chan_idx, ...])
        return img


class AdditiveGaussianNoise(NumpyDataAugmentation):
    def __init__(self, std_interval=(0, 0.1), proba=0.15):
        """
        Additive Gaussian noise data augmentation.
        The standard deviation (std) of the noise is drawn uniformly
        in the interval std_interval.
        Different std values are drawn for different channels.
        :param std_interval:
        :param proba: float; between 0 and 1.
        Probability to apply the augmentation to each sample.
        """
        super(AdditiveGaussianNoise, self).__init__(proba)
        assert std_interval[0] <= std_interval[1]
        self.std_interval = std_interval

    def per_channel_transform(self, chan_img):
        std = self.draw_parameter_in_range(self.std_interval)
        noise = np.random.normal(0.0, std, size=chan_img.shape)
        chan_img += noise
        return chan_img


class Gamma(NumpyDataAugmentation):
    def __init__(self, power_range=(0.7, 1.5), invert_intensities=False, proba=0.3):
        super(Gamma, self).__init__(proba)
        assert power_range[0] <= power_range[1]
        self.power_range = power_range
        self.invert_intensities = invert_intensities

    def per_channel_transform(self, chan_img):
        if np.random.random() < 0.5 and self.power_range[0] < 1:
            range = (self.power_range[0], 1)
        else:
            range = (max(self.power_range[0], 1), self.power_range[1])
        power = self.draw_parameter_in_range(range)
        if self.invert_intensities:
            chan_img *= -1
        # Scale the img to [0, 1]
        min_img = np.min(chan_img)
        max_img = np.max(chan_img)
        mean_img = np.mean(chan_img)
        std_img = np.std(chan_img)
        range = max_img - min_img
        chan_img = (chan_img - min_img) / (range + 1e-7)
        # Apply the gamma transformation
        chan_img = np.power(chan_img, power)
        # Rescale
        chan_img = (chan_img * range) + min_img
        # Preserve mean and std of the image before transformation
        chan_img -= np.mean(chan_img)
        chan_img *= std_img / np.std(chan_img)
        chan_img += mean_img
        if self.invert_intensities:
            chan_img *= -1
        return chan_img


class Contrast(NumpyDataAugmentation):
    def __init__(self, multiplier_range=(0.75, 1.25), proba=0.15):
        super(Contrast, self).__init__(proba)
        assert multiplier_range[0] <= multiplier_range[1]
        self.multiplier_range = multiplier_range

    def per_channel_transform(self, chan_img):
        multi = self.draw_parameter_in_range(self.multiplier_range)
        mn = np.mean(chan_img)
        min_img = np.min(chan_img)
        max_img = np.max(chan_img)
        # Accentuate the dispersion around the mean by factor
        chan_img = (chan_img - mn) * multi + mn
        # Preserve the min and max of the image before transformation
        chan_img = np.clip(chan_img, a_min=min_img, a_max=max_img)
        return chan_img


class MultiplicativeBrightness(NumpyDataAugmentation):
    def __init__(self, multiplier_range=(0.75, 1.25), proba=0.15):
        super(MultiplicativeBrightness, self).__init__(proba)
        assert multiplier_range[0] <= multiplier_range[1]
        self.multiplier_range = multiplier_range

    def per_channel_transform(self, chan_img):
        multi = self.draw_parameter_in_range(self.multiplier_range)
        chan_img *= multi
        return chan_img


# RandomZoom does not inherit from NumpyDataAugmentation
# because it applies to img + seg together
class RandomZoom:
    def __init__(self, scale_range=(0.9, 1.1), proba=0.3):
        self.proba = proba
        self.scale_range = scale_range

    def draw_parameter_in_range(self, range=(0,1)):
        if range[0] == range[1]:
            param = range[0]
        else:
            param = np.random.uniform(range[0], range[1])
        return param

    def seg_to_one_hot(self, seg):
        one_hot = np.eye(seg.max() + 1)[np.squeeze(seg)].astype(np.float32)
        one_hot = np.transpose(one_hot, (3, 0, 1, 2))
        return one_hot

    def proba_to_seg(self, seg_proba):
        seg = np.argmax(seg_proba, axis=0)
        seg = np.expand_dims(seg, axis=0)
        return seg

    def pad_if_needed(self, volume, target_shape):
        shape = volume.shape
        num_dim = len(shape)
        need_padding = np.any(shape < target_shape)
        if not need_padding:
            return volume
        else:
            pad_list =[]
            for dim in range(num_dim):
                diff = target_shape[dim] - shape[dim]
                if diff > 0:
                    margin = diff // 2
                    pad_dim = (margin, diff - margin)
                    pad_list.append(pad_dim)
                else:
                    pad_list.append((0, 0))
            padded_array = np.pad(
                volume,
                pad_list,
                'constant',
                constant_values = [(0,0)] * num_dim,
            )
            return padded_array

    def crop_if_needed(self, volume, target_shape):
        shape = volume.shape
        need_cropping = np.any(shape > target_shape)
        if not need_cropping:
            return volume
        else:
            crop_param = []
            for dim in range(3):
                diff = shape[dim+1] - target_shape[dim+1]
                if diff > 0:
                    margin = diff // 2
                    crop_param.append([margin, shape[dim+1] - (diff - margin)])
                else:
                    crop_param.append([0, shape[dim+1]])
            crop_param = np.array(crop_param)
            out = volume[:, crop_param[0,0]:crop_param[0,1], crop_param[1,0]:crop_param[1,1], crop_param[2,0]:crop_param[2,1]]
            return out

    def fix_shape(self, volume, target_shape):
        out = self.pad_if_needed(volume, target_shape)
        out = self.crop_if_needed(out, target_shape)
        return out

    def do_zoom(self, concat_img_and_seg_np):
        scale = self.draw_parameter_in_range(self.scale_range)

        # Add batch dimension and convert to torch tensor
        input_torch = torch.from_numpy(
            np.expand_dims(concat_img_and_seg_np, axis=0)
        )
        if torch.cuda.is_available():
            input_torch = input_torch.cuda()
        out_torch = nn.functional.interpolate(
            input_torch,
            scale_factor=scale,
            mode='trilinear',
            align_corners=False,
        )
        out_np = out_torch.cpu().numpy()
        out_np = out_np[0, ...]
        return out_np

    def __call__(self, img, seg):
        if np.random.uniform() <= self.proba:
            shape = img.shape  # (n_chan, x_dim, y_dim, z_dim)
            # Convert the segmentation to one hot encoding
            one_hot = self.seg_to_one_hot(seg)
            concat = np.concatenate([img, one_hot], axis=0)
            # Apply the zoom
            zoom_concat = self.do_zoom(concat)
            zoom_img = zoom_concat[:shape[0], ...]
            zoom_one_hot = zoom_concat[shape[0]:, ...]
            zoom_seg = self.proba_to_seg(zoom_one_hot)
            # Crop or pad if needed
            zoom_img = self.fix_shape(zoom_img, shape)
            zoom_seg = self.fix_shape(zoom_seg, shape)
            zoom_img = np.copy(zoom_img, order='C').astype(np.float32)
            zoom_seg = np.copy(zoom_seg, order='C').astype(np.uint8)
        else:
            zoom_img = img
            zoom_seg = seg
        return zoom_img, zoom_seg
