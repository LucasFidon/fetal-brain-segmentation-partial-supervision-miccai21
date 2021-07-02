"""
@brief  The class InferenceImage aims at managing the aggregation
        of the patches prediction into a volume in the same space
        as the original input volume.

@author Lucas Fidon (lucas.fidon@kcl.ac.uk)
@date   July 2021.
"""

import numpy as np

SUPPORTED_FUSION = ['uniform', 'gaussian']


def _get_fusion_kernel(patch_size, fusion='gaussian', margin=0):
    """
    Return a 3D kernel with the same size as a patch
    that will be used to assign weights to each voxel of a patch
    during the patch-based predictions aggregation.
    :param patch_size: int or tuple; size of the patch
    :param fusion: string; type of fusion.
    :param margin: int; if margin > 0, mask the prediction at
    a distance less margin to the border. Default value is 0.
    If you the 'gaussian' mode you should not need to use
    margin > 0.
    """
    if isinstance(patch_size, int):
        shape = (patch_size, patch_size, patch_size)
    else:
        shape = patch_size
    assert np.all(np.array(shape) > 2 * margin), \
        "Margin %d is too large for patch size %s" % (margin, str(shape))
    # Create the kernel
    if fusion == 'uniform': # Uniform kernel (default)
        kernel = np.ones(shape)
    elif fusion == 'gaussian':  # Gaussian kernel
        # Define the gaussian kernel
        sigma = 1.
        dist_border_center = 3 * sigma
        x, y, z = np.meshgrid(
            np.linspace(-dist_border_center, dist_border_center, shape[1]),
            np.linspace(-dist_border_center, dist_border_center, shape[0]),
            np.linspace(-dist_border_center, dist_border_center, shape[2]),
        )
        d = x*x + y*y + z*z
        kernel = np.exp(-d / (2. * sigma**2))
    else:
        error_msg = "Only the fusion strategy %s are supported. Received %s" % \
            (str(SUPPORTED_FUSION), fusion)
        raise ArgumentError(error_msg)
    # (optional) Set the contribution of voxels at distance less than margin
    # to the border of the patch to 0
    if margin > 0:
        kernel[:margin, :, :] = 0.
        kernel[-margin:, :, :] = 0.
        kernel[:, :margin, :] = 0.
        kernel[:, -margin:, :] = 0.
        kernel[:, :, :margin] = 0.
        kernel[:, :, -margin:] = 0.
    return kernel


class InferenceImage:
    def __init__(
            self,
            shape,
            patch_size,
            fusion='gaussian',
            margin=0,
            img_type='float',
            default_val=0,
    ):
        assert len(shape) == 4, \
            "only 3D application with shape of the form " \
            "(num_class, x_dim, y_dim, z_dim) are supported." \
            "Found shape: %s" % str(shape)
        # Volume in which the score maps for the different classes
        # will be aggregated for the different patches
        self.image = default_val * np.ones(shape).astype(img_type)  # c,x,y,z
        self.fusion_kernel = _get_fusion_kernel(patch_size, fusion, margin)
        # Map of cumulated weights of the previous patches fused into self.image
        self.count_map = np.zeros(shape[1:])  # x,y,z

    def add_patch(self, patch, patch_coord):
        """
        Aggregate a new batch of patches prediction to the inferred image.
        :param patch: 5d tensor of shape
        (num_patch, num_channels, x_dim, y_dim, z_dim)
        """
        batch_size = patch.shape[0]
        epsilon = 1e-7
        for b in range(batch_size):
            patch_np = patch.cpu().numpy()[b, :, :, :, :]  # c,x,y,z
            coord = patch_coord[b, :, :]

            # Sanity check of the coordinates
            assert coord[0, 1] <= self.count_map.shape[0], \
                'coord %d exceed the size of the image for the x-axis %d' % \
                (coord[0, 1], self.count_map.shape[0])
            assert coord[1, 1] <= self.count_map.shape[1], \
                'coord %d exceed the size of the image for the y-axis %d' % \
                (coord[1, 1], self.count_map.shape[1])
            assert coord[2, 1] <= self.count_map.shape[2], \
                'coord %d exceed the size of the image for the z-axis %d' % \
                (coord[2, 1], self.count_map.shape[2])

            # Create a mask corresponding to the positions of the patch in the inferred image.
            patch_mask = np.zeros_like(self.count_map).astype(np.bool)  # x,y,z
            patch_mask[coord[0,0]:coord[0,1],coord[1,0]:coord[1,1],coord[2,0]:coord[2,1]] = True

            # Merge the prediction for the new patch with the current inferred image.
            # This is a weighted sum of all the overlapping patches weighted by the fusion kernel
            # in the current patch prediction.
            new_patch_in_img = self.fusion_kernel[np.newaxis, :, :, :] * patch_np
            new_patch_in_img = new_patch_in_img.reshape(patch_np.shape[0], -1)
            new_patch_in_img += self.count_map[np.newaxis, patch_mask] * self.image[:, patch_mask]
            # Update the count map
            self.count_map[patch_mask] += self.fusion_kernel.flatten()
            # Normalize the weighted sum for the current patch
            new_patch_in_img /= (self.count_map[np.newaxis, patch_mask] + epsilon)
            # Update the total prediction volume
            self.image[:, patch_mask] = new_patch_in_img

    @property
    def segmentation(self):
        # Take the argmax over the scores
        pred_seg = np.argmax(self.image, axis=0)
        # Set voxels without prediction to background (label 0)
        pred_seg[self.count_map == 0] = 0
        return pred_seg

    @property
    def probability_maps(self):
        # Compute the softmax
        x = self.image.astype(float) - np.max(self.image, axis=0)
        exp_x = np.exp(x)
        pred_proba = exp_x / np.sum(exp_x, axis=0)
        # Set voxels without prediction to background (label 0)
        pred_proba[:, self.count_map == 0] = 0.
        pred_proba[0, self.count_map == 0] = 1.
        return pred_proba

    @property
    def score_maps(self):
        score = np.copy(self.image.astype(float))
        # Set voxels without prediction to background (label 0)
        # with a high score
        score[:, self.count_map == 0] = 0.
        score[0, self.count_map == 0] = 1000.
        return score
