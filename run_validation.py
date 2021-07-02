"""
@brief  PyTorch validation code for 3D segmentation.

@author Lucas Fidon (lucas.fidon@kcl.ac.uk)
@date   July 2021.
"""

import argparse
import os
import json
import numpy as np
import csv
from tqdm import tqdm
import torch
import torch.utils.data
import nibabel as nib
from run_train import get_loss
from src.dataset.dataset_evaluation import Fetal3DSegDataPathDataset
from src.evaluation_metrics.segmentation_metrics import dice_score, haussdorff_distance
from infer_seg import segment, get_network, create_image_loader


parser = argparse.ArgumentParser(description='Run validation for segmentation')
# Model options
parser.add_argument('--start_iter', default=-1, type=int)
parser.add_argument('--patch_size', default='[144,160,144]', type=str)
parser.add_argument('--flip', action='store_true')
parser.add_argument('--dataroot', default='.', type=str)
parser.add_argument('--dtype', default='float', type=str)
parser.add_argument('--groups', default=1, type=int)
parser.add_argument('--nthread', default=4, type=int)

# Device options
parser.add_argument('--delete', action='store_true')
parser.add_argument('--save', default='./logs/test_fetal3d_seg', type=str,
                    help='save parameters and logs in this folder')

# Data
parser.add_argument('--valid_data_csv',
                    default = os.path.join(
                        '/data',
                        'fetal_brain_srr_parcellation/validation.csv',
                    ),
                    type=str,
                    help='path to a csv file that maps sample id '
                         'to image path and segmentation path')


def read_logs(path):
    logs = []
    with open(path, 'r') as f:
        for line in f:
            line = line.replace('\n', '')
            line = line.replace('json_stats: ', '')
            data = json.loads(line)
            logs.append(data)
    return logs


def create_path_dataset(opt):
    """
    Create the dataset for the paths.
    Samples are of the form:
    (img_path, seg_path, mask_path, patient ID)
    :param opt: dict of parsed command line arguments.
    :return: PyTorch dataset.
    """
    return Fetal3DSegDataPathDataset(
        data_csv=opt.valid_data_csv,
        patch_size=json.loads(opt.patch_size)
    )


def main(opt):
    """
    Run inference for a network that have been trained with main_seg_fetal3d.py.
    :param opt: parsed command line arguments.
    """
    def compute_loss(pred_score, seg):
        # Add batch dimension
        pred_resized = np.expand_dims(pred_score, axis=0)
        seg_resized = np.expand_dims(seg, axis=0)
        y = torch.tensor(pred_resized).float().cuda()
        target = torch.tensor(seg_resized).long().cuda()
        return loss_func(y, target).cpu().numpy()

    def aggregate_metrics(pred_score, seg, pat_id, patch_loader):
        # pred_score and seg are assume to be in the original image space
        loss = float(compute_loss(pred_score, seg))
        loss_val.append(loss)
        # Predicted segmentation
        pred = np.argmax(
            pred_score,
            axis=0,
        )
        # Compute dice score values for all the classes
        dice_values = [
            dice_score(pred, seg, fg_class=c)
            for c in range(num_classes)
        ]
        dice_val.append(dice_values)
        # Compute the Hausdorff distance for all classes
        hausdorff_dist_values = [
            haussdorff_distance(seg, pred, fg_class=c, percentile=95)
            for c in range(num_classes)
        ]
        hausdorff_val.append(hausdorff_dist_values)
        # Patient logs
        pat_logs[pat_id] = {
            'loss': loss,
        }
        for c in range(num_classes):
            pat_logs[pat_id]['dice_class_%d' % c] = dice_values[c]
            pat_logs[pat_id]['hausdorff_class_%d' % c] = hausdorff_dist_values[c]
        # Save the prediction for the last epoch
        # Create the save folder
        save_folder = os.path.join(opt.save, 'inference_valid_set_iter%d' % iter)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        save_path = os.path.join(save_folder, '%s_parcellation_autoseg.nii.gz' % pat_id)
        affine = patch_loader.dataset.affine
        header = patch_loader.dataset.header
        pred_nii = nib.Nifti1Image(pred, affine, header)
        nib.save(pred_nii, save_path)

    def compute_global_metrics_and_save(iter):
        metrics = {}
        metrics['iter'] = iter
        loss_np = np.array(loss_val).astype(float)
        dice_np = np.array(dice_val).astype(float)
        hausdorff_np = np.array(hausdorff_val).astype(float)
        # Save stats about the loss
        # mean and std
        metrics['loss_mean'] = np.mean(loss_np)
        metrics['loss_std'] = np.std(loss_np)
        # median and mad (median absolute difference)
        metrics['loss_median'] = np.median(loss_np)
        metrics['loss_mad'] = np.median(
            np.abs(loss_np - np.median(loss_np))
        )
        # quartiles
        q1 = np.percentile(loss_np, 25)
        q3 = np.percentile(loss_np, 75)
        metrics['loss_p75'] = q3
        metrics['loss_p25'] = q1
        # min and max
        metrics['loss_max'] = np.max(loss_np)
        metrics['loss_min'] = np.min(loss_np)
        # Loss percentiles
        for perc in [1, 5, 10, 15, 20, 80, 85, 90, 95, 99]:
            metrics['loss_p%d' % perc] = np.percentile(loss_np, perc)
        # Save stats about the dice scores
        for c in range(1, num_classes):  # skip the background
            dice_class_c = dice_np[:,c]
            # mean and std
            metrics['dice_class_%d_mean' % c] = float(np.mean(dice_class_c))
            metrics['dice_class_%d_std' % c] = float(np.std(dice_class_c))
            # median and mad (median absolute difference)
            metrics['dice_class_%d_median' % c] = np.median(dice_class_c)
            metrics['dice_class_%d_mad' % c] = np.median(
                np.abs(dice_class_c - np.median(dice_class_c))
            )
            # quartiles
            q1 = float(np.percentile(dice_class_c, 25))
            q3 = float(np.percentile(dice_class_c, 75))
            metrics['dice_class_%d_p75' % c] = q3
            metrics['dice_class_%d_p25' % c] = q1
            metrics['dice_class_%d_IQ' % c] = q3 - q1
            # John Tuckey's method to define outliers (one-sided)
            outliers_rate = np.mean(dice_class_c <= q1 -
                                1.5 * (q3 - q1)) * 100  # in percentage
            metrics['dice_class_%d_outliers_rate' % c] = outliers_rate
            # min and max
            metrics['dice_class_%d_max' % c] = float(np.max(dice_class_c))
            metrics['dice_class_%d_min' % c] = float(np.min(dice_class_c))
            for perc in [1, 5, 10, 15, 20, 80, 85, 90, 95, 99]:
                metrics['dice_class_%d_p%d' % (c, perc)] = np.percentile(dice_class_c, perc)
        # Save stats about the Hausdorff values
        for c in range(1, num_classes):
            hausdorff_c = hausdorff_np[:,c]
            # mean and std
            metrics['hausdorff_%d_mean' % c] = float(np.mean(hausdorff_c))
            metrics['hausdorff_%d_std' % c] = float(np.std(hausdorff_c))
            # median and mad (median absolute difference)
            metrics['hausdorff_%d_median' % c] = np.median(hausdorff_c)
            metrics['hausdorff_%d_mad' % c] = np.median(
                np.abs(hausdorff_c - np.median(hausdorff_c))
            )
            # quartiles
            q1 = float(np.percentile(hausdorff_c, 25))
            q3 = float(np.percentile(hausdorff_c, 75))
            metrics['hausdorff_%d_p75' % c] = q3
            metrics['hausdorff_%d_p25' % c] = q1
            metrics['hausdorff_%d_IQ' % c] = q3 - q1
            # John Tuckey's method to define outliers (one-sided)
            outliers_rate = np.mean(hausdorff_c <= q1 -
                                1.5 * (q3 - q1)) * 100  # in percentage
            metrics['hausdorff_%d_outliers_rate' % c] = outliers_rate
            # min and max
            metrics['hausdorff_%d_max' % c] = float(np.max(hausdorff_c))
            metrics['hausdorff_%d_min' % c] = float(np.min(hausdorff_c))
            for perc in [1, 5, 10, 15, 20, 80, 85, 90, 95, 99]:
                metrics['hausdorff_%d_p%d' % (c, perc)] = np.percentile(hausdorff_c, perc)
        # Save the logs
        # All dice score values
        save_name = 'dice_scores_valid_iter%d' % iter
        np.save(
            os.path.join(opt.save, save_name),
            dice_np,
        )
        # All hausdorff distance values
        save_name = 'hausdorff_valid_iter%d' % iter
        np.save(
            os.path.join(opt.save, save_name),
            hausdorff_np,
        )
        # Save csv with logs about each study
        csv_name = os.path.join(opt.save, 'eval_valid_pat_iter%d.csv' % iter)
        columns = ['study', 'loss'] \
                  + ['dice_class_%d' % c for c in range(num_classes)]\
                  + ['hausdorff_class_%d' % c for c in range(num_classes)]
        with open(csv_name, mode='w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(columns)
            for study in list(pat_logs.keys()):
                writer.writerow([study] + [pat_logs[study][metric] for metric in columns[1:]])
        z = {**vars(opt), **metrics}
        # Add the evaluation logs for epoch t
        with open(os.path.join(opt.save, 'eval_valid.txt'), 'a') as flog:
            flog.write('json_stats: ' + json.dumps(z) + '\n')
        print(z)

    def restore(model_path):
        state_dict = torch.load(model_path)
        network.load_state_dict(state_dict['params'])

    def main_eval_one_iter(data_path_loader, network_model, iter_snapshot):
        # Run evaluation on all data for one snapshot
        for sample in tqdm(data_path_loader, dynamic_ncols=True):
            img_path = sample[0]
            seg_path = sample[1]
            mask_path = sample[2]
            pat_id = sample[3]
            # Create the patch loader for the current case
            patch_loader = create_image_loader(
                img_path=img_path,
                mask_path=mask_path,
                patch_size=json.loads(opt.patch_size),
            )
            # Get the output as numpy array
            score_map = segment(
                img_loader=patch_loader,
                network=network_model,
                num_class=num_classes,
            )
            # We need to ask the patch loader to put the score map
            # back into the space of the original image
            score_map_img_space = patch_loader.dataset.put_in_image_space(
                score_map
            )
            # Load the ground-truth segmentation (image space)
            seg = nib.load(seg_path).get_data().astype(np.uint8)
            # compute and aggregate metrics for this prediction
            aggregate_metrics(score_map_img_space, seg, pat_id, patch_loader)
        compute_global_metrics_and_save(iter_snapshot)

    # Restore hyperparameters
    log_path = os.path.join(opt.save, 'log.txt')
    assert os.path.exists(log_path), "Cannot found the model %s" % log_path
    log = read_logs(log_path)[-1]
    every_n = log['save_every_n_iter']
    model = log['model']
    assert model in ['unet'], "Only U-Net is supported for now."
    num_chanels = 1
    num_classes = log['num_classes']
    norm = 'instance'
    loss_name = log['loss']
    try:
        norm = log['norm']
    except:
        print('norm argument not found in logs')
        print('Use instance normalization by default')

    # Create the network
    network = get_network(num_chanels, num_classes, norm=norm)

    # Create the loss function
    loss_func = get_loss(loss_name=loss_name)

    trainable_model_parameters = filter(
        lambda p: p.requires_grad, network.parameters())
    n_parameters = sum([np.prod(p.size()) for p in trainable_model_parameters])
    print('\nTotal number of parameters:', n_parameters)

    if opt.flip:
        print('RL flipping at inference time is used')

    # Maintain values of all the per-example criteria for logs
    dice_val = []
    hausdorff_val = []
    loss_val = []
    pat_logs = {}

    iter = opt.start_iter
    if iter == -1:
        iter = every_n
    model_path = os.path.join(opt.save, 'model_iter%d.pt7' % iter)
    while os.path.exists(model_path):
        restore(model_path)
        # Run eval on validation set
        dice_val = []
        hausdorff_val = []
        loss_val = []
        pat_logs = {}
        valid_data_path_loader = create_path_dataset(opt)
        main_eval_one_iter(valid_data_path_loader, network, iter)
        iter += every_n
        model_path = os.path.join(opt.save, 'model_iter%d.pt7' % iter)


if __name__ == '__main__':
    opt = parser.parse_args()
    print('parsed options:', vars(opt))
    assert os.path.exists(opt.save), "Cannot found the log folder %s" % opt.save

    # Filename where to save validation results
    eval_valid_path = os.path.join(opt.save, 'eval_valid.txt')

    # Filename where to save training results
    eval_train_path = os.path.join(opt.save, 'eval_train.txt')

    # Option to delete existing results/logs
    if opt.delete:
        print('delete previous logs')
        if os.path.exists(eval_valid_path):
            os.system('rm %s' % eval_valid_path)
        if os.path.exists(eval_train_path):
            os.system('rm %s' % eval_train_path)
    main(opt)
