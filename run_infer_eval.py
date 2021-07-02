import numpy as np
import nibabel as nib
import pickle
from time import time
from src.utils.definitions import *
from src.evaluation_metrics.segmentation_metrics import dice_score, haussdorff_distance

DATA_DIR = [
    Controls_LEUVEN_TESTINGSET,
    SB_LEUVEN_TESTINGSET,
    CORRECTED_ZURICH_DATA_DIR,
]
SAVE_FOLDER = '/data/saved_res_fetal'

# GENERAL EVALUATION OPTIONS
DO_EVAL = True
METRIC_NAMES = ['dice', 'hausdorff']
MAX_HD = (144. / 2.) * 0.8  # distance from the center to the border (57.6 mm)
METHOD_NAMES = ['cnn']

# MODELS with 10 different seeds and train/valid splits
FULL_DICE_ATLAS_MODEL_LIST = [  # baseline 1
    '%s/logs_DGX/fetal3dseg_atlas_atlas_only_bs3_mean_dice_beta90_fold0/model_iter2900.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_atlas_only_bs3_mean_dice_beta90_fold1/model_iter2500.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_atlas_only_bs3_mean_dice_beta90_fold2/model_iter1400.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_atlas_only_bs3_mean_dice_beta90_fold3/model_iter2800.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_atlas_only_bs3_mean_dice_beta90_fold4/model_iter2000.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_atlas_only_bs3_mean_dice_beta90_fold5/model_iter1900.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_atlas_only_bs3_mean_dice_beta90_fold6/model_iter2000.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_atlas_only_bs3_mean_dice_beta90_fold7/model_iter2500.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_atlas_only_bs3_mean_dice_beta90_fold8/model_iter2600.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_atlas_only_bs3_mean_dice_beta90_fold9/model_iter2500.pt7' % REPO_PATH,
]
PARTIAL_MARG_DICE_MODEL_LIST = [  # baseline 3
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_beta90_fold0/model_iter14200.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_beta90_fold1/model_iter12600.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_beta90_fold2/model_iter11800.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_beta90_fold3/model_iter13500.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_beta90_fold4/model_iter8100.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_beta90_fold5/model_iter17900.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_beta90_fold6/model_iter13100.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_beta90_fold7/model_iter10800.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_beta90_fold8/model_iter10800.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_beta90_fold9/model_iter13500.pt7' % REPO_PATH,
]
PARTIAL_MARG_CE_MODEL_LIST = [
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_cross_entropy_partial_beta90_fold0/model_iter8900.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_cross_entropy_partial_beta90_fold1/model_iter5000.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_cross_entropy_partial_beta90_fold2/model_iter9500.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_cross_entropy_partial_beta90_fold3/model_iter10700.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_cross_entropy_partial_beta90_fold4/model_iter8300.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_cross_entropy_partial_beta90_fold5/model_iter10600.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_cross_entropy_partial_beta90_fold6/model_iter7300.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_cross_entropy_partial_beta90_fold7/model_iter10600.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_cross_entropy_partial_beta90_fold8/model_iter6700.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_cross_entropy_partial_beta90_fold9/model_iter7500.pt7' % REPO_PATH,
]
PARTIAL_FOCAL_MODEL_LIST = [
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_focal_loss_partial_beta90_fold0/model_iter13200.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_focal_loss_partial_beta90_fold1/model_iter7700.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_focal_loss_partial_beta90_fold2/model_iter7300.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_focal_loss_partial_beta90_fold3/model_iter12000.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_focal_loss_partial_beta90_fold4/model_iter6300.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_focal_loss_partial_beta90_fold5/model_iter11600.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_focal_loss_partial_beta90_fold6/model_iter7800.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_focal_loss_partial_beta90_fold7/model_iter10100.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_focal_loss_partial_beta90_fold8/model_iter8200.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_focal_loss_partial_beta90_fold9/model_iter8700.pt7' % REPO_PATH,
]
PARTIAL_LEAF_DICE_MODEL_LIST = [  # ours
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_mean_dice_partial_masked_beta90_fold0/model_iter3400.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_mean_dice_partial_masked_beta90_fold1/model_iter2200.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_mean_dice_partial_masked_beta90_fold2/model_iter3900.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_mean_dice_partial_masked_beta90_fold3/model_iter2900.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_mean_dice_partial_masked_beta90_fold4/model_iter1300.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_mean_dice_partial_masked_beta90_fold5/model_iter3100.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_mean_dice_partial_masked_beta90_fold6/model_iter4100.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_mean_dice_partial_masked_beta90_fold7/model_iter1900.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_mean_dice_partial_masked_beta90_fold8/model_iter3000.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_mean_dice_partial_masked_beta90_fold9/model_iter4300.pt7' % REPO_PATH,
]
PARTIAL_DICE_SOFT_TARGET_MODEL_LIST = [  # baseline 2
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_mean_dice_partial_soft_labels_beta90_fold0/model_iter11100.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_mean_dice_partial_soft_labels_beta90_fold1/model_iter21300.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_mean_dice_partial_soft_labels_beta90_fold2/model_iter16300.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_mean_dice_partial_soft_labels_beta90_fold3/model_iter10900.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_mean_dice_partial_soft_labels_beta90_fold4/model_iter15900.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_mean_dice_partial_soft_labels_beta90_fold5/model_iter19900.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_mean_dice_partial_soft_labels_beta90_fold6/model_iter9900.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_mean_dice_partial_soft_labels_beta90_fold7/model_iter16400.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_mean_dice_partial_soft_labels_beta90_fold8/model_iter13300.pt7' % REPO_PATH,
    '%s/logs_DGX/fetal3dseg_atlas_partialsegv2_bs3_mean_dice_partial_soft_labels_beta90_fold9/model_iter17000.pt7' % REPO_PATH,
]
MODELS = PARTIAL_LEAF_DICE_MODEL_LIST
MODEL_ID = 'CameraReady_Partial_Leaf_Dice_fold0-9'
PRED_FOLDER_LUCAS = os.path.join(
    SAVE_FOLDER,
    'fetal_seg_pred_%s' % MODEL_ID
)

def print_results(metrics, method_names=METHOD_NAMES, save_path=None):
    print('\nGlobal statistics for the metrics')
    for method in method_names:
        print('\n\033[93m----------')
        print(method.upper())
        print('----------\033[0m')
        for roi in ALL_ROI:
            print('\033[92m%s\033[0m' % roi)
            for metric in METRIC_NAMES:
                key = '%s_%s' % (metric, roi)
                num_data = len(metrics[method][key])
                if num_data == 0:
                    print('No data for %s' % key)
                    continue
                print('%d cases' % num_data)
                mean = np.mean(metrics[method][key])
                std = np.std(metrics[method][key])
                median = np.median(metrics[method][key])
                q3 = np.percentile(metrics[method][key], 75)
                p95 = np.percentile(metrics[method][key], 95)
                q1 = np.percentile(metrics[method][key], 25)
                p5 = np.percentile(metrics[method][key], 25)
                print(key)
                if metric == 'dice':
                    print('mean=%.1f std=%.1f median=%.1f q1=%.1f p5=%.1f' % (mean, std, median, q1, p5))
                else:
                    print('mean=%.1f std=%.1f median=%.1f q3=%.1f p95=%.1f' % (mean, std, median, q3, p95))
            print('-----------')
    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(metrics, f, pickle.HIGHEST_PROTOCOL)


def compute_evaluation_metrics(pred_seg_path, gt_seg_path, dataset_path):
    def load_np(seg_path):
        seg = nib.load(seg_path).get_fdata().astype(np.uint8)
        return seg
    pred_seg_folder, pred_seg_name = os.path.split(pred_seg_path)
    pred_seg = load_np(pred_seg_path)
    gt_seg = load_np(gt_seg_path)

    if dataset_path == CORRECTED_ZURICH_DATA_DIR:
        print('Merge CC with WM')
        pred_seg[pred_seg == LABELS['corpus_callosum']] = LABELS['wm']

    if dataset_path == SB_LEUVEN_TESTINGSET:
        print('Change the CC label from 5 to 8')
        gt_seg[gt_seg == 5] = LABELS['corpus_callosum']

    # Compute the metrics
    dice_values = {}
    haus_values = {}
    for roi in DATASET_LABELS[dataset_path]:
        dice_values[roi] = dice_score(
            pred_seg,
            gt_seg,
            fg_class=LABELS[roi],
        )
        haus_values[roi] = min(
            MAX_HD,
            haussdorff_distance(
                pred_seg,
                gt_seg,
                fg_class=LABELS[roi],
                percentile=95,
            )
        )
    print('\n\033[92mEvaluation for %s\033[0m' % pred_seg_name)
    print('Dice scores:')
    print(dice_values)
    print('Hausdorff95 distances:')
    print(haus_values)
    return dice_values, haus_values


def main(dataset_path_list):
    if not os.path.exists(PRED_FOLDER_LUCAS):
        os.mkdir(PRED_FOLDER_LUCAS)
    # Initialize the metric dict
    metrics = {
        method: {'%s_%s' % (metric, roi): [] for roi in ALL_ROI for metric in METRIC_NAMES}
        for method in METHOD_NAMES
    }

    # Run the batch inference
    for dataset in dataset_path_list:
        for f_n in os.listdir(dataset):
            print('\n--------------')
            print('Start inference for case %s' % f_n)
            if '.' in f_n:
                continue
            input_path = os.path.join(dataset, f_n, 'srr.nii.gz')
            output_path = os.path.join(PRED_FOLDER_LUCAS, f_n)
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            pred_path = os.path.join(
                output_path,
                'srr_parcellation_cnn_autoseg.nii.gz',
            )
            if not os.path.exists(pred_path):  # Skip inference if the predicted segmentation already exists
                # Set the models to use
                cmd_models = '--model'
                for model in MODELS:
                    cmd_models += ' %s' % model
                # Inference command line
                cmd = 'python %s/infer_seg.py --input %s --output_folder %s --num_classes %d %s' % \
                    (REPO_PATH, input_path, output_path, NUM_CLASS, cmd_models)
                print(cmd)
                os.system(cmd)

            # Eval
            if DO_EVAL:
                gt_seg_path = os.path.join(dataset, f_n, 'parcellation.nii.gz')
                for method in METHOD_NAMES:
                    dice, haus = compute_evaluation_metrics(pred_path, gt_seg_path, dataset_path=dataset)
                    for roi in DATASET_LABELS[dataset]:
                        metrics[method]['dice_%s' % roi].append(100 * dice[roi])
                        metrics[method]['hausdorff_%s' % roi].append(haus[roi])

    # Save and print the metrics aggregated
    if DO_EVAL:
        save_metrics_path = os.path.join(PRED_FOLDER_LUCAS, 'metrics.pkl')
        print_results(metrics, save_path=save_metrics_path)
    else:
        print('\nNo evaluation was run.')


def main_ori_vs_after_correction_zurich_data():
    # Measure the difference between the original data from Zurich
    # and the manually corrected segmentations
    cases_ids = os.listdir(CORRECTED_ZURICH_DATA_DIR)
    metrics = {'%s_%s' % (metric, roi): []
               for roi in DATASET_LABELS[CORRECTED_ZURICH_DATA_DIR]
               for metric in METRIC_NAMES}
    print('Compare Zurich segmentation before and after manual corrections')
    print('%d cases to evaluate\n' % len(cases_ids))
    for case_id in cases_ids:
        print('\n----------------')
        print('Case %s' % case_id)
        new_seg = os.path.join(CORRECTED_ZURICH_DATA_DIR, case_id, 'parcellation.nii.gz')
        old_seg = os.path.join(ORI_ZURICH_DATA_DIR, case_id, 'parcellation.nii.gz')
        dice, haus = compute_evaluation_metrics(
            new_seg, old_seg, dataset_path=CORRECTED_ZURICH_DATA_DIR)
        for roi in DATASET_LABELS[CORRECTED_ZURICH_DATA_DIR]:
            metrics['dice_%s' % roi].append(100 * dice[roi])
            metrics['hausdorff_%s' % roi].append(haus[roi])

    # Print the global results
    print('\nGLOBAl METRICS')
    for metric in METRIC_NAMES:
        for roi in DATASET_LABELS[CORRECTED_ZURICH_DATA_DIR]:
            key = '%s_%s' % (metric, roi)
            mean = np.mean(metrics[key])
            std = np.std(metrics[key])
            median = np.median(metrics[key])
            q3 = np.percentile(metrics[key], 75)
            q1 = np.percentile(metrics[key], 25)
            IQR = q3 - q1
            print(key)
            print('mean=%f std=%f median=%f IQR=%f' % (mean, std, median, IQR))


def main_results_analysis(pkl_files_list):
    """
    Useful for averaging the results of several evaluations.
    :param pkl_files_list: list metrics computed with one of the main function above
    :return:
    """
    print('')
    for method in METHOD_NAMES:
        for roi in ALL_ROI:
            print('\033[92m%s\033[0m' % roi)
            for metric in METRIC_NAMES:
                key = '%s_%s' % (metric, roi)
                mean_list = []
                std_list = []
                median_list = []
                iqr_list = []
                q1_list = []
                q3_list = []
                for pkl_file in pkl_files_list:
                    with open(pkl_file, 'rb') as f:
                        metrics = pickle.load(f)
                    key = '%s_%s' % (metric, roi)
                    mean_list.append(np.mean(metrics[method][key]))
                    std_list.append(np.std(metrics[method][key]))
                    median_list.append(np.median(metrics[method][key]))
                    q3 = np.percentile(metrics[method][key], 75)
                    q1 = np.percentile(metrics[method][key], 25)
                    iqr_list.append(q3 - q1)
                    q3_list.append(q3)
                    q1_list.append(q1)
                print(key)
                if metric == 'dice':
                    print('mean=%.1f (%.1f) std=%.1f (%.1f) median=%.1f (%.1f) q1=%.1f (%.1f)' %
                      (np.mean(mean_list), np.std(mean_list),
                       np.mean(std_list), np.std(std_list),
                       np.mean(median_list), np.std(median_list),
                       np.mean(q1_list), np.std(q1_list)
                    ))
                else:
                    print('mean=%.1f (%.1f) std=%.1f (%.1f) median=%.1f (%.1f) q3=%.1f (%.1f)' %
                      (np.mean(mean_list), np.std(mean_list),
                       np.mean(std_list), np.std(std_list),
                       np.mean(median_list), np.std(median_list),
                       np.mean(q3_list), np.std(q3_list)
                    ))
            print('-----------')


if __name__ == '__main__':
    t_start = time()

    if not os.path.exists(SAVE_FOLDER):
        os.mkdir(SAVE_FOLDER)

    # PIPELINE EVALUATION
    main(DATA_DIR)

    # ZURICH DATA CORRECTIONS COMPARISON TO ORIGINAL
    # main_ori_vs_after_correction_zurich_data()


    total_time = int(time() - t_start)
    print('\nTotal time=%dmin%dsec' % (total_time // 60, total_time % 60))
