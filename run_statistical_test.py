from scipy.stats import wilcoxon
import pickle
import os
from run_infer_eval import SAVE_FOLDER


def load_metrics(pkl_file):
    with open(pkl_file, 'rb') as f:
        metrics = pickle.load(f)['cnn']
    return metrics

def main(metrics1_pkl, metrics2_pkl):
    def guess_method_name(pkl_file_name):
        folder = os.path.split(os.path.dirname(pkl_file_name))[1]
        method = folder.replace('fetal_seg_pred_', '')
        method = method.replace('_flipRL', '')
        return method
    met1_name = guess_method_name(metrics1_pkl)
    met2_name = guess_method_name(metrics2_pkl)
    met1 = load_metrics(metrics1_pkl)
    met2 = load_metrics(metrics2_pkl)
    for metric in met1.keys():
        print('\033[92m%s\033[0m' % metric)
        print('%s vs %s' % (met1_name, met2_name))
        print(wilcoxon(met1[metric], met2[metric]))


if __name__ == '__main__':
    fully_sup_pkl = os.path.join(
        SAVE_FOLDER,
        'fetal_seg_pred_CameraReady_Full_Dice_fold0-9_flipRL',
        'metrics.pkl'
    )
    ls_dice_pkl = os.path.join(
        SAVE_FOLDER,
        'fetal_seg_pred_CameraReady_Partal_Leaf_Dice_fold0-9_flipRL',
        'metrics.pkl'
    )
    main(fully_sup_pkl, ls_dice_pkl)
    print('\n')
    marginal_dice_pkl = os.path.join(
        SAVE_FOLDER,
        'fetal_seg_pred_CameraReady_Marginalized_Dice_fold0-9_flipRL',
        'metrics.pkl'
    )
    main(fully_sup_pkl, marginal_dice_pkl)
    print('\n')
    soft_target_dice_pkl = os.path.join(
        SAVE_FOLDER,
        'fetal_seg_pred_CameraReady_Soft_Target_Dice_fold0-9_flipRL',
        'metrics.pkl'
    )
    main(fully_sup_pkl, soft_target_dice_pkl)
    print('\n')
    main(ls_dice_pkl, marginal_dice_pkl)
    print('\n')
    main(ls_dice_pkl, soft_target_dice_pkl)
    print('\n')
    main(marginal_dice_pkl, soft_target_dice_pkl)

