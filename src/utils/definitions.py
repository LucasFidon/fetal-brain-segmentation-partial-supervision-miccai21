import os

MIN_SIZE = [144, 160, 144]
NUM_CLASS = 9  # number of classes predicted by the model
ALL_ROI = [
    'wm', 'csf', 'cerebellum', 'external_csf',
    'cortical_gm', 'deep_gm', 'brainstem', 'corpus_callosum',
]
LABELS = {
    'wm': 1,
    'csf': 2,  # ventricles
    'cerebellum': 3,
    'external_csf': 4,
    'cortical_gm': 5,
    'deep_gm': 6,
    'brainstem': 7,
    'corpus_callosum': 8,
    'background': 0,
}
LABELSET_MAP = {
    9: [LABELS['external_csf'], LABELS['cortical_gm'], LABELS['deep_gm'], LABELS['brainstem'], LABELS['corpus_callosum']],
    10: [LABELS['cortical_gm'], LABELS['deep_gm'], LABELS['brainstem'], LABELS['corpus_callosum']],
    11: [LABELS['cortical_gm'], LABELS['deep_gm'], LABELS['brainstem']],
}

# USEFUL PATHS
REPO_PATH = '/workspace/fetal-brain-segmentation-partial-supervision-miccai21'
TRAINED_MODELS = os.path.join(REPO_PATH, 'data', 'MICCAI21_partial_supervision_trained_models')

# Testing data
CORRECTED_ZURICH_DATA_DIR = os.path.join(  # 40 cases
    '/data',
    'Fetal_SRR_and_Seg',
    'FetalDataZurichCorrected',
    'TrainingSet',
)
Controls_LEUVEN_TESTINGSET = os.path.join(  # 19 controls cases
    '/data',
    'Fetal_SRR_and_Seg',
    'Controls_Leuven',
)
SB_LEUVEN_TESTINGSET = os.path.join(  # 41 SB cases
    '/data',
    'Fetal_SRR_and_Seg',
    'SB_Leuven',
)
DATASET_LABELS = {
    CORRECTED_ZURICH_DATA_DIR:
        ['wm', 'csf', 'cerebellum', 'external_csf', 'cortical_gm', 'deep_gm', 'brainstem'],
    SB_LEUVEN_TESTINGSET:
        ['wm', 'csf', 'cerebellum', 'external_csf', 'corpus_callosum'],
    Controls_LEUVEN_TESTINGSET:
        ['wm', 'csf', 'cerebellum', 'external_csf'],
}