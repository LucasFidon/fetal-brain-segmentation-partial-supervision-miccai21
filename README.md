# Label-set Loss Functions for Partial Supervision: Application to Fetal Brain 3D MRI Parcellation

This repository contains a copy of the code that was used for 
the experiments described in our paper

L. Fidon, M. Aertsen, D. Emam, N. Mufti, F. Guffens, T. Deprest, P. Demaerel, A. L. David, A. Melbourne, S. Ourselin, J. Deprest, T. Vercauteren.
[Label-set Loss Functions for Partial Supervision: Application to Fetal Brain 3D MRI Parcellation][arxiv]
MICCAI 2021.


## Installation
This repository is based on our github repository for label-set loss functions
https://github.com/LucasFidon/label-set-loss-functions
To install our package run
```bash
pip install git+https://github.com/LucasFidon/label-set-loss-functions.git
```

## Data
The pre-trained models and the 40 3D MRIs of the [FeTA dataset][feta] (data release 1) with our corrected manual segmentations
that we used for evaluation can be downloaded [here][zenodo].

After downloading the folder ```\MICCAI21_partial_supervision_trained_models``` that contains the pre-trained models,
please move the folder in ```fetal-brain-segmentation-partial-supervision-miccai21\data```.

## How to cite

L. Fidon, M. Aertsen, D. Emam, N. Mufti, F. Guffens, T. Deprest, P. Demaerel, A. L. David, A. Melbourne, S. Ourselin, J. Deprest, T. Vercauteren.
[Label-set Loss Functions for Partial Supervision: Application to Fetal Brain 3D MRI Parcellation][arxiv]
International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2021.

Bibtex:
```
@inproceedings{fidon2021partial,
  title={Label-set Loss Functions for Partial Supervision: Application to Fetal Brain {3D MRI} Parcellation},
  author={Fidon, Lucas and Aertsen, Michael and Emam, Doaa and Mufti, Nada and Guffens, Fr{\'e}d{\'e}ric and Deprest, Thomas and Demaerel, Philippe and L. David, Anna and Melbourne, Andrew and Ourselin, S{\'e}bastien and Deprest, Jan and Vercauteren, Tom},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2021},
  organization={Springer}
}
```
[arxiv]: https://arxiv.org/abs/2107.03846
[zenodo]: https://zenodo.org/record/5148612#.YQqbHHWYVhF
[feta]: https://zenodo.org/record/4541606#.YQqdpnWYU5k