# EMDB-X
The aim is to extend the in-the-wild pose estimation dataset EDMB with hand gesture and facial expression annotations. The goal is to use dense landmark predictors trained on a synthetic dataset to provide 2D labels. We then fit the 3D SMPL-X parameters, including hands and faces, to match the 2D observations and obtain pseudo-groundtruth labels.

### SET UP THE WORKING ENVIRONMENT
Before starting to use the current repo, you need to download the [EMDB Dataset](https://eth-ait.github.io/emdb/), the [SMPL-X model](https://smpl-x.is.tue.mpg.de/), the [SMPL model](https://smpl.is.tue.mpg.de/). The worksapce organization has to be the following:
```
code 
├── data
│   └── EMDB_dataset
│       ├── P0
│       ├── P1
│       └── ...
│   └── smplx_models
│       ├── smpl
│       │   ├── SMPL_FEMALE.pkl (renamed)
│       │   ├── SMPL_MALE.pkl (renamed)
│       │   └── SMPL_NEUTRAL.pkl (renamed)
│       └── smplx
│           ├── SMPLX_FEMALE.pkl 
│           ├── SMPLX_MALE.pkl 
│           ├── SMPLX_NEUTRAL.pkl 
│           ├── SMPLX_FEMALE.npz 
│           ├── SMPLX_MALE.npz
│           └── SMPLX_NEUTRAL.npz 
├── emdb
├── EMDBX
└── SMPLer-X
```
where `emdb` is the original EMDB repo that can be cloned from [GitHub](https://github.com/eth-ait/emdb) and `SMPLer-X` is the original [SMPLer-X repo](https://github.com/SMPLCap/SMPLer-X).

Make sure to add a `__init__.py` file in the `emdb` folder.
