
## Usage
  `train.py` -- Train classifier.
  `inference.py` -- Inference without GradCAM visualization (uses trainer.test())
  `prediction.py` -- (GradCAM vizualization on during validation without training (epochs 0))

  Place all settings in config file and use `train.py --config [config file name here]`


  ## Original solver settings
    OPT = 'adam'  # adam, sgd
    WEIGHT_DECAY = 0.0001
    MOMENTUM = 0.9  # only when OPT is sgd
    BASE_LR = 0.001
    LR_SCHEDULER = 'step'  # step, multistep, reduce_on_plateau
    LR_DECAY_RATE = 0.1
    LR_STEP_SIZE = 5  # only when LR_SCHEDULER is step
    LR_STEP_MILESTONES = [10, 15]  # only when LR_SCHEDULER is multistep

## Wandb logging
Set to True in config file.


## Directory Structure
EagleID/
├── data/                  # Datasets and data-related scripts
├── dataset/               # Annotations for datasets
├── models/                # Model architecture code
├── configs/               # Configuration files (e.g., hyperparameters, dataset info)
├── experiments/           # Logging and experiment management (results, logs, etc.)
├── notebooks/             # Jupyter notebooks for prototyping and demo
├── scripts/               # BASH scripts for running programs
├── utils/                 # Utility functions and helper scripts
├── tools/                 # Training, evaluation, and inference
├── checkpoints/           # Model checkpoints and saved weights
├── README.md              # Project description and usage guide
├── requirements.txt       # Dependencies list
└── .gitignore             # Files to ignore in version control


OR

AgeClassifier/
├── config/                  # Configuration files (YAML, JSON)
│   ├── config.yaml          # Primary config file
│   └── hpc_config.yaml
│
├── data/                    # Data-related scripts and datasets
│   ├── __init__.py
│   ├── dataset.py           # Custom dataset logic
│   ├── transforms.py        # Data augmentation and transforms
│   ├── annotations/         # Annotations (if small) for datasets
│   └── cache/               # Cached/preprocessed data
│
├── notebooks/               # Jupyter notebooks for exploratory analysis
│   ├── dataset_analysis.ipynb
│   └── data_preprocessing.ipynb
│
├── models/                  # Model architecture files
│   ├── __init__.py
│   ├── model.py             # Main model class
│   ├── resnet_model.py
│   └── loss.py              # Loss functions
│
├── training/                # Training-related scripts
│   ├── train.py             # Main training script
│   └── inference.py         # Inference and testing script
│
├── utils/                   # Utility functions
│   ├── optimizer.py         # Optimizer logic
│   ├── gradcam.py           # GradCAM logic
│   └── metrics.py           # Evaluation metrics
│
├── checkpoints/             # Model checkpoints
│   ├── model_latest.ckpt
│   └── model_best.ckpt
│
├── results/                 # Generated results (e.g., GradCAM outputs, predictions)
│   ├── gradcam_outputs/     
│   └── predictions/         
│
├── requirements.txt         # Python dependencies
├── README.md                # Project overview
└── .gitignore
