# Cookie Cutter ML Lightning

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

<a target="_blank" href="https://python.org/">
    <img src="https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat&logo=python&logoColor=white" />
</a>

<a target="_blank" href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.0-EE4C2C.svg?style=flat&logo=pytorch" />
</a>

<a target="_blank" href="https://wandb.ai/">
    <img src="https://img.shields.io/badge/wandb-FFCC33.svg?style=flat&logo=WeightsAndBiases&logoColor=black" />
</a>

<a target="_blank" href="https://lightning.ai/">
    <img src="https://img.shields.io/badge/-Lightning-792ee5?logo=lightning&logoColor=white" />
</a>

<a target="_blank" href="https://pre-commit.com/">
    <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=yellow" />
</a>

<a target="_blank" href="https://black.readthedocs.io/en/stable/">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" />
</a>


Rapid prototyping project using boilerplate that is ready to use out of the box.

## Installation

The project requires the following dependencies:

- Python 3.10
- conda
- make

For installation from scratch, run the following command:

```bash
make env
```

This will create a conda environment called `ccml_lightning` and install all the required dependencies.

If you already have a conda environment `(env_name)`, you can install the dependencies using the following command:

```bash
conda activate (env_name)
make requirements
```

Fill the `.env` file with the appropriate values.

```bash
touch .env

# REQUIRED FOR LOGGING
echo "WANDB_MODE=disabled" >> .env  # Set to online to log to Weights and Biases
echo "WANDB_API_KEY=" >> .env  # Weights and Biases API key for logging
echo "WANDB_ENTITY=" >> .env  # In case you want to log to a specific entity
echo "WANDB_PROJECT=ccml_lightning" >> .env  # Weights and Biases project name defaults to root dir name
```

```bash
# OPTIONAL, the defaults are set in the config.py file
echo "DATA_DIR=" >> .env  # Directory to the data
echo "SAMPLE_DATA_DIR=" >> .env  # Directory to the sample data used for testing

echo "MODELS_DIR=" >> .env  # Directory to the models
echo "REPORTS_DIR=" >> .env  # Directory to the reports
echo "LOGS_DIR=" >> .env  # Directory to the logs
```

🚨 **Do not push the `.env` file to the repository.**

--------

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen.
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`.
├── README.md          <- The top-level README for developers using this project.
├── configs            <- Configuration files for the ml experiments.
├── data
│   ├── sample         <- Data to be used for one-off testing (like images for augmentations).
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries.
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         ml and configuration for tools like black.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting.
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`.
│
├── setup.cfg          <- Configuration file for flake8.
│
└── ml   <- Source code for use in this project.
    │
    ├── __init__.py              <- Makes ml a Python module.
    │
    └──  config.py               <- Store useful variables and configuration.
```


--------

## Usage

### Training
```bash
python3 train_classification.py --help
# usage: train_classification.py [-h] --config CONFIG [--fast-dev-run]
#
#Train a model
#
#options:
#  -h, --help       show this help message and exit
#  --config CONFIG  Path to the config file
#  --fast-dev-run   Run a fast dev run

python3 train_classification.py --config c10-rn18.json
```
Look at the `configs` folder for more configuration files.

---

### 🔧 Configuration File Overview
This configuration file (`configs/c10-cct.json`) is used to train a Compact Convolution Transformer (CCT) on the CIFAR10 dataset using PyTorch Lightning. It defines all necessary settings for data loading, model architecture, optimization, logging, training, and evaluation.

The config is passed to the training script as follows:
```bash
python train_classification.py --config configs/c10-cct.json
```

---

### 🔑 Top-Level Keys Explained

#### **`batch_size`**
Defines the number of samples per batch used during training. Affects memory usage and convergence speed.

---

#### **`dataset`**
Specifies which dataset to use. In this case:
```json
"dataset": {
  "name": "c10"
}
```
refers to the CIFAR10 dataset.

---

#### **`log_latent`**
A boolean indicating whether to log latent representations (for visualization).

---

#### **`loss`**
Specifies the loss function and its parameters.
In this config:
- `type`: `"CrossEntropyLoss"`
- `params`: Includes options like `label_smoothing` and `reduction`.

---

#### **`metrics`**
Defines evaluation metrics to monitor performance.
Example:
```json
"acc": {
  "type": "Accuracy",
  "params": {
    "task": "multiclass",
    "average": "macro",
    "num_classes": 10
  }
}
```
This calculates macro-averaged accuracy for a 10-class classification task.

---

#### **`model`**
Specifies the model architecture and its parameters:
- `type`: `"32-cct"` (refers to a 32x32 Compact Convolution Transformer)
- `params`: Image size, input channels, and number of output classes.

---

#### **`opt` (Optimizer)**
Defines the optimizer and its hyperparameters:
- `type`: `"AdamW"`
- `params`: Learning rate, weight decay, betas, epsilon, etc.

---

#### **`scheduler`**
A list of learning rate and weight decay schedulers.
Each entry is a tuple: `[param_name, scheduler_expression]`.

For example:
```json
["lr", "CatSched(LinSched(1e-9, 1e-3), CosSched(1e-3, 1e-6), 2)"]
```
- Schedules are defined via composable string expressions parsed with AST from the `ml.scheduler` module.

📌 **Note:** This design allows flexible, nested scheduler definitions.

---

#### **`pl` (PyTorch Lightning Configuration)**
Houses configuration for:
- **`checkpoint`**: Settings for saving model checkpoints (e.g., based on validation accuracy).
- **`trainer`**: Controls PyTorch Lightning's training loop (devices, precision, gradient clipping, epoch limits, etc.).

---

#### **`run_id`**
An optional identifier for the current training run (useful for logging or experiment tracking).

---

#### **`run_name`**
A string used to label the run. Helps in organizing logs and outputs.

---

--------
