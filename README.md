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
