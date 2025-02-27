from loguru import logger

from ml.aug.prime.diffeomorphism import Diffeo
from ml.aug.prime.rand_filter import RandomFilter
from ml.aug.prime.color_jitter import RandomSmoothColor

AUG_MAP = {"diffeo": Diffeo, "color_jit": RandomSmoothColor, "rand_filter": RandomFilter}

DEFAULT_PARAMS = {
    "color_jit": {"cut": 500, "T": 0.05, "freq_bandwidth": 20},
    "rand_filter": {"kernel_size": 3, "sigma": 4.0},
}

CIFAR_PARAMS = {
    "diffeo": {
        "sT": 1.0,
        "rT": 1.0,
        "scut": 1.0,
        "rcut": 1.0,
        "cutmin": 2,
        "cutmax": 100,
        "alpha": 1.0,
    },
    "color_jit": {"cut": 100, "T": 0.01, "freq_bandwidth": None},
    "rand_filter": {"kernel_size": 3, "sigma": 4.0},
}

IMAGE_NET_PARAMS = {
    "diffeo": {
        "sT": 1.0,
        "rT": 1.0,
        "scut": 1.0,
        "rcut": 1.0,
        "cutmin": 2,
        "cutmax": 500,
        "alpha": 1.0,
    },
    "color_jit": {"cut": 500, "T": 0.05, "freq_bandwidth": 20},
    "rand_filter": {"kernel_size": 3, "sigma": 4.0},
}

ALL_PARAMS = {
    "c": CIFAR_PARAMS,
    "in": IMAGE_NET_PARAMS,
}


def make_aug_list(dataset=None):
    logger.trace(f"{dataset=} found in ALL_PARAMS: {dataset in ALL_PARAMS}")
    params = ALL_PARAMS.get(dataset, DEFAULT_PARAMS)

    aug_config = []
    for aug, aug_params in params.items():
        logger.trace(f"Adding augmentation {aug} with params {aug_params}")
        aug_config += [AUG_MAP[aug](**aug_params, stochastic=True)]

    return aug_config
