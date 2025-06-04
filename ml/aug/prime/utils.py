from loguru import logger

from .diffeomorphism import Diffeo
from .rand_filter import RandomFilter
from .color_jitter import RandomSmoothColor
from .modules import GeneralizedPRIMEModule, PRIMEAugModule

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
        "cut_min": 2,
        "cut_max": 100,
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
        "cut_min": 2,
        "cut_max": 500,
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


def make_prime(dataset):
    logger.trace(f"Creating PRIME with dataset {dataset}")

    aug_module = PRIMEAugModule(make_aug_list(dataset))

    return GeneralizedPRIMEModule(
        aug_module,
        mixture_width=3,
        mixture_depth=-1,
        max_depth=3,
    )
