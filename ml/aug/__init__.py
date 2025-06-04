from .afa import OrigAFA, RealNDimFourier, ScaledNDimFourierAFA
from .apr import APR
from .augmix import GPUAugMix
from .prime import GeneralizedPRIMEModule, PRIMEAugModule, Diffeo, RandomSmoothColor, RandomFilter
from .ta import TrivialAugment, TrivialAugmentWide, TrivialMix

__all__ = [
    "OrigAFA",
    "RealNDimFourier",
    "ScaledNDimFourierAFA",
    "APR",
    "GPUAugMix",
    "GeneralizedPRIMEModule",
    "PRIMEAugModule",
    "Diffeo",
    "RandomSmoothColor",
    "RandomFilter",
    "TrivialAugment",
    "TrivialAugmentWide",
    "TrivialMix",
]
