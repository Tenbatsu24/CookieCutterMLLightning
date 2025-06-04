from .augmix import GPUAugMix

from ml.util import AUG_TYPE, STORE

STORE.register(AUG_TYPE, "gpu_augmix", GPUAugMix)

__all__ = ["GPUAugMix"]
