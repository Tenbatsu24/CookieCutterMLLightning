from .augmix import AugMix

from ml.util import AUG_TYPE, STORE

STORE.register(AUG_TYPE, "augmix", AugMix)

__all__ = ["AugMix"]
