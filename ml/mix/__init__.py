from .mixup import RandomMixUp
from .cutmix import RandomCutMix

from ml.util import MIX_TYPE, STORE

STORE.register(MIX_TYPE, "mixup", RandomMixUp)
STORE.register(MIX_TYPE, "cutmix", RandomCutMix)

__all__ = ["RandomMixUp", "RandomCutMix"]
