from .mixup import RandomMixUp
from .cutmix import RandomCutMix
from .cutmixup import CutMixUp, make_cutmix_mixup

from ml.util import MIX_TYPE, STORE

STORE.register(MIX_TYPE, "mixup", RandomMixUp)
STORE.register(MIX_TYPE, "cutmix", RandomCutMix)
STORE.register(MIX_TYPE, "cutmixup", CutMixUp)

__all__ = ["RandomMixUp", "RandomCutMix", "CutMixUp", "make_cutmix_mixup"]
