from .trivial import TrivialAugment, TrivialAugmentWide, TrivialMix

from ml.util import AUG_TYPE, STORE

STORE.register(AUG_TYPE, "ta", TrivialAugment)
STORE.register(AUG_TYPE, "taw", TrivialAugmentWide)
STORE.register(AUG_TYPE, "tmix", TrivialMix)

__all__ = ["TrivialAugment", "TrivialAugmentWide", "TrivialMix"]
