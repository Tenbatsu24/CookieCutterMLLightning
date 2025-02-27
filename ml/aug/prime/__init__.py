from .modules import (
    GeneralizedPRIMEModule,
    PRIMEAugModule,
)
from .diffeomorphism import Diffeo
from .color_jitter import RandomSmoothColor
from .rand_filter import RandomFilter

from ml.util import AUG_TYPE, STORE

STORE.register(AUG_TYPE, "diffeo", Diffeo)
STORE.register(AUG_TYPE, "color_jit", RandomSmoothColor)
STORE.register(AUG_TYPE, "rand_filter", RandomFilter)
STORE.register(AUG_TYPE, "p_aug_mod", PRIMEAugModule)
STORE.register(AUG_TYPE, "gen_p_mod", GeneralizedPRIMEModule)
