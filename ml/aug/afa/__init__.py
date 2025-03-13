from .spatial import OrifAFA
from .real import RealNDimFourier
from .fourier import ScaledNDimFourierAFA

from ml.util import STORE, AUG_TYPE

STORE.register(AUG_TYPE, "afa", OrifAFA)
STORE.register(AUG_TYPE, "r_afa", RealNDimFourier)
STORE.register(AUG_TYPE, "f_afa", ScaledNDimFourierAFA)

__all__ = ["OrifAFA", "RealNDimFourier", "ScaledNDimFourierAFA"]
