from .spatial import OrigAFA
from .real import RealNDimFourier
from .fourier import ScaledNDimFourierAFA

from ml.util import STORE, AUG_TYPE

STORE.register(AUG_TYPE, "afa", OrigAFA)
STORE.register(AUG_TYPE, "r_afa", RealNDimFourier)
STORE.register(AUG_TYPE, "f_afa", ScaledNDimFourierAFA)

__all__ = ["OrigAFA", "RealNDimFourier", "ScaledNDimFourierAFA"]
