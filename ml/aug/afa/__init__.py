from .spatial import AFA
from .fourier import FourierAFA

from ml.util import STORE, AUG_TYPE

STORE.register(AUG_TYPE, "afa", AFA)
STORE.register(AUG_TYPE, "f_afa", FourierAFA)

__all__ = ["AFA", "FourierAFA"]
