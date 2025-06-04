from ml.util import STORE, DATA_TYPE

from .util import make_loaders, generalisation_test
from .mnist import M, FM, KM
from .cifar import C10, C100, C10C, C100C, C10CBar, C100CBar, STL10


STORE.register(DATA_TYPE, "m", M)
STORE.register(DATA_TYPE, "fm", FM)
STORE.register(DATA_TYPE, "km", KM)

STORE.register(DATA_TYPE, "c10", C10)
STORE.register(DATA_TYPE, "c10c", C10C)
STORE.register(DATA_TYPE, "c10cb", C10CBar)

STORE.register(DATA_TYPE, "c100", C100)
STORE.register(DATA_TYPE, "c100c", C100C)
STORE.register(DATA_TYPE, "c100cb", C100CBar)

STORE.register(DATA_TYPE, "stl", STL10)

__all__ = ["make_loaders", "generalisation_test"]
