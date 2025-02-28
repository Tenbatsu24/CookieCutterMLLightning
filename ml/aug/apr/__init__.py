from .mix_data import APR

from ml.util import STORE, AUG_TYPE

STORE.register(AUG_TYPE, "apr", APR)

__all__ = ["APR"]
