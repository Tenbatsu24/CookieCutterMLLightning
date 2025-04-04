from .cifar import ResNet18, ResNeXt29

from ml.util import MODEL_TYPE, STORE

STORE.register(MODEL_TYPE, "rn18", ResNet18)
STORE.register(MODEL_TYPE, "rnxt29", ResNeXt29)

__all__ = ["ResNet18", "ResNeXt29"]
