from .compact_transformers import (
    cct_7_3x1_32,
    cct_14_7x2_224,
    cvt_7_4_32,
    cvt_14_16_224,
    vit_lite_7_4_32,
)
from .cifar import ResNet18, ResNeXt29
from .imagenet import resnet18, resnet50
from .dino import vit_small, vit_base, vit_large, vit_giant

from ml.util import MODEL_TYPE, STORE

STORE.register(MODEL_TYPE, "rn18", ResNet18)
STORE.register(MODEL_TYPE, "rnxt29", ResNeXt29)
STORE.register(MODEL_TYPE, "32-cct", cct_7_3x1_32)
STORE.register(MODEL_TYPE, "224-cct", cct_14_7x2_224)
STORE.register(MODEL_TYPE, "32-cvt", cvt_7_4_32)
STORE.register(MODEL_TYPE, "224-cvt", cvt_14_16_224)
STORE.register(MODEL_TYPE, "32-vit-lite", vit_lite_7_4_32)
STORE.register(MODEL_TYPE, "rn18im", resnet18)
STORE.register(MODEL_TYPE, "rn50im", resnet50)
STORE.register(MODEL_TYPE, "vit-s", vit_small)
STORE.register(MODEL_TYPE, "vit-b", vit_base)
STORE.register(MODEL_TYPE, "vit-l", vit_large)
STORE.register(MODEL_TYPE, "vit-g", vit_giant)

__all__ = [
    "ResNet18",
    "ResNeXt29",
    "resnet18",
    "resnet50",
    "cct_7_3x1_32",
    "cct_14_7x2_224",
    "cvt_7_4_32",
    "cvt_14_16_224",
    "vit_lite_7_4_32",
    "vit_small",
    "vit_base",
    "vit_large",
    "vit_giant",
]
