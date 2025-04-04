from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100, Imagenette

from ml.util import STORE, DATA_TYPE

STORE.register(DATA_TYPE, "m", MNIST)
STORE.register(DATA_TYPE, "fm", FashionMNIST)
STORE.register(DATA_TYPE, "c10", CIFAR10)
STORE.register(DATA_TYPE, "c100", CIFAR100)
STORE.register(DATA_TYPE, "im10", Imagenette)
