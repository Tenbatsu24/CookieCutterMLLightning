from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100, Imagenette

from ml.util import STORE, DATA_TYPE

STORE.register(DATA_TYPE, "mnist", MNIST)
STORE.register(DATA_TYPE, "fmnist", FashionMNIST)
STORE.register(DATA_TYPE, "c10", CIFAR10)
STORE.register(DATA_TYPE, "c100", CIFAR100)
STORE.register(DATA_TYPE, "imagenette", Imagenette)
