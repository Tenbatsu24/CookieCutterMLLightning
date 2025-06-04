from torchvision.datasets import MNIST, FashionMNIST, KMNIST, EMNIST


class M(MNIST):
    mean = (0.1307,)
    std = (0.3081,)
    num_classes = 10
    image_size = 28


class FM(FashionMNIST):
    mean = (0.2860,)
    std = (0.3530,)
    num_classes = 10
    image_size = 28


class KM(KMNIST):
    mean = (0.1918,)
    std = (0.3059,)
    num_classes = 10
    image_size = 28
