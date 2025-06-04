import os

import numpy as np

from loguru import logger
from torchvision.datasets.utils import download_url, extract_archive
from torchvision.datasets import CIFAR10, CIFAR100, VisionDataset, STL10 as STL10Base, Imagenette


class C10(CIFAR10):
    mean = (0.4915, 0.4823, 0.4468)
    std = (0.2470, 0.2435, 0.2616)
    num_classes = 10
    image_size = 32


class C100(CIFAR100):
    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2762)
    num_classes = 100
    image_size = 32


class C10C(VisionDataset):
    mean = C10.mean
    std = C10.std
    num_classes = C10.num_classes
    image_size = C10.image_size

    url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1"
    filename = "CIFAR-10-C.tar"
    base_folder = "CIFAR-10-C"
    md5 = "56bf5dcef84df0e2308c6dcbcbbd8499"
    per_severity = 10000

    severities = [1, 2, 3, 4, 5]
    corruptions = [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "speckle_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "gaussian_blur",
        "snow",
        "frost",
        "fog",
        "spatter",
        "brightness",
        "contrast",
        "saturate",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
    ]

    def __init__(
        self,
        root,
        download=False,
        severity=1,
        corruption="gaussian_noise",
        transform=None,
        target_transform=None,
    ):
        assert severity in self.severities
        assert corruption in self.corruptions

        super().__init__(root, transform=transform, target_transform=target_transform)
        self.slice = slice((severity - 1) * self.per_severity, severity * self.per_severity)

        if download:
            if (root / self.filename).exists():
                logger.info(f"{self.__class__.__name__} already downloaded")
            else:
                download_url(self.url, root, self.filename, self.md5)

        if not os.path.exists(os.path.join(root, self.base_folder)):
            logger.info(f"Extracting {self.__class__.__name__}")
            extract_archive(root / self.filename, root)

        # now load the picked numpy arrays
        images_file_path = os.path.join(self.root, self.base_folder, f"{corruption}.npy")
        self.data = np.load(images_file_path)[self.slice]
        labels_file_path = os.path.join(self.root, self.base_folder, f"labels.npy")
        self.targets = np.load(labels_file_path)[self.slice]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class C100C(C10C):
    mean = C100.mean
    std = C100.std
    num_classes = C100.num_classes
    image_size = C100.image_size

    url = "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar?download=1"
    filename = "CIFAR-100-C.tar"
    base_folder = "CIFAR-100-C"
    md5 = "11f0ed0f1191edbf9fa23466ae6021d3"
    per_severity = 10000

    severities = [1, 2, 3, 4, 5]
    corruptions = [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "speckle_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "gaussian_blur",
        "snow",
        "frost",
        "fog",
        "spatter",
        "brightness",
        "contrast",
        "saturate",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
    ]


class C10CBar(VisionDataset):
    mean = C10.mean
    std = C10.std
    num_classes = C10.num_classes
    image_size = C10.image_size

    filename = "CIFAR-10-C-Bar.zip"
    base_folder = "CIFAR10-c-bar"
    per_severity = 10000

    severities = [1, 2, 3, 4, 5]
    corruptions = [
        "blue_noise",
        "brownish_noise",
        "checkerboard_cutout",
        "inverse_sparkles",
        "pinch_and_twirl",
        "ripple",
        "circular_motion_blur",
        "lines",
        "sparkles",
        "transverse_chromatic_abberation",
    ]

    def __init__(
        self,
        root,
        download=False,
        severity=1,
        corruption="lines",
        transform=None,
        target_transform=None,
    ):
        assert severity in self.severities
        assert corruption in self.corruptions

        super().__init__(root, transform=transform, target_transform=target_transform)
        self.slice = slice((severity - 1) * self.per_severity, severity * self.per_severity)

        if download:
            raise NotImplementedError("CIFAR C-Bar dataset(s) cannot be downloaded automatically")

        if not os.path.exists(os.path.join(root, self.base_folder)):
            logger.info(f"Extracting {self.__class__.__name__}")
            extract_archive(os.path.join(root, self.filename), root)

        # now load the picked numpy arrays
        images_file_path = os.path.join(self.root, self.base_folder, f"{corruption}.npy")
        self.data = np.load(images_file_path)[self.slice]
        labels_file_path = os.path.join(self.root, self.base_folder, f"labels.npy")
        self.targets = np.load(labels_file_path)[self.slice]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class C100CBar(C10CBar):
    mean = C100.mean
    std = C100.std
    num_classes = C100.num_classes
    image_size = C100.image_size

    filename = "CIFAR-100-C-Bar.zip"
    base_folder = "CIFAR100-c-bar"


class STL10(STL10Base):
    mean = C10.mean
    std = C10.std

    num_classes = 10
    image_size = 96

    def _check_integrity(self) -> bool:
        # if the folder exists, then is gucci
        if not (self.root / self.base_folder).exists():
            # check if file name exists
            if not (self.root / self.filename).exists():
                return False
            else:
                extract_archive(self.root / self.filename, self.root)
                return True
        else:
            return True


class IM10(Imagenette):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    num_classes = 10
    image_size = 256


if __name__ == "__main__":
    from ml.config import PROCESSED_DATA_DIR

    ds = C10(PROCESSED_DATA_DIR, download=True)
    print(ds.data.shape)

    ds = C100(PROCESSED_DATA_DIR, download=True)
    print(ds.data.shape)

    _ = C10C(PROCESSED_DATA_DIR, download=True, severity=1, corruption="gaussian_noise")
    _ = C100C(PROCESSED_DATA_DIR, download=True, severity=1, corruption="gaussian_noise")
    # _ = C10CBar(PROCESSED_DATA_DIR, download=True, extract_only=False, severity=1, corruption='blue_noise_sample')
    # _ = C100CBar(PROCESSED_DATA_DIR, download=True, extract_only=False, severity=1, corruption='blue_noise_sample')

    ds = STL10(PROCESSED_DATA_DIR, download=True)
    print(ds.data.shape)

    ds = IM10(PROCESSED_DATA_DIR, download=True)
    print(len(ds._samples))
