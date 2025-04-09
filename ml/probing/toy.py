from math import sqrt

import torch

from torchvision.datasets import VisionDataset


class ToySet(VisionDataset):
    img_size = (1, 1)
    ds_pixels = 1
    ds_channels = 2
    ds_classes = 2
    cmean, cstd = 2.0, 1.0  # cluster mean and center
    mean = torch.Tensor((0, 0))
    std = torch.Tensor((sqrt(cmean**2 + cstd**2), sqrt(cmean**2 + cstd**2)))

    def __init__(self, train: bool = True, n_samples=100, **kwargs) -> None:
        self.n_samples = n_samples

        # generate clustered data
        self.data = torch.cat(
            [
                torch.normal(-self.cmean, self.cstd, (n_samples // 2, 2)),
                torch.normal(+self.cmean, self.cstd, (n_samples // 2, 2)),
            ]
        )

        # generate cluster labels
        self.lbls = torch.cat([torch.full((n_samples // 2,), 0), torch.full((n_samples // 2,), 1)])

        super().__init__(train, **kwargs)

    def __getitem__(self, index: int):
        return (self.data[index] - self.mean) / self.std, self.lbls[index]

    def __len__(self) -> int:
        return self.n_samples


if __name__ == "__main__":
    import argparse

    import torch.nn as nn

    from torch.utils.data import DataLoader

    from ml.probing import Prober, LinearAnalysis, KNNAnalysis

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_samples", type=int, default=100)
    parser.add_argument("--valid_samples", type=int, default=100)
    args = parser.parse_args()

    train_set = ToySet(train=True, n_samples=args.train_samples)
    valid_set = ToySet(train=False, n_samples=args.valid_samples)

    # prepare dataloaders
    train_dl = DataLoader(dataset=train_set, shuffle=True, batch_size=10)
    valid_dl = DataLoader(dataset=valid_set)

    # prepare prober
    prober = Prober(
        encoders={"": nn.Identity()},
        analyses={"lin": LinearAnalysis(n_epochs=100), "knn": KNNAnalysis(k=20)},
        train_dl=train_dl,
        valid_dl=valid_dl,
        n_classes=2,
    )

    # train and validate
    prober.probe()
