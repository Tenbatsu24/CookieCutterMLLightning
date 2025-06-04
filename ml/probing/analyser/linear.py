from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics import Accuracy
from ml_collections import ConfigDict
from torch.utils.data import DataLoader

from ml.trainer import BaseTrainer
from ml.probing.util import Analysis, EmbeddingDataset


DEFAULT_LINEAR_PROBE_CONFIG = {
    "n_epochs": 100,
    "opt": {
        "params": {"betas": [0.9, 0.999], "eps": 1e-08, "lr": 1e-9, "weight_decay": 1e-6},
        "type": "AdamW",
    },
    "scheduler": [
        ["lr", "CatSched(LinSched(1e-9, 1e-3), CosSched(1e-3, 1e-6), 5)"],
        ["weight_decay", "CosSched(0.005, 0.01)"],
    ],
}


class LinearAnalysis(Analysis):
    def __init__(self, depth=1, config: dict = None):
        super().__init__()
        self.config = ConfigDict(
            config if config else DEFAULT_LINEAR_PROBE_CONFIG, convert_dict=True
        )

        self.depth = depth
        self.n_epochs = self.config.n_epochs

        self.clf = None
        self.opt = None
        self.acc_1 = None
        self.sched = None
        self.device = torch.device("cpu")

    @torch.inference_mode(False)
    @torch.no_grad()
    def prepare(
        self,
        n_features: int,
        n_classes: int,
        device: torch.device = None,
        generator: torch.Generator = None,
    ):
        from torch.nn import init

        if self.depth < 1:
            raise ValueError("Depth must be greater than 0")
        if self.depth == 1:
            self.clf = nn.Linear(n_features, n_classes)
        else:
            layers = []
            for i in range(self.depth - 1):
                layers.append(nn.Linear(n_features, n_features))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.5))
            layers.append(nn.Linear(n_features, n_classes))
            self.clf = nn.Sequential(*layers)

        [self.opt], self.sched = BaseTrainer.make_opt_sched(self.config, self.clf)
        self.acc_1 = Accuracy(task="multiclass", num_classes=n_classes)
        if device and device.type == "cuda":
            self.clf = self.clf.to(device=device)
            self.acc_1 = self.acc_1.to(device=device)
            self.device = device

        # m.reset_parameters() is equal to:
        bound = 1 / sqrt(n_features)
        for m in self.clf.modules():
            if isinstance(m, nn.Linear):
                init.uniform_(m.weight, -bound, bound)
                if m.bias is not None:
                    init.zeros_(m.bias)

    @torch.inference_mode(False)
    @torch.no_grad()
    def train(self, train_data: EmbeddingDataset, verbose=True):
        self.clf.train()

        with DataLoader(
            train_data,
            batch_size=1024,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
            persistent_workers=False,
        ) as train_loader:

            self.sched.prep(self.n_epochs * len(train_loader), self.n_epochs, len(train_loader))
            step_counter = 0
            for _ in range(self.n_epochs):  # training
                for embeddings, targets in train_loader:
                    self.opt.zero_grad(set_to_none=True)  # step clf parameters
                    self.sched.step(step_counter)

                    with torch.enable_grad():
                        loss = F.cross_entropy(
                            self.clf(embeddings.to(self.device)), targets.to(self.device)
                        )
                    loss.backward()
                    self.opt.step()
                    step_counter += 1

    def valid(self, valid_data: EmbeddingDataset):
        self.clf.eval()
        self.acc_1.reset()

        with DataLoader(
            valid_data,
            batch_size=256,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
        ) as valid_loader:
            for embeddings, targets in valid_loader:
                predictions = self.clf(embeddings.to(self.device))
                self.acc_1.update(predictions, targets.to(self.device))

        return float(self.acc_1.compute())

    def cleanup(self):
        self.clf = None
        self.opt = None
        self.acc_1 = None
        self.sched = None
        self.device = torch.device("cpu")
