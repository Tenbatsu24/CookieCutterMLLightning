from math import sqrt
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm
from torch.optim import AdamW
from torchmetrics import Accuracy

from ml.probing.util import Analysis


class LinearAnalysis(Analysis):
    def __init__(self, n_epochs: int):
        super().__init__()
        self.n_epochs = n_epochs
        self.clf = None
        self.opt = None
        self.acc = None
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

        self.clf = nn.Linear(n_features, n_classes)
        self.opt = AdamW(self.clf.parameters())
        self.acc = Accuracy(task="multiclass", num_classes=n_classes)
        if device and device.type == "cuda":
            self.clf = self.clf.to(device=device)
            self.acc = self.acc.to(device=device)
            self.device = device

        # m.reset_parameters() is equal to:
        bound = 1 / sqrt(self.clf.in_features)
        init.uniform_(self.clf.weight, -bound, bound, generator=generator)
        if self.clf.bias is not None:
            init.uniform_(self.clf.bias, -bound, bound, generator=generator)

    @torch.inference_mode(False)
    @torch.no_grad()
    def train(self, train_data: List[Tuple[torch.Tensor, torch.Tensor]], verbose=True):
        self.clf.train()
        train_pbar = tqdm(range(self.n_epochs), leave=False)
        train_pbar.set_description(f"Training")

        for epoch in train_pbar:  # training
            for embeddings, targets in train_data:
                self.opt.zero_grad(set_to_none=True)  # step clf parameters

                with torch.enable_grad():
                    loss = F.cross_entropy(
                        self.clf(embeddings.to(self.device)), targets.to(self.device)
                    )
                loss.backward()
                self.opt.step()

            train_pbar.set_postfix({"loss": float(loss)})

    def valid(self, valid_data: List[Tuple[torch.Tensor, torch.Tensor]]):
        self.clf.eval()
        valid_pbar = tqdm(valid_data, leave=False)
        valid_pbar.set_description("Validation")

        self.acc.reset()
        for embeddings, targets in valid_pbar:
            predictions = self.clf(embeddings.to(self.device))
            self.acc.update(predictions, targets.to(self.device))
            valid_pbar.set_postfix({"acc": float(self.acc.compute())})

        return float(self.acc.compute())

    def cleanup(self):
        self.clf = None
        self.opt = None
        self.acc = None
        self.device = torch.device("cpu")
