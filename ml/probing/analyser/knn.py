from warnings import warn
from typing import List, Tuple

import torch

from tqdm.auto import tqdm
from torchmetrics import Accuracy

from ml.probing.util import Analysis


class KNNAnalysis(Analysis):
    id = "knn"

    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.index = None
        self.labels = None
        self.acc = None
        self.device = torch.device("cpu")

    def prepare(
        self,
        n_features: int,
        n_classes: int = 0,
        device: torch.device = None,
        generator: torch.Generator = None,
    ):
        import faiss.contrib.torch_utils

        self.index = faiss.IndexFlat(n_features)
        self.acc = Accuracy(task="multiclass", num_classes=n_classes)
        if device and device.type == "cuda":
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, device.index, self.index)
            self.acc = self.acc.to(device=device)
            self.device = device

    def train(self, train_data: List[Tuple[torch.Tensor, torch.Tensor]]):
        train_pbar = tqdm(train_data, leave=False)
        train_pbar.set_description(f"Training")

        labels = []
        for embeddings, targets in train_pbar:
            self.index.add(embeddings.to(self.device))
            labels.append(targets.to(self.device))
        self.labels = torch.cat(labels)

    def valid(self, valid_data: List[Tuple[torch.Tensor, torch.Tensor]]):
        valid_pbar = tqdm(valid_data, leave=False)
        valid_pbar.set_description("Validation")

        self.acc.reset()
        for embeddings, targets in valid_pbar:
            _, indices = self.index.search(embeddings.to(self.device), self.k)
            predictions = self.labels[indices].mode()[0]

            try:  # catch bug from torchmetric?
                self.acc.update(predictions, targets.to(self.device))
            except Exception as e:
                warn(f"Could not compute accuracy: {str(e)})")
            valid_pbar.set_postfix({"acc": float(self.acc.compute())})

        return float(self.acc.compute())

    def cleanup(self):
        if self.index is not None:
            self.index.reset()
        self.index = None
        self.labels = None
        self.acc = None
        self.device = torch.device("cpu")
