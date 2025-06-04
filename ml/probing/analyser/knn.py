from warnings import warn

import torch

from torchmetrics import Accuracy

from ml.probing.util import Analysis, EmbeddingDataset


class KNNAnalysis(Analysis):

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

    def train(self, train_data: EmbeddingDataset):
        labels = []
        with torch.utils.data.DataLoader(
            train_data, batch_size=256, shuffle=False, num_workers=0
        ) as train_dl:
            for embeddings, targets in train_dl:
                self.index.add(embeddings.to(self.device))
                labels.append(targets.to(self.device))
        self.labels = torch.cat(labels)

    def valid(self, valid_data: EmbeddingDataset):

        self.acc.reset()
        with torch.utils.data.DataLoader(
            valid_data, batch_size=256, shuffle=False, num_workers=0
        ) as valid_dl:
            for embeddings, targets in valid_dl:
                _, indices = self.index.search(embeddings.to(self.device), self.k)
                predictions = self.labels[indices].mode()[0]

                try:  # catch bug from torchmetric?
                    self.acc.update(predictions, targets.to(self.device))
                except Exception as e:
                    warn(f"Could not compute accuracy: {str(e)})")

        return float(self.acc.compute())

    def cleanup(self):
        if self.index is not None:
            self.index.reset()
        self.index = None
        self.labels = None
        self.acc = None
        self.device = torch.device("cpu")
