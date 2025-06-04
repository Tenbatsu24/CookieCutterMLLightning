import torch

from torchmetrics import Accuracy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from ml.probing.util import Analysis, EmbeddingDataset


# LinearDiscriminantAnalysis supported on GPU:
# https://scikit-learn.org/stable/modules/array_api.html#pytorch-support
class LinDiscrAnalysis(Analysis):
    from sklearn import config_context

    def __init__(self):
        super().__init__()
        self.clf = None
        self.opt = None
        self.acc = None
        self.device = torch.device("cpu")

    @torch.inference_mode(False)
    @torch.no_grad()
    @config_context(array_api_dispatch=True)
    def prepare(
        self,
        n_features: int,
        n_classes: int,
        device: torch.device = None,
        generator: torch.Generator = None,
    ):
        self.clf = LDA()
        self.acc = Accuracy(task="multiclass", num_classes=n_classes)

        if device and device.type == "cuda":
            self.acc = self.acc.to(device=device)
            self.device = device

    @torch.inference_mode(False)
    @torch.no_grad()
    @config_context(array_api_dispatch=True)
    def train(self, train_data: EmbeddingDataset, verbose=True):
        with torch.utils.data.DataLoader(
            train_data, batch_size=256, shuffle=False, num_workers=0
        ) as train_dl:
            X, y = [], []
            for embeddings, targets in train_dl:
                X.append(embeddings)
                y.append(targets)

        X, y = torch.cat(X), torch.cat(y)
        self.clf.fit(X.to(self.device), y.to(self.device))

        del X, y

    @config_context(array_api_dispatch=True)
    def valid(self, valid_data: EmbeddingDataset):
        self.acc.reset()
        with torch.utils.data.DataLoader(
            valid_data, batch_size=256, shuffle=False, num_workers=0
        ) as valid_dl:
            for embeddings, targets in valid_dl:
                predictions = self.clf.predict_proba(embeddings.to(self.device))
                self.acc.update(predictions, targets.to(self.device))

        return float(self.acc.compute())

    def cleanup(self):
        self.clf = None
        self.acc = None
        self.device = torch.device("cpu")
