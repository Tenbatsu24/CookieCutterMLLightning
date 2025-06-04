import torch

from torchmetrics import Accuracy

from ml.probing.util import Analysis, EmbeddingDataset


class LogRegAnalysis(Analysis):
    def __init__(self):
        super().__init__()
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
        from sklearn.linear_model import LogisticRegression

        self.clf = LogisticRegression()
        self.acc = Accuracy(task="multiclass", num_classes=n_classes)

    @torch.inference_mode(False)
    @torch.no_grad()
    def train(self, train_data: EmbeddingDataset, verbose=True):
        with torch.utils.data.DataLoader(
            train_data, batch_size=256, shuffle=False, num_workers=0
        ) as train_dl:

            X, y = [], []
            for embeddings, targets in train_dl:
                X.append(embeddings)
                y.append(targets)

        X, y = torch.cat(X).numpy(), torch.cat(y).numpy()
        self.clf.fit(X, y)
        del X, y

    def valid(self, valid_data: EmbeddingDataset):
        self.acc.reset()

        with torch.utils.data.DataLoader(
            valid_data, batch_size=256, shuffle=False, num_workers=0
        ) as valid_dl:
            for embeddings, targets in valid_dl:
                predictions = self.clf.predict_proba(embeddings.numpy())
                self.acc.update(torch.from_numpy(predictions), targets)

        return float(self.acc.compute())

    def cleanup(self):
        self.clf = None
        self.acc = None
        self.device = torch.device("cpu")
