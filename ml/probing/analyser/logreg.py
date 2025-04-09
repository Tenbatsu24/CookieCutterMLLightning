from typing import List, Tuple

import torch

from tqdm.auto import tqdm
from torchmetrics import Accuracy

from ml.probing.util import Analysis


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

        self.clf = LogisticRegression(multi_class="multinomial")
        self.acc = Accuracy(task="multiclass", num_classes=n_classes)

    @torch.inference_mode(False)
    @torch.no_grad()
    def train(self, train_data: List[Tuple[torch.Tensor, torch.Tensor]], verbose=True):
        X, y = zip(*train_data)
        X, y = torch.cat(X).numpy(), torch.cat(y).numpy()
        self.clf.fit(X, y)

    def valid(self, valid_data: List[Tuple[torch.Tensor, torch.Tensor]]):
        valid_pbar = tqdm(valid_data, leave=False)
        valid_pbar.set_description("Validation")

        self.acc.reset()
        for embeddings, targets in valid_pbar:
            predictions = self.clf.predict_proba(embeddings.numpy())
            self.acc.update(torch.from_numpy(predictions), targets)
            valid_pbar.set_postfix({"acc": float(self.acc.compute())})

        return float(self.acc.compute())

    def cleanup(self):
        self.clf = None
        self.acc = None
        self.device = torch.device("cpu")
