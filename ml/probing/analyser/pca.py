from typing import List, Tuple

import torch
import wandb
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from ml.config import FIGURES_DIR
from ml.probing.util import Analysis


class PCAAnalysis(Analysis):
    NEEDS_PREP = True

    def __init__(self, n_components: int = 2):
        super().__init__()
        self.pca = None
        self.n_components = n_components

    def prepare(
        self,
        n_features: int,
        n_classes: int,
        device: torch.device = None,
        generator: torch.Generator = None,
    ) -> None:
        from sklearn.decomposition import PCA

        self.n_components = n_features
        self.pca = PCA(n_components=self.n_components)

    def train(self, train_data: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        pass

    def valid(self, valid_data: List[Tuple[torch.Tensor, torch.Tensor]]) -> dict[str, str]:
        # perform the PCA analysis on the valid data. Create a copy, take to cpu and then perform pca

        valid_pbar = tqdm(valid_data, leave=False)
        valid_pbar.set_description("PCA Analysis")

        data = []
        classes = []
        for embeddings, targets in valid_pbar:
            data.append(embeddings.clone().cpu())
            classes.append(targets.clone().cpu())

        data = torch.cat(data).numpy()
        pcad_data = self.pca.fit_transform(data)
        classes = torch.cat(classes).numpy()

        # create a scatter plot of the data
        plt.figure(figsize=(8, 8))
        plt.scatter(pcad_data[:, 0], pcad_data[:, 1], c=classes, cmap="viridis", s=10)
        plt.tight_layout()

        to_save_dir = wandb.run.dir or FIGURES_DIR

        # save the plot to wandb and to the wandb.run directory
        plt.savefig(f"{to_save_dir}/PCA.png")
        plt.close()

        return {"type": "image", "data": f"{to_save_dir}/PCA.png"}

    def cleanup(self) -> None:
        self.pca = None
