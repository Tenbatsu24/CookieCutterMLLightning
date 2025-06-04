from typing import List, Tuple

import numpy as np
import torch
import wandb

import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from ml.config import FIGURES_DIR
from ml.probing.util import Analysis

plt.switch_backend("agg")


class PCAAnalysis(Analysis):
    NEEDS_PREP = True
    NEEDS_TRAINING = False

    def __init__(self, n_components: int = 2):
        super().__init__()
        self.pca = None
        self.tsne = None
        self.n_components = n_components

    def prepare(
        self,
        n_features: int = None,
        n_classes: int = None,
        device: torch.device = None,
        generator: torch.Generator = None,
    ) -> None:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA

        self.pca = PCA(n_components=self.n_components)
        self.tsne = TSNE(n_components=2)

    def train(self, train_data: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        pass

    def valid(self, valid_data: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[dict[str, str]]:
        # perform the PCA analysis on the valid data. Create a copy, take to cpu and then perform pca

        valid_pbar = tqdm(valid_data, leave=False)
        valid_pbar.set_description("PCA Analysis")

        data, classes = zip(*valid_pbar)

        data = torch.stack(data, dim=0).numpy()
        pcad_data = self.pca.fit_transform(data)
        classes = np.array(classes)

        explained_variance = self.pca.explained_variance_ratio_
        # get the top2 explained variance ratios
        idxs = np.argsort(explained_variance)[-2:][::-1]

        # create a scatter plot of the data
        plt.figure(figsize=(8, 8))
        plt.scatter(pcad_data[:, idxs[0]], pcad_data[:, idxs[1]], c=classes, cmap="viridis", s=10)

        # explained variance ratio
        plt.title(f"PCA Analysis (Explained Variance: {np.sum(explained_variance[idxs]):.2f})")
        plt.xlabel("PC 1 (Explained Variance: {:.2f})".format(explained_variance[idxs[0]]))
        plt.ylabel("PC 2 (Explained Variance: {:.2f})".format(explained_variance[idxs[1]]))
        plt.tight_layout()

        if wandb.run is not None:
            to_save_dir = wandb.run.dir
        else:
            # if wandb is not running, save to the figures directory
            to_save_dir = FIGURES_DIR
        # save the plot to wandb and to the wandb.run directory
        plt.savefig(f"{to_save_dir}/PCA.png")
        plt.close()

        # do tsne analysis
        tsne_data = self.tsne.fit_transform(data)

        plt.figure(figsize=(8, 8))
        plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=classes, cmap="viridis", s=10)
        plt.title(f"t-SNE Analysis")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.tight_layout()
        plt.savefig(f"{to_save_dir}/tSNE.png")
        plt.close()

        return [
            {"type": "image", "name": "PCA", "data": f"{to_save_dir}/PCA.png"},
            {"type": "image", "name": "tSNE", "data": f"{to_save_dir}/tSNE.png"},
        ]

    def cleanup(self) -> None:
        self.pca = None
        self.tsne = None
