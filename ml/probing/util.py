from time import time
from typing import List, Dict, Union

import wandb
import torch
import torch.nn as nn
import lightning.pytorch as pl

from loguru import logger
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedShuffleSplit

NESTED_METRICS = List[Dict[str, Union[str, Dict[str, float | List]]]]


# inject a new method to DataLoader __enter__ and __exit__ methods to allow for context manager
def _enter(self):
    return self


DataLoader.__enter__ = _enter


def _exit(self, *args):
    if hasattr(self, "_num_workers") and self._num_workers > 0:
        self._shutdown_workers()
    if hasattr(self, "_persistent_workers") and self._persistent_workers:
        self._shutdown_workers()
    if hasattr(self, "_dataset") and self._dataset is not None:
        del self._dataset
    if hasattr(self, "_sampler") and self._sampler is not None:
        del self._sampler
    if hasattr(self, "_batch_sampler") and self._batch_sampler is not None:
        del self._batch_sampler
    if hasattr(self, "_collate_fn") and self._collate_fn is not None:
        del self._collate_fn
    if hasattr(self, "_worker_init_fn") and self._worker_init_fn is not None:
        del self._worker_init_fn
    if hasattr(self, "_worker_queue") and self._worker_queue is not None:
        self._worker_queue.close()
        self._worker_queue.join_thread()

    return False


DataLoader.__exit__ = _exit


def wrap_to_tuple_maybe(element):
    if not isinstance(element, tuple):
        return (element,)
    return element


class EmbeddingDataset(Dataset):

    def __init__(self, embeddings: List[torch.Tensor], targets: List[torch.Tensor]):
        self.embeddings = embeddings
        self.targets = targets

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.targets[idx]

    def dim(self):
        return self.embeddings[0].shape[-1]


class Analysis(object):
    NEEDS_PREP = True
    NEEDS_TRAINING = True

    def prepare(
        self,
        n_features: int,
        n_classes: int,
        device: torch.device = None,
        generator: torch.Generator = None,
    ) -> None:
        raise NotImplementedError()

    def train(self, train_data: EmbeddingDataset) -> None:
        raise NotImplementedError()

    def valid(self, valid_data: EmbeddingDataset) -> Union[float, dict[str], str]:
        raise NotImplementedError()

    def cleanup(self) -> None:
        raise NotImplementedError()

    def __del__(self):
        self.cleanup()


@torch.no_grad()
def load_data(encoder: nn.Module, dl: DataLoader, device: torch.device = None):
    loading_pbar = tqdm(dl, leave=False)
    loading_pbar.set_description("Loading embeddings")

    if hasattr(encoder, "toggle_image_latent"):
        initial_values = {
            "patch_latent": encoder.patch_latent,
            "image_latent": encoder.image_latent,
        }
        logger.info(f"Switching encoder to return image latent. Initially got: {initial_values}")

        # make "patch_latent" False and "image_latent" True
        encoder.toggle_image_latent(True)
        encoder.toggle_patch_latent(False)

    # store training mode and switch to eval
    mode = encoder.training
    encoder.eval()

    data, mem = [], 0
    flag = False
    for batch in loading_pbar:
        inputs, targets = batch[0], batch[1]

        if device:
            inputs, targets = inputs.to(device), targets.to(device)

        *_, embeddings = wrap_to_tuple_maybe(encoder(inputs, return_latent=True))
        if not flag:
            flag = True

        # embeddings: (batch_size, n_features), targets: (batch_size, ...)
        data.extend(list(zip(embeddings.cpu(), targets.cpu())))

        mem += embeddings.element_size() * embeddings.nelement()
        loading_pbar.set_postfix({"mem": f"{mem * 1e-6:.1f}MB"})

    # restore previous mode
    encoder.train(mode)

    if hasattr(encoder, "toggle_image_latent"):
        logger.info(f"Restoring encoder to initial state: {initial_values}")
        encoder.toggle_image_latent(initial_values["image_latent"])
        encoder.toggle_patch_latent(initial_values["patch_latent"])

    return data


class Prober(pl.Callback):
    def __init__(
        self,
        encoders: Dict[str, nn.Module],
        analyses: Dict[str, Analysis],
        valid_dl: DataLoader,
        probe_every: int = 1,
        seed=None,
        train_dl: DataLoader = None,
        n_classes: int = 0,
        data_subsets=(0.01, 0.05, 0.1, 0.5, 1.0),
        val_data_subsets=(1.0,),
        val_analyses=("linear", "knn-5", "pca"),
    ):
        super().__init__()

        self.encoders = encoders
        self.analyses = analyses
        self.valid_dl = valid_dl

        self.train_dl = train_dl
        self.n_classes = n_classes

        self.seed = seed
        self.probe_every = probe_every
        self.data_subsets = data_subsets

        self.val_analyses = val_analyses
        self.val_data_subsets = val_data_subsets

    @torch.no_grad()
    def eval_probe(
        self, train_data: EmbeddingDataset, valid_data: EmbeddingDataset, device=None, test=False
    ):
        metrics = {}
        analyses = (
            {k: v for k, v in self.analyses.items() if k in self.val_analyses}
            if (not test)
            else self.analyses
        )

        for analysis_id, analysis in analyses.items():

            try:
                if analysis.NEEDS_TRAINING:
                    if train_data is None:
                        logger.error(
                            f"Analysis {analysis_id} needs training data but none was provided."
                        )
                        continue
                    if len(train_data) == 0:
                        logger.error(
                            f"Analysis {analysis_id} needs training data but none was provided."
                        )

                if analysis.NEEDS_TRAINING and not test:
                    subsets = self.val_data_subsets
                else:
                    subsets = self.data_subsets

                if analysis.NEEDS_TRAINING:
                    for subset in subsets:
                        # prepare probe: everything is random if seed is None
                        generator = torch.Generator(device=device)
                        if self.seed is None:
                            generator.seed()
                        else:
                            generator.manual_seed(self.seed)

                        if analysis.NEEDS_PREP:
                            n_features = valid_data.dim()
                            analysis.prepare(n_features, self.n_classes, device, generator)

                        logger.info(f"Training {analysis_id} on {subset:.0%} of training data")

                        if subset == 1.0:
                            train_subset = train_data
                        else:
                            sss = StratifiedShuffleSplit(
                                n_splits=1, train_size=subset, random_state=self.seed
                            )
                            train_idx, _ = next(sss.split(train_data, train_data.targets))
                            train_subset = torch.utils.data.Subset(train_data, train_idx)

                        analysis.train(train_subset)
                        metrics[f"{analysis_id}/{subset}"] = analysis.valid(valid_data)

                        analysis.cleanup()  # free space
                else:
                    # prepare probe: everything is random if seed is None
                    generator = torch.Generator(device=device)
                    if self.seed is None:
                        generator.seed()
                    else:
                        generator.manual_seed(self.seed)

                    if analysis.NEEDS_PREP:
                        n_features = valid_data.dim()
                        analysis.prepare(n_features, self.n_classes, device, generator)

                    analysis.train(train_data)
                    metrics[analysis_id] = analysis.valid(valid_data)

                    analysis.cleanup()  # free space
            except Exception as e:
                logger.error(f"Error in analysis {analysis_id}: {e}")
                if hasattr(analysis, "cleanup"):
                    try:
                        analysis.cleanup()
                    except Exception as e:
                        logger.error(f"Error in cleanup of analysis {analysis_id}: {e}")
                else:
                    logger.error(f"Analysis {analysis_id} has no cleanup method.")

        return metrics

    @torch.no_grad()
    def probe(self, device_enc=None, device_emb=None, verbose=True, test=False):
        """Args:
        - device_enc: device to encode images, should match encoder (None defaults to cpu)
        - device_emb: device to analyze embeddings, batches are used if possible (None defaults to device_enc)
        """
        device_emb = device_emb or device_enc  # use encoder device for embeddings by default

        out = {}
        for enc_id, encoder in self.encoders.items():
            if verbose:
                tqdm.write(
                    f"\nStarting analyses {list(self.analyses.keys())} of {enc_id}..", end=""
                )
                t = time()

            # prepare data: training data is random if seed is None and shuffle=True
            if self.train_dl is not None and any(
                [analysis.NEEDS_TRAINING for analysis in self.analyses.values()]
            ):
                if self.train_dl.generator is not None:
                    if self.seed is None:
                        self.train_dl.generator.seed()
                    else:
                        self.train_dl.generator.manual_seed(self.seed)

                # load data
                train_data = EmbeddingDataset(
                    *zip(*load_data(encoder, self.train_dl, device=device_enc))
                )  # store embeddings on cpu
            else:
                train_data = None

            valid_data = EmbeddingDataset(
                *zip(*load_data(encoder, self.valid_dl, device=device_enc))
            )  # store embeddings on cpu

            # evaluate data
            metrics = self.eval_probe(
                train_data, valid_data, device=device_emb, test=test
            )  # move to device_emb on demand

            for key, val in metrics.items():
                metric_key = f"probe/{enc_id}" if key == "" else f"probe/{enc_id}/{key}"
                out[metric_key] = val

            # cleanup data
            del train_data, valid_data

        return out

    def handle_probe_metrics(self, pl_module, key, value, prefix):
        if not isinstance(value, list):
            if isinstance(value, dict):
                pl_module.log_dict(
                    {
                        "epoch": pl_module.current_epoch,
                        **{f"{prefix}/{key}/{i_key}": i_value for i_key, i_value in value.items()},
                    }
                )
            else:
                pl_module.log_dict({"epoch": pl_module.current_epoch, f"{prefix}/{key}": value})
        else:  # it is a list with structure {"type": ..., "data": ...}
            for nested_metrics in value:
                if nested_metrics["type"] == "image":
                    name = nested_metrics["name"]
                    wandb.log(
                        {
                            "epoch": pl_module.current_epoch,
                            f"{prefix}/{key}/{name}": wandb.Image(nested_metrics["data"]),
                        }
                    )
                else:
                    self.handle_probe_metrics(pl_module, key, nested_metrics["data"], prefix)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.probe_every is None:
            # probe every epoch
            for key, value in self.probe(pl_module.device).items():
                self.handle_probe_metrics(pl_module, key, value, "val_p")
        else:
            if (trainer.current_epoch + 1) % self.probe_every == 0:
                # only probe every so many epochs
                for key, value in self.probe(pl_module.device).items():
                    self.handle_probe_metrics(pl_module, key, value, "val_p")

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        for key, value in self.probe(pl_module.device, test=True).items():
            self.handle_probe_metrics(pl_module, key, value, "test_p")
