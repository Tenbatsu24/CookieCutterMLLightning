from time import time
from typing import List, Tuple, Dict, Union

import torch
import torch.nn as nn
import lightning.pytorch as pl
import wandb

from tqdm.auto import tqdm
from torch.utils.data import DataLoader


def wrap_to_tuple_maybe(element):
    if not isinstance(element, tuple):
        return (element,)
    return element


class Analysis(object):
    NEEDS_PREP = True

    def prepare(
        self,
        n_features: int,
        n_classes: int,
        device: torch.device = None,
        generator: torch.Generator = None,
    ) -> None:
        raise NotImplementedError()

    def train(self, train_data: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        raise NotImplementedError()

    def valid(
        self, valid_data: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Union[float, dict[str], str]:
        raise NotImplementedError()

    def cleanup(self) -> None:
        raise NotImplementedError()

    def __del__(self):
        self.cleanup()


@torch.no_grad()
def load_data(encoder: nn.Module, dl: DataLoader, device: torch.device = None):
    loading_pbar = tqdm(dl, leave=False)
    loading_pbar.set_description(f"Loading embeddings")

    # store training mode and switch to eval
    mode = encoder.training
    encoder.eval()

    data, mem = [], 0
    for batch in loading_pbar:
        inputs, targets = batch[0], batch[1]

        if device:
            inputs, targets = inputs.to(device), targets.to(device)

        *_, embeddings = wrap_to_tuple_maybe(encoder(inputs, return_latent=True))
        data.append((embeddings.contiguous().squeeze().cpu(), targets.cpu()))

        mem += embeddings.element_size() * embeddings.nelement()
        loading_pbar.set_postfix({"mem": f"{mem * 1e-6:.1f}MB"})

    # restore previous mode
    encoder.train(mode)
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
    ):
        super().__init__()

        self.encoders = encoders
        self.analyses = analyses
        self.valid_dl = valid_dl

        self.train_dl = train_dl
        self.n_classes = n_classes

        self.seed = seed
        self.probe_every = probe_every

    @torch.no_grad()
    def eval_probe(
        self,
        train_data: List[Tuple[torch.Tensor, torch.Tensor]],
        valid_data: List[Tuple[torch.Tensor, torch.Tensor]],
        device=None,
    ):
        metrics = {}
        for analysis_id, analysis in self.analyses.items():

            # prepare probe: everything is random if seed is None
            generator = torch.Generator(device=device)
            if self.seed is None:
                generator.seed()
            else:
                generator.manual_seed(self.seed)

            if analysis.NEEDS_PREP:
                n_features = valid_data[0][0].shape[-1]
                analysis.prepare(n_features, self.n_classes, device, generator)

            analysis.train(train_data)
            metrics[analysis_id] = analysis.valid(valid_data)

            analysis.cleanup()  # free space

        return metrics

    @torch.no_grad()
    def probe(self, device_enc=None, device_emb=None, verbose=True):
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
            if self.train_dl is not None:
                if self.train_dl.generator is not None:
                    if self.seed is None:
                        self.train_dl.generator.seed()
                    else:
                        self.train_dl.generator.manual_seed(self.seed)

                # load data
                train_data = load_data(
                    encoder, self.train_dl, device=device_enc
                )  # store embeddings on cpu
            else:
                train_data = None

            valid_data = load_data(
                encoder, self.valid_dl, device=device_enc
            )  # store embeddings on cpu

            # evaluate data
            metrics = self.eval_probe(
                train_data, valid_data, device=device_emb
            )  # move to device_emb on demand

            for key, val in metrics.items():
                metric_key = f"probe/{enc_id}" if key == "" else f"probe/{enc_id}/{key}"
                out[metric_key] = val

            if verbose:
                t = time() - t
                tqdm.write(f" ..{enc_id} took {int(t // 60):02d}:{int(t % 60):02d}min", end="")

            # cleanup data
            del train_data, valid_data

        if verbose:
            tqdm.write(
                " => "
                + str(
                    {
                        key: f"{val:.3}"
                        for key, val in filter(lambda kv: not isinstance(kv[1], dict), out.items())
                    }
                )
            )

        return out

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # if trainer.current_epoch % self.probe_every == 0: # only probe every so many epochs
        for key, value in self.probe(pl_module.device).items():
            if not isinstance(value, dict):
                pl_module.log(key, value)
            else:
                if value["type"] == "image":
                    wandb.log({key: wandb.Image(value["data"])})
                else:
                    raise NotImplementedError(f"Unsupported type {value['type']} for {key}")
