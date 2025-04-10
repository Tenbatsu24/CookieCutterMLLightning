from typing import Union, Sequence

import torch
import lightning as pl
import torchmetrics

from lightning import Callback
from ml_collections import ConfigDict

from ml.loss import get_loss
from ml.metric import get_metric
from ml.optim import init_optims_from_config
from ml.scheduling import Schedule, Scheduler
from ml.util import STORE, MODEL_TYPE, MIX_TYPE, AUG_TYPE


class BaseTrainer(pl.LightningModule):
    def __init__(self, config: ConfigDict, normalisation: torch.nn.Module, valid_dl):
        super().__init__()

        # initialise the config
        self.config = config

        # initialise the module
        self.before_batch_train = self.make_before_batch_train()
        self.normalisation = normalisation

        self.model = self.make_model()

        self.optims, self.scheduler = self.make_opt_sched()

        self.criterion = self.make_criterion()

        metrics = self.make_metrics()
        metrics = torchmetrics.MetricCollection(metrics)
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        self.val_dl = valid_dl

        self.save_hyperparameters(self.config.to_dict())

    def make_before_batch_train(self):
        steps = []

        if AUG_TYPE in self.config:
            for aug in self.config.__getattr__(AUG_TYPE):
                steps.append(STORE.get(AUG_TYPE, aug.type)(**aug.params))

        if MIX_TYPE in self.config:
            mix = self.config.__getattr__(MIX_TYPE)
            steps.append(STORE.get(MIX_TYPE, mix.type)(**mix.params))

        return torch.nn.Sequential(*steps)

    def make_model(self):
        """
        Create the model.
        :return:
        """
        return STORE.get(MODEL_TYPE, self.config.model.type)(**self.config.model.params)

    def make_opt_sched(self):
        """
        Create the optimisers.
        :return:
        """
        opt, group_names = init_optims_from_config(self.config, self.model)

        scheduler = Scheduler()
        for key, sched in self.config.scheduler:

            if key == "lr":
                for group_num, group_name in enumerate(group_names):
                    scheduler.add(opt.param_groups[group_num], key, Schedule.parse(sched))

            if key == "weight_decay":
                for group_num, group_name in filter(
                    lambda x: x[1] == "other", enumerate(group_names)
                ):
                    scheduler.add(opt.param_groups[group_num], key, Schedule.parse(sched))

        return [opt], scheduler

    def make_criterion(self):
        """
        Create the loss function.
        :return:
        """
        return get_loss(self.config.loss.type)(**self.config.loss.params)

    def make_metrics(self):
        """
        Create the metrics.
        :return:
        """
        metrics = {
            short_name: get_metric(metric_dict.type)(**metric_dict.params)
            for short_name, metric_dict in self.config.metrics.items()
        }
        return metrics

    def batch_to_loss(self, batch):
        x, y = batch
        y_hat = self.model(self.normalisation(x))
        loss = self.criterion(y_hat, y)
        return loss, y, y_hat

    def log_loss(self, loss, prefix, prog_bar, on_epoch, on_step):
        # loss can be a dict
        if isinstance(loss, dict):
            for key, value in loss.items():
                self.log(
                    f"{prefix}/{key}", value, prog_bar=prog_bar, on_epoch=on_epoch, on_step=on_step
                )
            return loss["loss"]
        else:
            self.log(f"{prefix}/loss", loss, prog_bar=prog_bar, on_epoch=on_epoch, on_step=on_step)
            return loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        batch = self.before_batch_train(batch)
        loss, y, y_hat = self.batch_to_loss(batch)

        loss = self.log_loss(loss, prefix="train", prog_bar=True, on_epoch=False, on_step=True)
        batch_metrics = self.train_metrics(y_hat, y)

        self.log_dict(batch_metrics, prog_bar=True, on_epoch=False, on_step=True)

        return loss

    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        loss, y, y_hat = self.batch_to_loss(batch)

        loss = self.log_loss(loss, prefix="val", prog_bar=True, on_epoch=True, on_step=False)
        self.val_metrics.update(y_hat, y)

        return loss

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), prog_bar=True, on_epoch=True, on_step=False)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        loss, y, y_hat = self.batch_to_loss(batch)

        loss = self.log_loss(loss, prefix="test", prog_bar=True, on_epoch=True, on_step=False)
        self.test_metrics.update(y_hat, y)

        return loss

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics.compute(), prog_bar=True, on_epoch=True, on_step=False)
        self.test_metrics.reset()

    def configure_optimizers(self):
        return self.optims, []

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        """
        Override this method to configure callbacks for the trainer.
        """

        callbacks = [self.scheduler]

        if self.config.log_latent:
            from ml.probing import Prober
            from ml.probing.analyser import PCAAnalysis

            valid_dl = self.val_dl
            encoders = {"model": self.model}
            analyses = {"pca": PCAAnalysis(n_components=2)}

            prober = Prober(
                encoders=encoders,
                analyses=analyses,
                valid_dl=valid_dl,
                probe_every=1,
            )

            callbacks.append(prober)

        return callbacks
