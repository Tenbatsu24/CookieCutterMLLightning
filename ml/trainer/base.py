from typing import Any, Optional, Union, Sequence

import torch
import torch.nn as nn
import lightning as pl
from lightning import Callback

from ml_collections import ConfigDict

from ml.module.default import Module


class BaseTrainer(pl.LightningModule):
    def __init__(self, config: ConfigDict, module: Module):
        super().__init__()

        # initialise the config
        self.config = config

        # initialise the module
        self.module = module

        self.save_hyperparameters(self.config.to_dict())

    def forward(self, x):
        return self.module(x)

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:
        self.module.step_scheduler()

    def training_step(self, batch, batch_idx):
        loss, metrics_dict = self.module.train_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, on_epoch=False, on_step=True)
        self.log_dict(metrics_dict, prog_bar=False, on_epoch=False, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics_dict = self.module.val_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log_dict(metrics_dict, prog_bar=False, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        loss, metrics_dict = self.module.test_step(batch, batch_idx)
        self.log("test/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log_dict(metrics_dict, prog_bar=False, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        return self.module.get_optimizers(), []

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        """
        Override this method to configure callbacks for the trainer.
        """
        return self.module.get_custom_callbacks()


class SAMTrainer(BaseTrainer):

    def __init__(self, config, module):
        super().__init__(config, module)

        self.automatic_optimization = False

    @classmethod
    def set_bn_eval(cls, m):
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()

    @classmethod
    def set_bn_train(cls, m):
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.train()

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()

        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        self.module.apply(self.set_bn_eval)
        loss_1, _ = self.module.train_step(batch, batch_idx)

        self.manual_backward(loss_1)
        [optimizer.first_step(zero_grad=True) for optimizer in optimizers]

        # second forward-backward pass
        self.module.apply(self.set_bn_train)
        loss_2, metrics_dict = self.module.train_step(batch, batch_idx)

        self.log("train/loss", loss_2, prog_bar=True, on_epoch=False, on_step=True)
        self.log(
            "train/sharpness",
            torch.abs(loss_1 - loss_2),
            prog_bar=False,
            on_epoch=False,
            on_step=True,
        )
        self.log_dict(metrics_dict, prog_bar=False, on_epoch=False, on_step=True)

        self.trainer.fit_loop.epoch_loop.manual_optimization._on_before_step()

        self.manual_backward(loss_2)
        [optimizer.second_step(zero_grad=True) for optimizer in optimizers]

        self.trainer.fit_loop.epoch_loop.manual_optimization._on_after_step()

        return loss_1
