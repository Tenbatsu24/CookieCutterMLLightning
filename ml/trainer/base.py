from typing import Union, Sequence, Any

import wandb
import torch
import lightning as pl
import torchmetrics

from loguru import logger
from tqdm.auto import tqdm
from lightning import Callback
from ml_collections import ConfigDict

from ml.loss import get_loss
from ml.config import MODELS_DIR
from ml.metric import get_metric
from ml.mix import make_cutmix_mixup
from ml.optim import init_optims_from_config
from ml.scheduling import Schedule, Scheduler
from ml.util import STORE, MODEL_TYPE, MIX_TYPE, AUG_TYPE


class OnlyImageAugmentationWrapper(torch.nn.Module):
    """
    Wrapper for the augmentation module to only apply the augmentation to the image.
    """

    def __init__(self, aug_module):
        super().__init__()
        self.aug_module = aug_module

    def forward(self, *batch):
        x, *y = batch
        x = self.aug_module(x)
        return x, *y


class MyIdentity(torch.nn.Module):
    """
    A module that does nothing.
    """

    def forward(self, *args, **kwargs):
        if len(args) == 0:
            return args[0]
        else:
            return args


class MySequential(torch.nn.Sequential):
    """
    A sequential module that allows for the use of a custom forward method.
    """

    def forward(self, *args, **kwargs):
        for module in self:
            args = module(*args, **kwargs)
        return args


class BaseTrainer(pl.LightningModule):
    def __init__(self, config: ConfigDict, normalisation: torch.nn.Module, valid_dl, train_dl):
        super().__init__()

        # initialise the config
        self.config = config

        # initialise the module
        self.aug = self.make_aug()
        self.normalisation = normalisation

        self.model = self.make_model()

        self.optims, self.scheduler = self.make_opt_sched(self.config, self.model)

        self.criterion = self.make_criterion()

        metrics = self.make_metrics()
        metrics = torchmetrics.MetricCollection(metrics)
        if self.config.dataset.name == "in":
            # don't use the metrics for training for ImageNet
            # since that makes it slower x2 as 140gb of training data metrics will be saved
            self.train_metrics = torchmetrics.MetricCollection({}).clone(prefix="train/")
        else:
            self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        if hasattr(self.config, MIX_TYPE) and self.config.__getattr__(MIX_TYPE):
            # replace train metrics with L2 metrics
            self.train_metrics = torchmetrics.MetricCollection(
                {
                    "brier": torchmetrics.MeanSquaredError(squared=True),
                    "log_cosh": torchmetrics.LogCoshError(num_outputs=self.config.num_classes),
                }
            ).clone(prefix="train/")

        # define summary metrics for wandb
        for key, metric in self.val_metrics.items():
            if isinstance(metric, torchmetrics.Metric):
                wandb.define_metric(key, summary="max" if metric.higher_is_better else "min")

        self.val_dl: torch.utils.data.DataLoader = valid_dl
        self.train_dl: torch.utils.data.DataLoader = train_dl

        self.num_classes: int = self.config.num_classes

        self.save_hyperparameters(self.config.to_dict())

    def make_aug(self):
        steps = []

        if AUG_TYPE in self.config:
            for aug in self.config.__getattr__(AUG_TYPE):
                aug = ConfigDict(aug, convert_dict=True)
                aug_module = STORE.get(AUG_TYPE, aug.type)(**aug.params)
                if aug.image_only:
                    aug_module = OnlyImageAugmentationWrapper(aug_module)
                steps.append(aug_module)

        if MIX_TYPE in self.config and self.config.__getattr__(MIX_TYPE):
            steps.append(make_cutmix_mixup(self.config))

        return MySequential(*steps) if len(steps) > 0 else MyIdentity()

    def make_model(self):
        """
        Create the model.
        :return:
        """

        model = STORE.get(MODEL_TYPE, self.config.model.type)(**self.config.model.params)

        # check if finetune is set and find the model
        if hasattr(self.config, "finetune") and self.config.finetune.enable:
            # load the model from the checkpoint
            checkpoint = torch.load(MODELS_DIR / self.config.finetune.state_dict_path)

            # replace key with fc to classifier
            keys_to_del = []
            keys_to_replace = []

            for key in checkpoint.keys():
                if key.startswith("fc"):
                    keys_to_replace.append(key)

            for key in keys_to_replace:
                value = checkpoint[key]
                if (
                    self.config.model.params.num_classes
                    and value.shape[0] != self.config.model.params.num_classes
                ):
                    continue
                else:
                    checkpoint[key.replace("fc", "classifier")] = value
                    keys_to_del.append(key)

            for key in keys_to_del:
                del checkpoint[key]

            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)

            logger.warning(f"Missing keys: {missing_keys}")
            logger.warning(f"Unexpected keys: {unexpected_keys}")

            if hasattr(self.config.finetune, "frozen") and self.config.finetune.frozen:
                for name, param in model.named_parameters():
                    if "classifier" not in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

        return model

    @classmethod
    def make_opt_sched(cls, config, model):
        """
        Create the optimisers.
        :return:
        """
        opt, group_names = init_optims_from_config(config, model)

        scheduler = Scheduler()
        for key, sched in config.scheduler:

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

    def batch_to_loss(self, batch, train=False):
        x, y = batch

        if train:
            with torch.no_grad():
                x, y = self.aug(x, y)

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

    def log_metrics(self, batch_metrics):
        metric_dict = {
            k: torch.mean(v) if len(v.shape) > 0 else v
            for k, v in batch_metrics.items()
            if not torch.all(torch.isnan(v))
        }
        self.log_dict(metric_dict, prog_bar=True, on_epoch=False, on_step=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, y, y_hat = self.batch_to_loss(batch, train=True)

        loss = self.log_loss(loss, prefix="train", prog_bar=True, on_epoch=False, on_step=True)

        with torch.no_grad():
            if hasattr(self.config, MIX_TYPE) and self.config.__getattr__(MIX_TYPE):
                y_hat = torch.softmax(y_hat, dim=1)

            batch_metrics = self.train_metrics(y_hat, y)
            self.log_metrics(batch_metrics)

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

        callbacks: list[Any] = [self.scheduler]

        if self.config.log_latent:
            from ml.probing import Prober
            from ml.probing.analyser import PCAAnalysis

            probe_every = (
                None if not hasattr(self.config, "probe_every") else self.config.probe_every
            )
            encoders = {"model": self.model}
            analyses = {
                "mf": PCAAnalysis(),
            }

            prober = Prober(
                encoders=encoders,
                analyses=analyses,
                valid_dl=self.val_dl,
                probe_every=probe_every,
                train_dl=self.train_dl,
                n_classes=self.num_classes,
                val_analyses=("mf",),  # a subset of analyses to run on validation
            )

            callbacks.append(prober)

        return callbacks

    def manual_test(self, test_loader=None, name="nunya"):
        """
        Validate after training an epoch

        :return: A log that contains information about validation
        """

        self.model.to("cuda")
        device = next(self.model.parameters()).device

        self.test_metrics.reset()
        self.test_metrics.to(device)

        test_loader_wrapped = tqdm(
            enumerate(test_loader), desc=f"Testing... {name}", total=len(test_loader)
        )

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in test_loader_wrapped:
                data, target = data.to(device), target.to(device)

                data = self.normalisation(data)
                output = self.model(data)

                self.test_metrics.update(output, target.long())

                test_loader_wrapped.set_postfix(
                    {k: v.item() for k, v in self.test_metrics.compute().items()}
                )

        log = {k: v.item() for k, v in self.test_metrics.compute().items()}

        self.test_metrics.reset()

        return log
