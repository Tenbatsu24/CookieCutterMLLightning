import torch

from ml.loss import get_loss
from ml.optim import init_optims_from_config
from ml.scheduling import Schedule, Scheduler
from ml.util import STORE, MODEL_TYPE, MIX_TYPE


class Module(torch.nn.Module):
    """
    This is a base class for all modules. It inherits from torch.nn.Module and provides
    a common interface for all modules.
    """

    def __init__(self, config, normalisation):
        super().__init__()
        self.config = config

        self.before_batch_train = self.make_before_batch_train(normalisation)
        self.before_batch_val = normalisation

        self.model = self.make_model()

        self.optimisers, self.scheduler = self.make_opt_sched()

        self.criterion = self.make_criterion()

    def make_before_batch_train(self, normalisation):
        steps = []
        if MIX_TYPE in self.config:
            steps.append(STORE.get(MIX_TYPE, self.config.mix.type)(self.config.mix.params))
        steps.append(normalisation)
        return torch.nn.Sequential(*steps)

    def forward(self, x):
        """
        Forward pass through the model.
        """
        return self.model(x)

    def train_step(self, batch, _):
        """
        Training step. This method should be overridden by subclasses.
        """
        batch = self.before_batch_train(batch)
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        return loss, {"loss": loss.item()}

    def val_step(self, batch, _):
        """
        Validation step. This method should be overridden by subclasses.
        """
        batch = self.before_batch_val(batch)
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        return loss, {"loss": loss.item()}

    def test_step(self, batch, batch_idx):
        """
        Test step. This method should be overridden by subclasses.
        """
        return self.val_step(batch, batch_idx)

    def get_optimizers(self):
        """
        Get the optimizers for the model.
        """
        return self.optimisers

    def make_model(self):
        """
        Create the model.
        :return:
        """
        return STORE.get(MODEL_TYPE, self.config.model.type)(self.config.model.params)

    def make_opt_sched(self):
        """
        Create the optimisers.
        :return:
        """
        opts, group_names = init_optims_from_config(self.config, self.model)

        scheduler = Scheduler()
        for key, sched in self.config.scheduler.items():

            if key == "lr":
                for group_name in group_names:
                    scheduler.add(opts[group_name], key, Schedule.parse(sched))

            if key == "wd":
                for group_name in filter(lambda x: x == "other", group_names):
                    scheduler.add(opts[group_name], key, Schedule.parse(sched))

        return opts, scheduler

    def make_criterion(self):
        """
        Create the loss function.
        :return:
        """
        return get_loss(self.config.loss.type, self.config.loss.params)

    def get_custom_callbacks(self):
        """
        Get the custom callbacks for the model.
        :return:
        """
        return [self.scheduler]
