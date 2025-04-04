import lightning.pytorch as pl
import torchvision.transforms as T

from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger

from ml.module import Module
from ml.trainer import BaseTrainer, SAMTrainer
from ml.util import STORE, DATA_TYPE
from ml.config import PROCESSED_DATA_DIR, MODELS_DIR, WANDB_PROJECT


def prepare_mnist(batch_size, ds_name):
    mean, std = (0.1307,), (0.3081,)
    train_transform = T.Compose(
        [T.RandomCrop(28, padding=4), T.ToTensor(), T.Normalize(mean=mean, std=std)]
    )

    val_transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])

    ds = STORE.get(DATA_TYPE, ds_name)

    train_dataset = ds(PROCESSED_DATA_DIR, train=True, download=True, transform=train_transform)
    val_dataset = ds(PROCESSED_DATA_DIR, train=False, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
    )

    return train_loader, val_loader, T.Normalize(mean=mean, std=std)


def prepare_cifar(batch_size, ds_name):
    mean, std = (0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)
    train_transform = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, padding=4),
            T.ToTensor(),
        ]
    )

    val_transform = T.Compose(
        [
            T.ToTensor(),
        ]
    )

    ds = STORE.get(DATA_TYPE, ds_name)

    train_dataset = ds(PROCESSED_DATA_DIR, train=True, download=True, transform=train_transform)
    val_dataset = ds(PROCESSED_DATA_DIR, train=False, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
    )

    return train_loader, val_loader, T.Normalize(mean=mean, std=std)


def prepare_im(batch_size, ds_name):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_transform = T.Compose(
        [
            T.Resize(256),
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]
    )

    val_transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
        ]
    )

    ds = STORE.get(DATA_TYPE, ds_name)

    train_dataset = ds(PROCESSED_DATA_DIR, split="train", download=True, transform=train_transform)
    val_dataset = ds(PROCESSED_DATA_DIR, split="val", transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
    )

    return train_loader, val_loader, T.Normalize(mean=mean, std=std)


def train(cfg, fast_dev_run=False):
    logger = WandbLogger(
        project=WANDB_PROJECT, name=cfg.run_name, id=cfg.run_id, allow_val_change=True
    )
    checkpoints_dir = MODELS_DIR / logger.experiment.project / logger.experiment.id
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    last_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoints_dir,
        monitor=None,
        filename="last",
    )
    best_loss_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoints_dir,
        save_top_k=1,
        filename="best",
        **cfg.pl.checkpoint.to_dict(),
    )
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[
            last_checkpoint_callback,
            best_loss_checkpoint_callback,
            pl.callbacks.LearningRateMonitor(logging_interval="step", log_weight_decay=True),
        ],
        **cfg.pl.trainer.to_dict(),
        fast_dev_run=fast_dev_run,
    )

    ds_name = cfg.dataset.name
    batch_size = cfg.batch_size

    if ds_name in ["m", "fm"]:
        train_loader, val_loader, norm = prepare_mnist(batch_size, ds_name)
    elif ds_name in ["c10", "c100"]:
        train_loader, val_loader, norm = prepare_cifar(batch_size, ds_name)
    elif ds_name in ["im10"]:
        train_loader, val_loader, norm = prepare_im(batch_size, ds_name)
    else:
        raise ValueError(f"Unknown dataset {ds_name}")

    module = SAMTrainer(cfg, Module(cfg, normalisation=norm))

    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    from ml_collections import ConfigDict

    _cfg = {
        "dataset": {
            "name": "c10",
        },
        "batch_size": 128,
        "run_name": "test_run-sam",
        "run_id": None,
        "pl": {
            "checkpoint": {"monitor": "val/acc", "mode": "max"},
            "trainer": {
                "accelerator": "gpu",
                "precision": "32",
                "devices": 1,
                "strategy": 'auto',
                "num_sanity_val_steps": 0,
                "gradient_clip_val": 0.5,
                "gradient_clip_algorithm": "norm",
                "accumulate_grad_batches": 1,
                "val_check_interval": 1.0,
                "limit_val_batches": 1.0,
                "benchmark": True,
                "deterministic": True,
                "max_epochs": 10,
            },
        },
        # "mix": {"type": "mixup", "params": {"alpha": 0.2, "num_classes": 10}}
        "scheduler": [
            ("lr", "CosSched(0.1, 1e-6)"),
            ("weight_decay", "CosSched(2e-5, 1e-4)"),
        ],
        "opt": {
            "type": "SAM",  # SGD
            "base_type": "SGD",
            "params": {
                "rho": 0.5, "adaptive": True,
                "lr": 0.1, "momentum": 0.9, "weight_decay": 2e-5, "nesterov": True
            },
        },
        "model": {
            "type": "rn18",
            "params": {"num_classes": 10, "in_channels": 3, "return_latent": False},
        },
        "loss": {
            "type": "CrossEntropyLoss",
            "params": {"weight": None, "label_smoothing": 1e-5, "reduction": "mean"},
        },
    }

    _cfg = ConfigDict(_cfg, convert_dict=True)

    train(_cfg, fast_dev_run=False)
