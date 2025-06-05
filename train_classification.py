import json
import argparse

from pathlib import Path
from pprint import pprint

import torch
import lightning.pytorch as pl

from loguru import logger
from ml_collections import ConfigDict
from lightning.pytorch.loggers import WandbLogger

from ml.trainer import BaseTrainer
from ml.wandb_util import MyModelCheckpoint
from ml.config import MODELS_DIR, WANDB_PROJECT
from ml.data import make_loaders, generalisation_test


def set_run_name(cfg):
    run_name = f"{cfg.dataset.name}"

    if hasattr(cfg.dataset, "aug"):
        for aug in cfg.dataset.aug:
            run_name += f"_{aug['id']}"

    if hasattr(cfg, "aug"):
        for aug in cfg.aug:
            run_name += f"_{aug['id']}"

    if hasattr(cfg, "label_noise") and cfg.label_noise.rate > 0:
        run_name += f"_ln={cfg.label_noise.rate}"
    if hasattr(cfg, "subset") and cfg.subset.pct > 0:
        run_name += f"_sub={cfg.subset.pct / 100:.2f}"

    if hasattr(cfg.loss, "id"):
        run_name += f"-{cfg.loss.id}"

    if hasattr(cfg, "reg") and cfg.reg.type is not None:
        run_name += f"-{cfg.reg.id}"

    run_name += f"-{cfg.model.type}"

    if hasattr(cfg, "finetune") and cfg.finetune.enable:
        run_name += "-ft"

    logger.info(f"Run name: {run_name}")
    cfg.run_name = run_name
    return cfg


def train(cfg, fast_dev_run=False, test_only=False):
    if hasattr(cfg, "run_id") and cfg.run_id is not None:
        logger.info(f"Using existing run ID: {cfg.run_id}")

    wandb_logger = WandbLogger(
        project=WANDB_PROJECT, name=cfg.run_name, id=cfg.run_id, allow_val_change=True
    )
    if hasattr(cfg, "run_id") and cfg.run_id is not None and cfg.run_id != wandb_logger.experiment.id:
        logger.error(f"Given run ID {cfg.run_id} does not match the WandB run ID {wandb_logger.experiment.id}.")
        raise ValueError("Run ID mismatch. Please check your configuration.")

    checkpoints_dir = MODELS_DIR / wandb_logger.experiment.project / wandb_logger.experiment.id
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoints directory: {checkpoints_dir}")

    last_checkpoint_callback = MyModelCheckpoint(
        dirpath=checkpoints_dir,
        save_last=True,
        monitor=None,
        filename="last",
    )

    best_checkpoints = []
    if not (isinstance(cfg.pl.checkpoint, list) or isinstance(cfg.pl.checkpoint, tuple)) and isinstance(cfg.pl.checkpoint, ConfigDict):
        checkpoints_details = [cfg.pl.checkpoint]
    else:
        checkpoints_details = cfg.pl.checkpoint

    for checkpoint_details in checkpoints_details:
        checkpoint_details = ConfigDict(checkpoint_details, convert_dict=True)
        best_checkpoints.append(
            MyModelCheckpoint(
                save_last=False,
                dirpath=checkpoints_dir,
                save_top_k=1,
                **checkpoint_details.to_dict(),
            )
        )

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[
            last_checkpoint_callback,
            *best_checkpoints,
        ],
        **cfg.pl.trainer.to_dict(),
        fast_dev_run=fast_dev_run,
    )

    train_loader, val_loader, norm, val_transform = make_loaders(cfg)

    module_cls = BaseTrainer

    # get the run id from wandb if defined and set it to the config
    if hasattr(cfg, "run_id") and cfg.run_id is not None:
        # check if there is a last.ckpt file in the checkpoints directory
        if not (checkpoints_dir / "last.ckpt").exists():
            # check if wandb has any last checkpoint
            try:
                last_checkpoint_callback.get_checkpoint_from_wandb(
                    run_id=wandb_logger.experiment.id,
                )
                logger.success(
                    f"Checkpoint {last_checkpoint_callback.filename} restored from WandB for run {cfg.run_id}."
                )
            except FileNotFoundError as e:
                logger.error(str(e))

    cfg.run_id = wandb_logger.experiment.id
    module = module_cls(config=cfg, normalisation=norm, valid_dl=val_loader, train_dl=train_loader)

    if not test_only:
        candidate_last = checkpoints_dir / "last.ckpt"
        trainer.fit(
            module,
            train_loader,
            val_loader,
            ckpt_path=candidate_last if candidate_last.exists() else None,
        )

    for best_checkpoint in best_checkpoints:
        candidate_best = best_checkpoint.best_model_path or checkpoints_dir / best_checkpoint.filename

        if not Path(candidate_best).exists():
            try:
                best_checkpoint.get_checkpoint_from_wandb(
                    run_id=wandb_logger.experiment.id,
                )
                logger.success(
                    f"Checkpoint {best_checkpoint.filename} restored from WandB for run {cfg.run_id}."
                )
            except FileNotFoundError as e:
                logger.error(str(e))
                continue

        trainer.test(module, dataloaders=val_loader, ckpt_path=candidate_best)

        if hasattr(cfg.dataset, "gen_test"):
            module.load_state_dict(torch.load(candidate_best)["state_dict"])
            generalisation_test(cfg, module, val_transform)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file",
    )

    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run a fast dev run",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = json.load(f)

    # Convert the config to a ConfigDict
    cfg = ConfigDict(cfg, convert_dict=True)
    cfg = set_run_name(cfg)

    pprint(cfg.to_dict())

    train(cfg, fast_dev_run=args.fast_dev_run)


if __name__ == "__main__":
    main()
