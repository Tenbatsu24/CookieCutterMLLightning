import wandb
import lightning.pytorch as pl

from loguru import logger

from lightning.pytorch.callbacks import ModelCheckpoint


class MyModelCheckpoint(ModelCheckpoint):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._enable_version_counter = False

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        trainer.save_checkpoint(
            filepath, weights_only=not self.save_last
        )  # save weights only if not saving last

        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

        # check if wandb is available and if the run is active
        wandb_run = wandb.run
        need_to_upload = (
            wandb_run is not None
            and not wandb_run.disabled
            and not wandb_run.offline
            and not wandb_run._is_finished
        )
        if need_to_upload:
            wandb.save(
                filepath, base_path=self.dirpath, policy="now" if self.save_last else "live"
            )
            logger.info(f"Checkpoint saved to {filepath} and uploaded to WandB.")
        else:
            logger.info(f"Checkpoint saved to {filepath} but WandB is not active.")
