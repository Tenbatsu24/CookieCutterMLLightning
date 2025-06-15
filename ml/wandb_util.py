import torch
import wandb
import lightning.pytorch as pl

from loguru import logger

from lightning.pytorch.callbacks import ModelCheckpoint

from ml.config import WANDB_ENTITY, WANDB_PROJECT


class MyModelCheckpoint(ModelCheckpoint):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._enable_version_counter = False

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        weights_only = not self.save_last
        trainer.save_checkpoint(
            filepath, weights_only=weights_only
        )  # save weights only if not saving last

        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

        # if weights_only, remove model. from the keys of the state dict
        if weights_only:
            state_dict = torch.load(filepath, map_location=torch.device("cpu"), weights_only=True)
            # rename the keys to remove 'model.'
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            # save the state dict back to the file
            torch.save(state_dict, filepath)
            logger.info(f"Removed 'model.' prefix from state dict keys in {filepath}.")

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

    def get_checkpoint_from_wandb(self, run_id: str):
        """
        Retrieve a checkpoint file from WandB using the run ID and filename.
        """
        if wandb.run is None or wandb.run.disabled:
            raise RuntimeError("WandB is not initialized or disabled.")

        run_path = f"{WANDB_PROJECT}/{run_id}"
        if WANDB_ENTITY is not None:
            run_path = f"{WANDB_ENTITY}/{run_path}"

        try:
            file = wandb.run.restore(
                name=f"{self.filename}{self.FILE_EXTENSION}",
                run_path=run_path,
                replace=True,
                root=self.dirpath,
            )
        except ValueError as e:
            logger.error(f"Error retrieving checkpoint from WandB: {e}")
            file = None

        if file is None:
            raise FileNotFoundError(
                f"Checkpoint {self.filename} not found in WandB for run {run_id}."
            )
        else:
            # get the path to the file
            path_to_file = file.name
            # close the file
            try:
                file.close()
            except Exception as e:
                logger.warning(f"Failed to close the file {file.name}: {e}")
            # return the path to the file
            return path_to_file
