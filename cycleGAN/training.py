"""
This module provides training and hyperparameter tuning functionality for CycleGAN models.

It contains two main classes:
- Training: For running a single training session with a CycleGAN model
- Sweep: For conducting hyperparameter sweeps using Weights & Biases
"""

import lightning as L
import wandb

from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)

from cycleGAN import TrainableCycleGAN, TrainConfig, CycleGANConfig
from lightning.pytorch.loggers.wandb import WandbLogger

SEED = 42


class Training:
    """
    Handles the training process for a CycleGAN model.

    This class wraps the Lightning Trainer with appropriate configurations
    and callbacks for training CycleGAN models.

    Attributes:
        name (str): Name of the training run
        model (TrainableCycleGAN): The CycleGAN model to train
        datamodule (L.LightningDataModule): Data module providing the training data
        callbacks (list[L.Callback]): List of Lightning callbacks for the training process
    """

    def __init__(
        self,
        name: str,
        model: TrainableCycleGAN,
        datamodule: L.LightningDataModule,
        callbacks: list[L.Callback] = None,
    ):
        """
        Initialize a Training instance.

        Args:
            name (str): Name of the training run
            model (TrainableCycleGAN): The CycleGAN model to train
            datamodule (L.LightningDataModule): Data module providing the training data
            callbacks (list[L.Callback], optional): List of Lightning callbacks.
                If None, default callbacks will be used.
        """
        self.name = name
        self.model = model
        self.datamodule = datamodule
        self.callbacks = (
            callbacks if callbacks is not None else self.get_default_callbacks()
        )

    def __call__(self):
        """
        Execute the training process.

        Sets up a WandbLogger, configures a Lightning Trainer with the specified
        parameters, and runs the training process.
        """
        logger = WandbLogger(name=self.name, log_model=True)

        trainer = L.Trainer(
            accelerator="auto",
            devices="auto",
            max_epochs=self.model.hparams.train_config.max_epochs,
            logger=logger,
            log_every_n_steps=100,
            callbacks=self.callbacks,
            precision="16-mixed",
        )

        trainer.fit(self.model, datamodule=self.datamodule)

        wandb.finish()

    def get_default_callbacks(self):
        """
        Create default callbacks for the training process.

        Returns:
            list[L.Callback]: A list of default callbacks including learning rate monitoring,
                model checkpointing, and early stopping based on validation and training loss.
        """
        return [
            LearningRateMonitor(
                logging_interval="step", log_momentum=True, log_weight_decay=True
            ),
            ModelCheckpoint(
                monitor="valid_loss",
                filename="{epoch:02d}-{valid_loss:.2f}",
                save_top_k=3,
                mode="min",
            ),
            ModelCheckpoint(
                monitor="epoch", filename="latest-{epoch:02d}", save_top_k=1, mode="max"
            ),
            EarlyStopping(monitor="valid_loss", patience=5, mode="min"),
            EarlyStopping(monitor="train_loss", patience=5, mode="min"),
        ]


class Sweep:
    """
    Handles hyperparameter sweeps for CycleGAN models using Weights & Biases.

    This class configures and executes hyperparameter searches to find optimal
    training configurations for CycleGAN models.

    Attributes:
        run_name_prefix (str): Prefix for naming individual sweep runs
        project (str): W&B project name
        entity (str): W&B entity (username or organization)
        model_config (CycleGANConfig): Base configuration for the CycleGAN model
        sweep_config (dict): W&B sweep configuration defining the hyperparameter search space
        datamodule (L.LightningDataModule): Data module providing the training data
        count (int): Maximum number of runs to execute
    """

    def __init__(
        self,
        run_name_prefix: str,
        project: str,
        entity: str,
        model_config: CycleGANConfig,
        sweep_config: dict,
        datamodule: L.LightningDataModule,
        count: int = 10,
    ):
        self.run_name_prefix = run_name_prefix
        self.project = project
        self.entity = entity
        self.model_config = model_config
        self.sweep_config = sweep_config
        self.datamodule = datamodule
        self.count = count
        self.run_id = 0

    def run_training(self):
        self.run_id += 1

        run_name = f"{self.run_name_prefix}-{self.run_id}"

        wandb.init(name=run_name)
        config = wandb.config

        L.seed_everything(SEED)

        train_config = self.get_train_config(config)

        trainable_model = TrainableCycleGAN(self.model_config, train_config)

        training = Training(run_name, trainable_model, self.datamodule)
        training()

    def get_train_config(self, config: dict) -> TrainConfig:
        return TrainConfig(
            max_epochs=config["max_epochs"],
            start_epoch=config["start_epoch"],
            decay_epoch=config["decay_epoch"],
            learning_rate=config["learning_rate"],
            lambda_a=config["lambda_a"],
            lambda_b=config["lambda_b"],
            lambda_identity=config["lambda_identity"],
        )

    def __call__(self, *args, **kwds):
        sweep_id = wandb.sweep(
            sweep=self.sweep_config, project=self.project, entity=self.entity
        )
        wandb.agent(sweep_id=sweep_id, function=self.run_training, count=self.count)
        wandb.api.stop_sweep(sweep_id)
