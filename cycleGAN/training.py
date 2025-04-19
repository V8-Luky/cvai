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
    def __init__(
        self,
        name: str,
        model: TrainableCycleGAN,
        datamodule: L.LightningDataModule,
        callbacks: list[L.Callback] = None,
    ):
        self.name = name
        self.model = model
        self.datamodule = datamodule
        self.callbacks = (
            callbacks if callbacks is not None else self.get_default_callbacks()
        )

    def __call__(self):
        logger = WandbLogger(name=self.name, log_model=True)

        trainer = L.Trainer(
            accelerator="auto",
            devices="auto",
            max_epochs=self.model.hparams.train_config.max_epochs,
            logger=logger,
            log_every_n_steps=100,
            callbacks=self.callbacks,
        )

        trainer.fit(self.model, datamodule=self.datamodule)

        wandb.finish()

    def get_default_callbacks(self):
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
        wandb.teardown()
