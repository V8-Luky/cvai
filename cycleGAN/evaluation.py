import torch
import lightning as L
import wandb

from cycleGAN import (
    TrainableCycleGAN,
    TrainConfig,
    CycleGANConfig,
    PatchGanDiscriminator,
    PixelGanDiscriminator,
    ResidualGenerator,
    UnetGenerator,
)
from lightning.pytorch.loggers.wandb import WandbLogger


class Evaluation:
    def __init__(
        self,
        name: str,
        project: str,
        entity: str,
        artifact: str,
        datamodule: L.LightningDataModule,
    ):
        self.name = name
        self.project = project
        self.entity = entity
        self.artifact = artifact
        self.datamodule = datamodule

    def __call__(self):
        run = wandb.init(name=self.name, project=self.project, entity=self.entity)
        artifact = run.use_artifact(self.artifact, type="model")
        artifact_dir = artifact.download()

        checkpoint = torch.load(artifact_dir + "/model.ckpt")

        model = TrainableCycleGAN.load_from_checkpoint(checkpoint)
        logger = WandbLogger(name=self.name, log_model=True)

        trainer = L.Trainer(
            accelerator="auto",
            devices="auto",
            logger=logger,
            log_every_n_steps=100,
        )

        trainer.test(model, self.datamodule)

        wandb.finish()

    def get_model_config(self, config: dict) -> CycleGANConfig:
        config = config["model_config_detailed"]
        return CycleGANConfig(
            gen_type=ResidualGenerator if "cycleGAN.generator.ResidualGenerator" == config["gen_type"]else UnetGenerator,
            gen_channels=config["gen_channels"],
            disc_type=PatchGanDiscriminator if "cycleGAN.discriminator.PatchGanDiscriminator" == config["disc_type"] else PixelGanDiscriminator,
            disc_channels=config["disc_channels"],
        )
