import lightning as L
import torch.nn as nn
import torch

from dataclasses import dataclass
from .model import CycleGAN, CycleGANConfig


@dataclass
class TrainConfig:
    max_epochs: int = 200
    start_epoch: int = 0
    decay_epoch: int = 100
    learning_rate: float = 1e-4
    lambda_a: float = 10.0
    lambda_b: float = 10.0
    lambda_identity: float = 0.5


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (
            n_epochs - decay_start_epoch
        ) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def __call__(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (
            self.n_epochs - self.decay_start_epoch
        )


class TrainableCycleGAN(L.LightningModule):
    def __init__(
        self, model_config: CycleGANConfig, train_config: TrainConfig, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        config = dict()
        config["model_config"] = model_config
        config["train_config"] = train_config
        config["model_config_detailed"] = vars(model_config)
        config["train_config_detailed"] = vars(train_config)
        self.save_hyperparameters(config)
        

        self.model = None

        self.loss_gan = nn.MSELoss()
        self.loss_cycle = nn.L1Loss()
        self.loss_identity = nn.L1Loss()

    def forward(self, a, b):
        return self.model(a, b)

    def training_step(self, batch, batch_idx):
        fake_a, fake_b, loss = self._make_step(batch["a"], batch["b"], "train")
        return loss

    def validation_step(self, batch, batch_idx):
        _, _, loss = self._make_step(batch["a"], batch["b"], "valid")
        return loss

    def test_step(self, batch, batch_idx):
        return self._make_step(batch["a"], batch["b"], "test")

    def predict_step(self, batch, batch_idx):
        return self(batch["a"], batch["b"])

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            self.model.get_parameters(),
            lr=self.hparams.train_config.learning_rate,
            betas=(0.5, 0.999),
        )

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=LambdaLR(
                self.hparams.train_config.max_epochs,
                self.hparams.train_config.start_epoch,
                self.hparams.train_config.decay_epoch,
            ),
        )

        return [optimizer], [lr_scheduler]

    def configure_model(self):
        if self.model:
            return

        self.model = CycleGAN(self.hparams.model_config)

    def _make_step(self, a, b, stage: str):
        fake_a, fake_b = self(a, b)

        loss_gen_a, loss_gen_b, loss_cycle_a, loss_cycle_b, loss_idt_a, loss_idt_b = (
            self._generator_loss(a, b, fake_a, fake_b)
        )
        loss_disc_a, loss_disc_b = self._discriminator_loss(a, b, fake_a, fake_b)

        generator_loss = (
            loss_gen_a
            + loss_gen_b
            + loss_cycle_a
            + loss_cycle_b
            + loss_idt_a
            + loss_idt_b
        )
        discriminator_loss = (loss_disc_a + loss_disc_b) * 0.5
        loss = generator_loss + discriminator_loss

        self.log(f"{stage}_loss_gen_a", loss_gen_a)
        self.log(f"{stage}_loss_gen_b", loss_gen_b)
        self.log(f"{stage}_loss_cycle_a", loss_cycle_a)
        self.log(f"{stage}_loss_cycle_b", loss_cycle_b)
        self.log(f"{stage}_loss_idt_a", loss_idt_a)
        self.log(f"{stage}_loss_idt_b", loss_idt_b)
        self.log(f"{stage}_loss_disc_a", loss_disc_a)
        self.log(f"{stage}_loss_disc_b", loss_disc_b)
        self.log(f"{stage}_loss_generator", generator_loss)
        self.log(f"{stage}_loss_discriminator", generator_loss)
        self.log(f"{stage}_loss", generator_loss)

        return fake_a, fake_b, loss

    def _generator_loss(
        self,
        real_a: torch.Tensor,
        real_b: torch.Tensor,
        fake_a: torch.Tensor,
        fake_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the total generator loss including adversarial, cycle consistency,
        and optional identity loss.

        :return: Total generator loss.
        """
        lambda_idt = self.hparams.train_config.lambda_identity
        lambda_a = self.hparams.train_config.lambda_a
        lambda_b = self.hparams.train_config.lambda_b

        disc_a = self.model.discriminator_a(fake_a)
        disc_b = self.model.discriminator_b(fake_b)
        loss_gen_a = self.loss_gan(disc_a, torch.ones_like(disc_a))
        loss_gen_b = self.loss_gan(disc_b, torch.ones_like(disc_b))

        rec_a = self.model.generator_b_to_a(fake_b)
        rec_b = self.model.generator_a_to_b(fake_a)
        loss_cycle_a = self.loss_cycle(rec_a, real_a) * lambda_a
        loss_cycle_b = self.loss_cycle(rec_b, real_b) * lambda_b

        if lambda_idt > 0.0:
            loss_idt_a = (
                self.loss_identity(self.model.generator_b_to_a(real_a), real_a)
                * lambda_idt
                * lambda_a
            )
            loss_idt_b = (
                self.loss_identity(self.model.generator_a_to_b(real_b), real_b)
                * lambda_idt
                * lambda_b
            )
        else:
            loss_idt_a = 0
            loss_idt_b = 0

        return (
            loss_gen_a,
            loss_gen_b,
            loss_cycle_a,
            loss_cycle_b,
            loss_idt_a,
            loss_idt_b,
        )

    def _discriminator_loss(
        self,
        real_a: torch.Tensor,
        real_b: torch.Tensor,
        fake_a: torch.Tensor,
        fake_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the total discriminator loss using real and fake samples
        for both domain A and B.

        :return: Total discriminator loss.
        """
        disc_real_a = self.model.discriminator_a(real_a)
        disc_fake_a = self.model.discriminator_a(fake_a.detach())
        loss_disc_real_a = self.loss_gan(disc_real_a, torch.ones_like(disc_real_a))
        loss_disc_fake_a = self.loss_gan(disc_fake_a, torch.zeros_like(disc_fake_a))
        loss_disc_a = loss_disc_real_a + loss_disc_fake_a

        disc_real_b = self.model.discriminator_b(real_b)
        disc_fake_b = self.model.discriminator_b(fake_b.detach())
        loss_disc_real_b = self.loss_gan(disc_real_b, torch.ones_like(disc_real_b))
        loss_disc_fake_b = self.loss_gan(disc_fake_b, torch.zeros_like(disc_fake_b))
        loss_disc_b = loss_disc_real_b + loss_disc_fake_b

        return loss_disc_a, loss_disc_b
