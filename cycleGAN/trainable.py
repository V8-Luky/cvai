"""
This module provides a Lightning-compatible implementation of CycleGAN for training.

It contains:
- TrainConfig: A dataclass for configuring training parameters
- LambdaLR: A learning rate scheduler for gradually reducing learning rate
- TrainableCycleGAN: A Lightning module that implements the CycleGAN training logic
"""

import lightning as L
import torch.nn as nn
import torch
from torchvision.utils import save_image

from dataclasses import dataclass
from .model import CycleGAN, CycleGANConfig


@dataclass
class TrainConfig:
    """
    Configuration for training a CycleGAN.

    Attributes:
        max_epochs (int): Maximum number of epochs for training
        start_epoch (int): Starting epoch for training
        decay_epoch (int): Epoch at which to start learning rate decay
        learning_rate (float): Initial learning rate for the optimizer
        lambda_a (float): Weight for cycle consistency loss in domain A
        lambda_b (float): Weight for cycle consistency loss in domain B
        lambda_identity (float): Weight for identity loss
        gradient_acc_steps (int): Amount of steps to accumulate gradients for
        save_train (bool): Save the first image paris of the first batch per epoch for the train set
        save_valid (bool): Save the first image paris of the first batch per epoch for the valid set
    """
    max_epochs: int = 200
    start_epoch: int = 0
    decay_epoch: int = 100
    learning_rate: float = 1e-4
    lambda_a: float = 10.0
    lambda_b: float = 10.0
    lambda_identity: float = 0.5
    gradient_acc_steps: int = 10
    save_train: bool = True,
    save_valid: bool = False,


class LambdaLR:
    """
    Learning rate scheduler that decays the learning rate linearly after a specified epoch.
    """

    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (
            n_epochs - decay_start_epoch
        ) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def __call__(self, epoch):
        """
        Calculate the learning rate based on the current epoch.

        Args:
            epoch: Current epoch

        Returns: Learning rate for the current epoch
        """
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (
            self.n_epochs - self.decay_start_epoch
        )


class TrainableCycleGAN(L.LightningModule):
    """
    A Lightning module for training CycleGAN models.

    This module encapsulates the training logic, including the forward pass,
    loss computation, and optimizer configuration.

    Attributes:
        model_config (CycleGANConfig): Configuration for the CycleGAN model
        train_config (TrainConfig): Configuration for training parameters   
    """

    def __init__(
        self, 
        model_config: CycleGANConfig, 
        train_config: TrainConfig, 
        model_config_detailed = None, 
        train_config_detailed = None, 
        *args, 
        **kwargs
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

        self.automatic_optimization = False

    def forward(self, a, b):
        return self.model(a, b)

    def training_step(self, batch, batch_idx):
        fake_a, fake_b, losses = self._make_step(batch["a"], batch["b"], "train")

        optimizers = self.optimizers()
        for loss, optimizer in zip(losses, optimizers):
            loss /= self.hparams.train_config.gradient_acc_steps
            self.manual_backward(loss)
            if (batch_idx + 1) % self.hparams.train_config.gradient_acc_steps == 0 or self.trainer.is_last_batch:
                #self.clip_gradients(optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
                optimizer.step()
                optimizer.zero_grad()

        if self.hparams.train_config.save_train and batch_idx == 0:
            self.save_images(batch["a"], batch["b"], fake_a, fake_b)

    def on_train_epoch_end(self):
        for scheduler in self.lr_schedulers():
            scheduler.step()

    def validation_step(self, batch, batch_idx):
        fake_a, fake_b, _ = self._make_step(batch["a"], batch["b"], "valid")

        if self.hparams.train_config.save_valid and batch_idx == 0:
            self.save_images(batch["a"], batch["b"], fake_a, fake_b)

    def test_step(self, batch, batch_idx):
        fake_a, fake_b, _ = self._make_step(batch["a"], batch["b"], "test")
        self.save_images(batch["a"], batch["b"], fake_a, fake_b)
        return fake_a, fake_b

    def predict_step(self, batch, batch_idx):
        return self(batch["a"], batch["b"])

    def configure_optimizers(self):
        param_groups = (self.model.get_generator_params(), self.model.get_discriminator_a_params(), self.model.get_discriminator_b_params())
        optimizers = [torch.optim.Adam(params, lr=self.hparams.train_config.learning_rate, betas=(0.5, 0.999)) for params in param_groups]
        lr_schedulers = [torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=self._new_lr_lambda()) for optim in optimizers]

        return optimizers, lr_schedulers

    def configure_model(self):
        if self.model:
            return

        self.model = CycleGAN(self.hparams.model_config)

    def save_images(
        self,
        real_a: torch.Tensor,
        real_b: torch.Tensor,
        fake_a: torch.Tensor,
        fake_b: torch.Tensor,
    ):
        self.save_image(real_a, "real_a")
        self.save_image(real_b, "real_b")
        self.save_image(fake_a, "fake_a")
        self.save_image(fake_b, "fake_b")

    def save_image(self, tensor: torch.Tensor, name: str):
        save_image(tensor * 0.5 + 0.5, f"{self.trainer.ckpt_path}/{name}_ep{self.trainer.current_epoch}.jpg")


    def _make_step(self, a, b, stage: str):
        fake_a, fake_b = self(a, b)

        loss_gen_a, loss_gen_b, loss_cycle_a, loss_cycle_b, loss_idt_a, loss_idt_b = (
            self._generator_loss(a, b, fake_a, fake_b)
        )
        generator_loss = loss_gen_a + loss_gen_b + loss_cycle_a + loss_cycle_b + loss_idt_a + loss_idt_b
        loss_disc_a, loss_disc_b = self._discriminator_loss(a, b, fake_a, fake_b)
        discriminator_loss = loss_disc_a + loss_disc_b

        loss = discriminator_loss + generator_loss

        self.log(f"{stage}_loss_gen_a", loss_gen_a)
        self.log(f"{stage}_loss_gen_b", loss_gen_b)
        self.log(f"{stage}_loss_cycle_a", loss_cycle_a)
        self.log(f"{stage}_loss_cycle_b", loss_cycle_b)
        self.log(f"{stage}_loss_idt_a", loss_idt_a)
        self.log(f"{stage}_loss_idt_b", loss_idt_b)
        self.log(f"{stage}_loss_disc_a", loss_disc_a)
        self.log(f"{stage}_loss_disc_b", loss_disc_b)
        self.log(f"{stage}_loss_generator", generator_loss)
        self.log(f"{stage}_loss_discriminator", discriminator_loss)
        self.log(f"{stage}_loss", loss)

        return fake_a, fake_b, (generator_loss, loss_disc_a, loss_disc_b)

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
        loss_disc_a = (loss_disc_real_a + loss_disc_fake_a) * 0.5

        disc_real_b = self.model.discriminator_b(real_b)
        disc_fake_b = self.model.discriminator_b(fake_b.detach())
        loss_disc_real_b = self.loss_gan(disc_real_b, torch.ones_like(disc_real_b))
        loss_disc_fake_b = self.loss_gan(disc_fake_b, torch.zeros_like(disc_fake_b))
        loss_disc_b = (loss_disc_real_b + loss_disc_fake_b) * 0.5

        return loss_disc_a, loss_disc_b

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

    def _new_lr_lambda(self) -> LambdaLR:
        """
        Returns a new instance of the learning rate scheduler.
        """
        return LambdaLR(
            self.hparams.train_config.max_epochs,
            self.hparams.train_config.start_epoch,
            self.hparams.train_config.decay_epoch,
        )
