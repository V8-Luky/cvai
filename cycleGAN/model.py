from dataclasses import dataclass, field

import torch
import torch.nn as nn
from typing import Type
from collections.abc import Iterator

from .generator import UnetGenerator, ResidualGenerator
from .discriminator import PixelGanDiscriminator, PatchGanDiscriminator


@dataclass
class CycleGANConfig:
    gen_type: Type[ResidualGenerator] | Type[UnetGenerator] = ResidualGenerator
    gen_channels: int = 64
    gen_kwargs: dict = field(default_factory=dict)

    disc_type: Type[PatchGanDiscriminator] | Type[PixelGanDiscriminator] = PatchGanDiscriminator
    disc_channels: int = 64
    disc_kwargs: dict = field(default_factory=dict)

    lambda_a: float = 10.0
    lambda_b: float = 10.0
    lambda_identity: float = 0.5


class CycleGAN(nn.Module):
    def __init__(self, config: CycleGANConfig):
        super().__init__()
        self.config = config
        self.generator_a_to_b = config.gen_type(in_channels=3, channels=config.gen_channels, **config.gen_kwargs)
        self.generator_b_to_a = config.gen_type(in_channels=3, channels=config.gen_channels, **config.gen_kwargs)
        self.discriminator_a = config.disc_type(in_channels=3, channels=config.disc_channels, **config.disc_kwargs)
        self.discriminator_b = config.disc_type(in_channels=3, channels=config.disc_channels, **config.disc_kwargs)

        self.loss_gan = nn.MSELoss()
        self.loss_cycle = nn.L1Loss()
        self.loss_identity = nn.L1Loss()

        self.real_a: torch.Tensor | None = None
        self.fake_a: torch.Tensor | None = None
        self.real_b: torch.Tensor | None = None
        self.fake_b: torch.Tensor | None = None

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        self.real_a = a
        self.real_b = b
        self.fake_a = self.generator_b_to_a(b)
        self.fake_b = self.generator_a_to_b(a)

    def generator_loss(self) -> torch.Tensor:
        lambda_idt = self.config.lambda_identity
        lambda_a = self.config.lambda_a
        lambda_b = self.config.lambda_b

        disc_a = self.discriminator_a(self.fake_a)
        disc_b = self.discriminator_b(self.fake_b)
        loss_gen_a = self.loss_gan(disc_a, torch.ones_like(disc_a))
        loss_gen_b = self.loss_gan(disc_b, torch.ones_like(disc_b))

        rec_a = self.generator_b_to_a(self.fake_b)
        rec_b = self.generator_a_to_b(self.fake_a)
        loss_cycle_a = self.loss_cycle(rec_a, self.real_a) * lambda_a
        loss_cycle_b = self.loss_cycle(rec_b, self.real_b) * lambda_b

        if lambda_idt > 0.0:
            loss_idt_a = self.loss_identity(self.generator_b_to_a(self.real_a), self.real_a) * lambda_idt * lambda_a
            loss_idt_b = self.loss_identity(self.generator_a_to_b(self.real_b), self.real_b) * lambda_idt * lambda_b
        else:
            loss_idt_a = 0
            loss_idt_b = 0

        return loss_gen_a + loss_gen_b + loss_cycle_a + loss_cycle_b + loss_idt_a + loss_idt_b

    def discriminator_loss(self) -> torch.Tensor:
        disc_real_a = self.discriminator_a(self.real_a)
        disc_fake_a = self.discriminator_a(self.fake_a.detach())
        loss_disc_real_a = self.loss_gan(disc_real_a, torch.ones_like(disc_real_a))
        loss_disc_fake_a = self.loss_gan(disc_fake_a, torch.ones_like(disc_fake_a))
        loss_disc_a = loss_disc_real_a + loss_disc_fake_a

        disc_real_b = self.discriminator_b(self.real_b)
        disc_fake_b = self.discriminator_b(self.fake_b.detach())
        loss_disc_real_b = self.loss_gan(disc_real_b, torch.ones_like(disc_real_b))
        loss_disc_fake_b = self.loss_gan(disc_fake_b, torch.ones_like(disc_fake_b))
        loss_disc_b = loss_disc_real_b + loss_disc_fake_b

        return (loss_disc_a + loss_disc_b) * 0.5

    def get_discriminator_params(self) -> Iterator[nn.Parameter]:
        yield from self.discriminator_a.parameters()
        yield from self.discriminator_b.parameters()

    def get_generator_params(self) -> Iterator[nn.Parameter]:
        yield from self.generator_a.parameters()
        yield from self.generator_b.parameters()