"""
This module contains a pytorch-based CycleGAN implementation.

Code is based on the original implementation from Jun-Yan Zhu:
  - https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master

An easy walkthrough on a CycleGAN implementation is presented on YouTube by Aladdin Persson:
  - https://www.youtube.com/watch?v=4LktBHGCNfw&t=3018s
"""

from .generator import ResidualGenerator, UnetGenerator
from .discriminator import PatchGanDiscriminator, PixelGanDiscriminator
from .model import CycleGANConfig, CycleGAN
from .trainable import TrainableCycleGAN, TrainConfig
from .training import Training, Sweep