import torch

from cycleGAN import UnetGenerator, ResidualGenerator


def test_unet_generator():
    shape = (2, 3, 128, 128)
    data = torch.randn(shape)
    model = UnetGenerator()
    output = model(data)
    assert output.shape == shape


def test_residual_generator():
    shape = (2, 3, 128, 128)
    data = torch.randn(shape)
    model = ResidualGenerator()
    output = model(data)
    assert output.shape == shape