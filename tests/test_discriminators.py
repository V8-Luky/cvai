import torch

from cycleGAN import PixelGanDiscriminator, PatchGanDiscriminator


def test_pixelgan_discriminator():
    shape = (2, 3, 128, 128)
    data = torch.rand(shape)
    model = PixelGanDiscriminator()
    output = model(data)
    assert output.shape == (2, 1, 128, 128)


def test_patchgan_discriminator():
    shape = (2, 3, 128, 128)
    data = torch.rand(shape)
    model = PatchGanDiscriminator()
    output = model(data)
    assert output.shape == (2, 1, 14, 14)
    