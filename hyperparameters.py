from cycleGAN import TrainConfig, CycleGANConfig, ResidualGenerator, UnetGenerator, PatchGanDiscriminator, PixelGanDiscriminator


TRAIN_CONFIG = "train_config"
MODEL_CONFIG = "model_config"


def get_train_config_as_sweep_params(train_config: TrainConfig):
    """
    Returns a dictionary of the training configuration parameters for sweep.
    """
    return {
        "max_epochs": {"value": train_config.max_epochs},
        "start_epoch": {"value": train_config.start_epoch},
        "decay_epoch": {"value": train_config.decay_epoch},
        "learning_rate": {"value": train_config.learning_rate},
        "lambda_a": {"value": train_config.lambda_a},
        "lambda_b": {"value":   train_config.lambda_b},
        "lambda_identity": {"value": train_config.lambda_identity},
        "gradient_acc_steps": {"value": train_config.gradient_acc_steps},
    }


"""
Some Input from ChatGPT:

Here’s a structured way to think about it:
Generators:
    ResidualGenerator (ResNet-based) is typically better for style transfer tasks like this.
    UnetGenerator is usually better for tasks where the output should preserve a lot of the input structure (e.g., segmentation, not big style shifts).

Discriminators:
    PatchGAN is the standard for CycleGAN — it works great because it forces texture consistency.
    PixelGAN is lighter but often weaker for detailed textures (like Ghibli textures).

# | Gen Type | Disc Type | Key Changes
1 | ResidualGenerator | PatchGAN | (baseline A)
2 | ResidualGenerator | PatchGAN | learning_rate = 2e-6
3 | UnetGenerator | PatchGAN | (baseline C)
4 | ResidualGenerator | PatchGAN | gen_channels = disc_channels = 128
5 | ResidualGenerator | PatchGAN | lambda_identity = 5.0

The baselines:
"""
A = {
    TRAIN_CONFIG: TrainConfig(max_epochs=200, decay_epoch=100, learning_rate=1e-6, lambda_a=10.0, lambda_b=15.0, lambda_identity=0.5, gradient_acc_steps=10),
    MODEL_CONFIG: CycleGANConfig(gen_type=ResidualGenerator, gen_channels=64, disc_type=PatchGanDiscriminator, disc_channels=64),
}

B = {
    TRAIN_CONFIG: TrainConfig(max_epochs=200, decay_epoch=100, learning_rate=1e-6, lambda_a=10.0, lambda_b=15.0, lambda_identity=0.5, gradient_acc_steps=10),
    MODEL_CONFIG: CycleGANConfig(gen_type=ResidualGenerator, gen_channels=64, disc_type=PixelGanDiscriminator, disc_channels=64),
}

C = {
    TRAIN_CONFIG: TrainConfig(max_epochs=200, decay_epoch=100, learning_rate=1e-6, lambda_a=10.0, lambda_b=15.0, lambda_identity=0.5, gradient_acc_steps=10),
    MODEL_CONFIG: CycleGANConfig(gen_type=UnetGenerator, gen_channels=64, disc_type=PatchGanDiscriminator, disc_channels=64),
}

D = {
    TRAIN_CONFIG: TrainConfig(max_epochs=200, decay_epoch=100, learning_rate=1e-6, lambda_a=10.0, lambda_b=15.0, lambda_identity=0.5, gradient_acc_steps=10),
    MODEL_CONFIG: CycleGANConfig(gen_type=UnetGenerator, gen_channels=64, disc_type=PixelGanDiscriminator, disc_channels=64),
}
"""
When only selecting three, ChatGPT suggests:

Run | Gen Type | Disc Type | Key Changes
1 | ResidualGenerator | PatchGAN | (baseline A)
2 | ResidualGenerator | PatchGAN | gen_channels = disc_channels = 128
3 | UnetGenerator | PatchGAN | (baseline C)

"""



CONFIGURATIONS = {
    "A": {
        TRAIN_CONFIG: TrainConfig(max_epochs=200, decay_epoch=100, learning_rate=1e-5, lambda_a=10.0, lambda_b=15.0, lambda_identity=0.5, gradient_acc_steps=10),
        MODEL_CONFIG: CycleGANConfig(gen_type=ResidualGenerator, gen_channels=64, disc_type=PatchGanDiscriminator, disc_channels=64),
    },
    "B": {
        TRAIN_CONFIG: TrainConfig(max_epochs=200, decay_epoch=100, learning_rate=1e-5, lambda_a=10.0, lambda_b=15.0, lambda_identity=0.5, gradient_acc_steps=10),
        MODEL_CONFIG: CycleGANConfig(gen_type=ResidualGenerator, gen_channels=128, disc_type=PatchGanDiscriminator, disc_channels=128),
    },
    "C": {
        TRAIN_CONFIG: TrainConfig(max_epochs=200, decay_epoch=100, learning_rate=1e-5, lambda_a=10.0, lambda_b=15.0, lambda_identity=0.5, gradient_acc_steps=10),
        MODEL_CONFIG: CycleGANConfig(gen_type=UnetGenerator, gen_channels=64, disc_type=PatchGanDiscriminator, disc_channels=64),
    },
    "EXTRA_A": {
        # Original Learning Rate apparently
        TRAIN_CONFIG: TrainConfig(max_epochs=200, decay_epoch=100, learning_rate=2e-4, lambda_a=10.0, lambda_b=15.0, lambda_identity=5.0, gradient_acc_steps=10),
        MODEL_CONFIG: CycleGANConfig(gen_type=ResidualGenerator, gen_channels=64, disc_type=PatchGanDiscriminator, disc_channels=64),
    },
    "EXTRA_B": {
        TRAIN_CONFIG: TrainConfig(max_epochs=200, decay_epoch=20, learning_rate=1e-3, lambda_a=10.0, lambda_b=15.0, lambda_identity=0.5, gradient_acc_steps=10),
        MODEL_CONFIG: CycleGANConfig(gen_type=ResidualGenerator, gen_channels=64, disc_type=PixelGanDiscriminator, disc_channels=64),
    }
}


