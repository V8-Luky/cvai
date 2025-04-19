import torch
import lightning as L

from .mock_dataset import MockDataModule

from cycleGAN import TrainableCycleGAN
from cycleGAN import CycleGANConfig, TrainConfig


def training_sanity_check():
    model_config = CycleGANConfig()
    train_config = TrainConfig()

    trainable_model = TrainableCycleGAN(model_config, train_config)

    data_module = MockDataModule(
        (6, 3, 128, 128),
        (2, 3, 128, 128),
        (2, 3, 128, 128),
        (1, 3, 128, 128),
        batch_size=32,
    )

    trainer = L.Trainer(
        max_epochs=1,
        accelerator="auto",
        logger=False,
        enable_checkpointing=False,
        fast_dev_run=1,  # Runs a sanity check over all stages (train, val, test, predict) with a single batch
    )
    trainer.fit(trainable_model, data_module)
