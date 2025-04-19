import lightning as L
import torch

from  dataclasses import dataclass, field

class CycleGAN(L.LightningModule):
    def __init__(self, model_config, train_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(model_config)
        self.save_hyperparameters(train_config)
        self.model = None

    def forward(self, a, b):
        return self.model(a, b)

    def training_step(self, batch, batch_idx):
        return self._make_step(batch['a'], batch['b'])
        
    def validation_step(self, batch, batch_idx):
        return self._make_step(batch['a'], batch['b'])



    def predict_step(self, batch, batch_idx):
        return self(batch['a'], batch['b'])

    def configure_optimizers(self):
        from self.hparams.train_config

    def configure_model(self):
        if self.model:
            return
        
        self.model = from self.hparams.model_config

        return super().configure_model()
    
    def _make_step(self, a, b):
        fake_a, fake_b = self(a, b)

        return fake_a, fake_b

