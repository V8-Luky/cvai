import lightning as L

class CycleGAN(L.LightningModule):
    def __init__(self, model,  *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        # Implement the training step logic here
        pass

    def validation_step(self, *args, **kwargs):
        return super().validation_step(*args, **kwargs)
    
    def test_step(self, *args, **kwargs):
        return super().test_step(*args, **kwargs)
    
    def predict_step(self, *args, **kwargs):
        return super().predict_step(*args, **kwargs)
    
    def configure_optimizers(self):
        return super().configure_optimizers()
    
    def configure_model(self):
        return super().configure_model()