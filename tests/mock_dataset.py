import lightning as L
import torch


from torch.utils.data import Dataset, DataLoader


class MockDataset(Dataset):
    def __init__(self, shape):
        super().__init__()
        self.data = torch.randn(shape)

    def __getitem__(self, index):
        data = self.data[index]
        return {
            "a": data,
            "b": data
        }
    
    def __len__(self):
        return len(self.data)


class MockDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_shape: tuple,
        valid_shape: tuple,
        test_shape: tuple,
        predict_shape: tuple,
        batch_size: int,
    ):
        super().__init__()
        self.train_shape = train_shape
        self.valid_shape = valid_shape
        self.test_shape = test_shape
        self.predict_shape = predict_shape
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = MockDataset(self.train_shape)
        self.valid_dataset = MockDataset(self.valid_shape)
        self.test_dataset = MockDataset(self.test_shape)
        self.predict_dataset = MockDataset(self.predict_shape)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size)
