import lightning as pl
from torch.utils.data import DataLoader, random_split
from data.dataset import UnpairedImageDataset


class UnpairedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        domain_a_dir: str,
        domain_b_dir: str,
        transform=None,
        batch_size: int = 16,
        num_workers: int = 4,
        split_ratio: float = 0.8,
    ):
        super().__init__()
        self.domain_a_dir = domain_a_dir
        self.domain_b_dir = domain_b_dir
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratio = split_ratio

    def setup(self, stage=None):
        full_dataset = UnpairedImageDataset(
            domain_a_dir=self.domain_a_dir,
            domain_b_dir=self.domain_b_dir,
            transform=self.transform
        )
        train_size = int(self.split_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
