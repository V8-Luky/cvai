import lightning as pl
from torch.utils.data import DataLoader, random_split
from data.dataset import UnpairedImageDataset
from kagglehub import dataset_download
import torch
import os
from torch.utils.data._utils.collate import default_collate


class UnpairedDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for handling unpaired image datasets.

    Args:
        train_a_subdir (str): Subdirectory name for domain A training images.
        train_b_subdir (str): Subdirectory name for domain B training images.
        test_domain_dir (str, optional): Directory for test domain images.
        dataset_id (str, optional): KaggleHub dataset ID for downloading the dataset.
        transform (torchvision.transforms.Compose, optional): Transformations to apply to images.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        split_ratio (float): Ratio to split training/validation datasets.
    """
    def __init__(
        self,
        train_a_subdir: str = "trainA",
        train_b_subdir: str = "trainB_ghibli",
        test_domain_a_dir: str = None,
        test_domain_b_dir: str = None,
        dataset_id: str = None,
        transform=None,
        batch_size: int = 16,
        num_workers: int = 4,
        split_ratio: float = 0.8,
    ):
        super().__init__()
        self._domain_a_dir = None
        self._domain_b_dir = None
        self.train_a_subdir = train_a_subdir
        self.train_b_subdir = train_b_subdir
        self.test_domain_a_dir = test_domain_a_dir
        self.test_domain_b_dir = test_domain_b_dir
        self.dataset_id = dataset_id
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratio = split_ratio
        self.generator = torch.Generator().manual_seed(42)
  

    def setup(self, stage=None):
        """
        Setup datasets for training, validation, or testing.

        Args:
            stage (str, optional): Stage of setup ("fit", "validate", "test", or None for all).
        """
        if self._domain_a_dir is None or self._domain_b_dir is None:
            raise ValueError("Domain directories not properly set. Run prepare_data()")
        full_dataset = UnpairedImageDataset(
            domain_a_dir=self._domain_a_dir,
            domain_b_dir=self._domain_b_dir,
            transform=self.transform
        )
    
        if stage in (None, "fit", "validate"):
            train_size = int(self.split_ratio * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size], generator=self.generator)
    
        if stage == "test":
            if self.test_domain_a_dir is None or self.test_domain_b_dir is None:
                raise ValueError("Test dataset path not set.")
            
            self.test_dataset = UnpairedImageDataset(
                domain_a_dir=self.test_domain_a_dir,
                domain_b_dir=self.test_domain_b_dir,
                transform=self.transform
            )

    def prepare_data(self):
        """
        Download and prepare the dataset from KaggleHub.
        """
        if self.dataset_id:
            dataset_path = dataset_download(self.dataset_id)
            domain_a = os.path.join(dataset_path, self.train_a_subdir)
            domain_b = os.path.join(dataset_path, self.train_b_subdir)
            
            self._domain_a_dir = domain_a
            self._domain_b_dir = domain_b
            
            print(f"Domain A path: {self._domain_a_dir}")
            print(f"Domain B path: {self._domain_b_dir}")
            
            if self.test_domain_a_dir and self.test_domain_b_dir:
                print(f"Test domain path: {self.test_domain_a_dir}")
                print(f"Test domain path: {self.test_domain_b_dir}")


    def train_dataloader(self):
        """
        Returns the DataLoader for the training set.

        Returns:
            DataLoader: Training DataLoader.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, generator=self.generator)

    def val_dataloader(self):
        """
        Returns the DataLoader for the validation set.

        Returns:
            DataLoader: Validation DataLoader.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, generator=self.generator)

    def test_dataloader(self):
        """
        Returns the DataLoader for the test set.

        Returns:
            DataLoader: Test DataLoader.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, generator=self.generator)

