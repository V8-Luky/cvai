from pathlib import Path
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T


class UnpairedImageDataset(Dataset):
    """
    Dataset for unpaired image-to-image translation tasks (e.g., CycleGAN).

    Loads images from two separate domains (A and B). For each sample:
    - Picks an image from domain A based on index.
    - Picks a random image from domain B (if available).
    - Applies transformations and returns both.

    Args:
        domain_a_dir (str): Path to directory with domain A images.
        domain_b_dir (str, optional): Path to directory with domain B images. If None, only domain A is used.
        transform (torchvision.transforms.Compose, optional): Transformations to apply to images.
        image_extensions (tuple[str], optional): Allowed image file extensions.
    """
    def __init__(
        self,
        domain_a_dir: str,
        domain_b_dir: str = None,
        transform: T.Compose = None,
        image_extensions: tuple[str] = (".jpg", ".jpeg", ".png"),
    ):
        super().__init__()
        self.domain_a_paths = self._load_paths(domain_a_dir, image_extensions)
        self.domain_b_paths = self._load_paths(domain_b_dir, image_extensions) if domain_b_dir else []

        self.transform = transform or T.Compose([
            T.ToImage(),
            T.Resize((256, 256)),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize((0.5,), (0.5,))
        ])
        self.generator = torch.Generator().manual_seed(42)

    def _load_paths(self, directory: str, exts) -> list[Path]:
        """
        Loads all image paths from a directory with given extensions.

        Args:
            directory (str): Path to the image directory.
            exts (tuple[str]): Allowed file extensions.

        Returns:
            list[Path]: List of image file paths.
        """
        return sorted([
            p for p in Path(directory).glob("*") if p.suffix.lower() in exts
        ])

    def __len__(self):
        """
        Returns:
            int: Number of samples in the dataset (max of domain A and B sizes).
        """
        return max(len(self.domain_a_paths), len(self.domain_b_paths))

    def __getitem__(self, idx):
        """
        Fetches a single sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Dictionary with:
                - "a" (Tensor): Transformed domain A image.
                - "b" (Tensor or None): Transformed domain B image, or None if not available.
        """
        a_img_path = self.domain_a_paths[idx % len(self.domain_a_paths)]
        a_img = Image.open(a_img_path).convert("RGB")
        
        if self.domain_b_paths:
            b_img_path = self.domain_b_paths[torch.randint(len(self.domain_b_paths), (), generator=self.generator).item()]
            b_img = Image.open(b_img_path).convert("RGB")
            b_tensor = self.transform(b_img)
        else:
            b_tensor = None

        return {
            "a": self.transform(a_img),
            "b": b_tensor
        }
