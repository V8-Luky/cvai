from pathlib import Path
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class UnpairedImageDataset(Dataset):
    def __init__(
        self,
        domain_a_dir: str,
        domain_b_dir: str,
        transform: T.Compose = None,
        image_extensions: tuple[str] = (".jpg", ".jpeg", ".png"),
    ):
        super().__init__()
        self.domain_a_paths = self._load_paths(domain_a_dir, image_extensions)
        self.domain_b_paths = self._load_paths(domain_b_dir, image_extensions)

        self.transform = transform or T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])

    def _load_paths(self, directory: str, exts) -> list[Path]:
        return sorted([
            p for p in Path(directory).glob("*") if p.suffix.lower() in exts
        ])

    def __len__(self):
        return max(len(self.domain_a_paths), len(self.domain_b_paths))

    def __getitem__(self, idx):
        a_img_path = self.domain_a_paths[idx % len(self.domain_a_paths)]
        b_img_path = self.domain_b_paths[torch.randint(len(self.domain_b_paths), ()).item()]
    
        a_img = Image.open(a_img_path).convert("RGB")
        b_img = Image.open(b_img_path).convert("RGB")

        return {
            "a": self.transform(a_img),
            "b": self.transform(b_img)
        }
