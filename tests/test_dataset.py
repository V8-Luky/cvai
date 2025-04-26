from data.dataset import UnpairedImageDataset
from PIL import Image
import numpy as np
from data.datamodule import UnpairedDataModule
from pathlib import Path

def test_unpaired_dataset(tmp_path):
    domain_a = tmp_path / "domain_a"
    domain_b = tmp_path / "domain_b"
    domain_a.mkdir()
    domain_b.mkdir()

    img = Image.fromarray(np.uint8(np.random.rand(256, 256, 3) * 255))
    img.save(domain_a / "a.jpg")
    img.save(domain_b / "b.jpg")

    dataset = UnpairedImageDataset(str(domain_a), str(domain_b))
    sample = dataset[0]

    assert sample["a"].shape == (3, 256, 256)
    assert sample["b"].shape == (3, 256, 256)

def test_unpaired_dataset_larger_images(tmp_path):
    domain_a = tmp_path / "domain_a"
    domain_b = tmp_path / "domain_b"
    domain_a.mkdir()
    domain_b.mkdir()

    img = Image.fromarray(np.uint8(np.random.rand(512, 512, 3) * 255))
    img.save(domain_a / "a.jpg")
    img.save(domain_b / "b.jpg")

    dataset = UnpairedImageDataset(str(domain_a), str(domain_b))
    sample = dataset[0]

    assert sample["a"].shape == (3, 256, 256)
    assert sample["b"].shape == (3, 256, 256)

def test_datamodule_batch():
    path = Path("/home/jovyan/.cache/kagglehub/datasets/shubham1921/real-to-ghibli-image-dataset-5k-paired-images/versions/1/dataset")
    dm = UnpairedDataModule(
        dataset_id="shubham1921/real-to-ghibli-image-dataset-5k-paired-images",
        train_a_subdir="dataset/trainA",
        train_b_subdir="dataset/trainB_ghibli",
        test_domain_dir=str(path / "trainA"),
        batch_size=2,
        num_workers=0
    )
    dm.prepare_data() 
    dm.setup(stage="fit")

    train_batch = next(iter(dm.train_dataloader()))
    assert train_batch["a"].shape == (2, 3, 256, 256)
    assert train_batch["b"].shape == (2, 3, 256, 256)

    dm.setup(stage="test")
    test_batch = next(iter(dm.test_dataloader()))
    assert test_batch["a"].shape == (2, 3, 256, 256)
    assert test_batch["b"] is None


