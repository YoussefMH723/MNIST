from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import ToTensor

from pathlib import Path

def download_data(data_path):
    DATASET_PATH = data_path / "MNIST"
    if DATASET_PATH.is_dir():
        print(f"{DATASET_PATH} directory already exists")
        train_data = torchvision.datasets.MNIST(
            root=data_path,
            train=True,
            download=False,
            transform=ToTensor()
        )
        test_data = torchvision.datasets.MNIST(
            root=data_path,
            train=False,
            download=False,
            transform=ToTensor()
        )
        return train_data, test_data
    else:
        print(f"{DATASET_PATH} directory not found, downloading data .....")
        train_data = torchvision.datasets.MNIST(
            root=data_path,
            train=True,
            download=True,
            transform=ToTensor()
        )
        test_data = torchvision.datasets.MNIST(
            root=data_path,
            train=False,
            download=True,
            transform=ToTensor()
        )
        print("Data downloaded successfully....")
        return train_data, test_data
    
def create_dataloaders(data_path="data/", batch_size=64):
    '''
    Creates dataloaders and downloads data if it doesn't exist.

    Args:
        data_path: path of data folder (default: "data/")
        batch_size: (default: 64)

    returns:
        a tuple of (train_dataloader, test_dataloader, class_names)
    '''
    data_path = Path(data_path)
    train_data, test_data = download_data(data_path=data_path)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)
    class_names = train_data.classes
    return train_dataloader, test_dataloader, class_names

