import os
import random
import torch
import numpy as np
import torch.nn.functional as F
import torch.utils.data as data

def center_crop(img: np.ndarray, crop_size: int) -> np.ndarray:
    """Center crop an image to the specified size.
    
    Args:
        img (np.ndarray): Input image of shape (H, W) or (H, W, C)
        crop_size (int): Size of the square crop
        
    Returns:
        np.ndarray: Center cropped image
        
    Raises:
        ValueError: If image is smaller than crop_size
    """
    h, w = img.shape
    if h < crop_size or w < crop_size:
        raise ValueError(f"Image size ({h}, {w}) is smaller than crop size {crop_size}")
    
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return img[start_h:start_h+crop_size, start_w:start_w+crop_size]

class NumpyFolderDataset(data.Dataset):
    def __init__(self, folder_path: str, crop_size: int = 512):
        """Initialize dataset with center cropping.
        
        Args:
            folder_path (str): Path to folder containing .npy files
            crop_size (int): Size of center crop
            min_size (int): Minimum image size required
        """
        self.folder_path = folder_path
        self.crop_size = crop_size
        
        # Filter files and validate sizes
        self.files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        try:
            img = np.load(os.path.join(self.folder_path, self.files[idx]))
            img = center_crop(img, self.crop_size)
            tensor = torch.from_numpy(img).float().unsqueeze(0)
            return tensor
        except Exception as e:
            print(f"Error loading {self.files[idx]}: {e}")
            # Return a random valid file instead
            return self.__getitem__(random.randint(0, len(self) - 1))

class TransformedDataset(data.Dataset):
    """Dataset that applies a transform to its items"""
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        if isinstance(item, tuple):
            # If dataset returns (input, target) pairs
            result = tuple(self.transform(x) for x in item)
        else:
            # If dataset returns single items
            result = self.transform(item)
        return result

class TemporalPairedDataset(data.Dataset):
    def __init__(self, lr_dataset, lr_temporal, hr_dataset, hr_temporal):
        self.lr_dataset = lr_dataset
        self.hr_dataset = hr_dataset
        self.hr_center = hr_temporal // 2
        self.lr_start = self.hr_center - (lr_temporal // 2)
        
    def __len__(self):
        return len(self.hr_dataset)
    
    def __getitem__(self, idx):
        lr_data = self.lr_dataset[self.lr_start + idx]
        hr_data = self.hr_dataset[idx]
        return lr_data, hr_data

class RandomPairDataset(data.Dataset):
    def __init__(self, dataset, intervals):
        self.dataset = dataset
        self.intervals = intervals
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        interval = random.choice(self.intervals)
        lr_data = self.dataset[idx]
        if idx + interval < len(self.dataset):
            hr_data = self.dataset[idx + interval]
        elif idx - interval >= 0:
            hr_data = self.dataset[idx - interval]
        else:
            hr_data = self.dataset[idx]
        if random.random() < 0.5:
            return lr_data, hr_data
        else:
            return hr_data, lr_data

def create_train_dataset(train_dir, intervals):
    # Create dataset for each subdirectory
    datasets = []
    for subdir in os.listdir(train_dir):
        subdir_path = os.path.join(train_dir, subdir)
        if os.path.isdir(subdir_path):
            dataset = NumpyFolderDataset(subdir_path)
            dataset = RandomPairDataset(dataset, intervals)
            datasets.append(dataset)
    
    # Combine all datasets
    dataset = data.ConcatDataset(datasets)
    # Apply transform to dataset
    dataset = TransformedDataset(dataset, torch.log1p)
    return dataset

def create_val_dataset(noise_dir, refer_dir):
    # Create dataset
    dataset = TemporalPairedDataset(
        lr_dataset=TransformedDataset(NumpyFolderDataset(noise_dir), torch.log1p),
        hr_dataset=NumpyFolderDataset(refer_dir),
        lr_temporal=1,
        hr_temporal=1
    )
    return dataset
