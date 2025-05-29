# prepare the dataset for training

import os
import cv2
import torch
import torch.nn.functional as F
import torch.utils.data as data
import bisect
import math
from pathlib import Path
from tqdm import tqdm
import json
import gc
import numpy as np

def get_device():
    """Get the device to use for training"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def gather_video_sequences(root_path):
    # Dictionary to store frames grouped by video sequence
    video_sequences = {}
    
    # Valid innermost folder names
    valid_folders = {'h', 'lr', 'mb'}
    
    # Walk through all directories
    for root, _, files in os.walk(root_path):
        if 'over' in root:
            continue
        
        # Get the innermost folder name
        current_folder = os.path.basename(root)
        
        # Skip if the innermost folder is not one of the valid names
        if current_folder not in valid_folders:
            continue
            
        frame_files = list(filter(lambda f: f.endswith('.png'), files))
   
        if not frame_files:
            continue
        
        frame_files.sort()
        video_sequences[root] = frame_files
    
    return video_sequences

class VideoFrameDataset(data.Dataset):
    def __init__(self, dataset, segment_size=1024, time_step=5):
        self.dataset = dataset
        self.segment_size = segment_size
        self.ids = list(self.dataset.keys())
        self.gap = segment_size - time_step + 1
        self.lengths = [math.ceil(len(self.dataset[i]) / self.gap) for i in self.ids]
        self.cumulative_lengths = np.cumsum(self.lengths)
        

    def __len__(self):
        return sum(self.lengths)
    
    def _get_frame(self, video_id, frame_name):
        # Read and process frame
        frame = cv2.imread(os.path.join(self.ids[video_id], frame_name), cv2.IMREAD_UNCHANGED)
        frame = torch.from_numpy(frame).float() / 65536.0  # Shape: (H, W)
        return frame
    
    def __getitem__(self, idx):
        video_id = bisect.bisect_right(self.cumulative_lengths, idx)
        start_idx = idx - self.cumulative_lengths[video_id - 1] if video_id > 0 else idx
        start_idx = start_idx * self.gap
        end_idx = min(start_idx + self.segment_size, len(self.dataset[self.ids[video_id]]))
        
        # Get frames
        frames = [self._get_frame(video_id, frame_name) 
                 for frame_name in self.dataset[self.ids[video_id]][start_idx:end_idx]]
        
        # Stack frames to get (T, H, W)
        combined = torch.stack(frames)
        return [video_id, start_idx, end_idx], combined.unsqueeze(0)

class DataPreparator:
    def __init__(self, dataset, batch_size, num_workers, spatiotemporal):
        self.device = get_device()
        self.spatiotemporal = spatiotemporal
        self.kernel = torch.ones(1, 1, *spatiotemporal, requires_grad=False) / np.prod(spatiotemporal)
        self.kernel = self.kernel.to(self.device)
        
        # Initialize DataLoader for efficient batch processing
        self.dataloader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,  # Faster data transfer to GPU
            drop_last=False   # Keep all data
        )
    
    def prepare_batch(self, batch):
        """Prepare a batch of data for GPU processing"""
        idx, data = batch
        return idx, data.to(self.device)
    
    @torch.no_grad()
    def process_batch(self, images):
        """Process a batch of data"""
        squared = torch.square(images)
        squared_mean = F.conv3d(squared, self.kernel)
        del squared
        mean = F.conv3d(images, self.kernel)
        del images
        mean_squared = torch.square(mean)
        del mean
        variance = squared_mean - mean_squared
        output = mean_squared / variance
        del squared_mean
        del mean_squared
        del variance
        return output
    
    def get_batch_iterator(self):
        """Get an iterator for batches"""
        for batch in self.dataloader:
            yield self.prepare_batch(batch)

def process_and_save_dataset(input_dir, output_dir, spatiotemporal):
    """Process dataset and save results"""
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize dataset and preparator
    data = gather_video_sequences(input_dir)
    dataset = VideoFrameDataset(data, segment_size=1000 + spatiotemporal[0], time_step=spatiotemporal[0])
    preparator = DataPreparator(dataset, batch_size=1, num_workers=1, spatiotemporal=spatiotemporal)

    for idx, batch in tqdm(preparator.get_batch_iterator(), desc="Processing batches"):
        print(idx, batch.shape)
        processed = preparator.process_batch(batch)
        
        folder_name = output_path / f"{idx[0]}" / f"{idx[1]}_{idx[2]}"
        folder_name.mkdir(parents=True, exist_ok=True)
        for i in range(processed.shape[2]):
            video_file_name = folder_name / f"{i}.npy"
            tmp = processed[0,0,i].cpu().numpy()
            np.save(video_file_name, tmp)
        
        print(idx, processed.shape)
        
        # Clear memory
        del batch
        del processed
        gc.collect()
        torch.cuda.empty_cache()
    
        print(f"Completed processing {video_file_name}")

    data['video_ids'] = list(enumerate(data.keys()))
    with open(output_path / "data_description.json", "w") as f:
        json.dump(data, f)

if __name__ == "__main__":
    input_dir = input("Input directory: ")
    output_dir = input("Output directory: ")
    num_workers = int(input("Number of workers: "))
    spatiotemporal = tuple(int(x) for x in input("Spatiotemporal: ").split(","))
    process_and_save_dataset(input_dir, output_dir, spatiotemporal)
