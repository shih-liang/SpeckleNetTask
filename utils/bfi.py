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

torch.backends.cudnn.deterministic = True 

def normalize_frame(frame):
    frame = torch.from_numpy(frame).double() / 16384.0  # Shape: (H, W)
    return frame

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
    def __init__(self, data_path, segment_size=1024, time_step=5):
        self.time_step = time_step
        self.folder_description = gather_video_sequences(data_path)
        self.segment_size = segment_size
        self.ids = list(self.folder_description.keys())
        self.gap = segment_size - time_step + 1
        self.lengths = [self._compute_length(i) for i in self.ids]
        self.cumulative_lengths = np.cumsum(self.lengths)

    def get_data_description(self):
        return {i: {id: self.folder_description[id]} for i, id in enumerate(self.ids)}
    
    def _compute_length(self, video_id):
        length = len(self.folder_description[self.ids[video_id]])
        div, mod = divmod(length, self.gap)
        if mod <= self.time_step:
            return div
        return div + 1
    
    def __len__(self):
        return sum(self.lengths)
    
    def _get_frame(self, video_id, frame_name):
        # Read and process frame
        frame = cv2.imread(os.path.join(self.ids[video_id], frame_name), cv2.IMREAD_UNCHANGED)
        frame = normalize_frame(frame)
        return frame
    
    def __getitem__(self, idx):
        video_id = bisect.bisect_right(self.cumulative_lengths, idx)
        start_idx = idx - self.cumulative_lengths[video_id - 1] if video_id > 0 else idx
        start_idx = start_idx * self.gap
        end_idx = min(start_idx + self.segment_size, self.lengths[video_id])
        if end_idx - self.lengths[video_id] <= self.time_step:
            end_idx = self.lengths[video_id]
        
        # Get frames
        frames = [self._get_frame(video_id, frame_name) 
                 for frame_name in self.data_description[self.ids[video_id]][start_idx:end_idx]]
        
        # Stack frames to get (T, H, W)
        combined = torch.stack(frames)
        return [video_id, start_idx, end_idx], combined.unsqueeze(0)

class DataPreparator:
    def __init__(self, dataset, num_workers, spatiotemporal):
        self.device = get_device()
        self.spatiotemporal = spatiotemporal
        self.kernel = torch.ones(1, 1, *spatiotemporal, requires_grad=False).double() / np.prod(spatiotemporal)
        self.kernel = self.kernel.to(self.device)
        
        # Initialize DataLoader for efficient batch processing
        self.dataloader = data.DataLoader(
            dataset,
            batch_size=1,
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
    dataset = VideoFrameDataset(input_dir, segment_size=1000 + spatiotemporal[0], time_step=spatiotemporal[0])
    preparator = DataPreparator(dataset, num_workers=1, spatiotemporal=spatiotemporal)

    for idx, batch in tqdm(preparator.get_batch_iterator(), desc="Processing batches"):
        processed = preparator.process_batch(batch)
        
        folder_name = output_path / f"{idx[0].item()}"
        start_idx = idx[1].item()
        end_idx = idx[2].item()
        folder_name.mkdir(parents=True, exist_ok=True)
        print(f"Processed video {idx[0].item()} with shape: {processed.shape} and frames: {start_idx} - {end_idx}")
        for i in range(processed.shape[2]):
            video_file_name = folder_name / f"{i + start_idx}.npy"
            tmp = processed[0,0,i].cpu().numpy()
            np.save(video_file_name, tmp)
        
        # Clear memory
        del batch
        del processed
        gc.collect()
        torch.cuda.empty_cache()
    
        print(f"Completed processing {video_file_name}")

    data_description = dataset.get_data_description()
    with open(output_path / "data_description.json", "w") as f:
        json.dump(data_description, f)

if __name__ == "__main__":
    input_dir = input("Input directory: ")
    output_dir = input("Output directory: ")
    num_workers = int(input("Number of workers: "))
    spatiotemporal = tuple(int(x) for x in input("Spatiotemporal: ").split(","))
    process_and_save_dataset(input_dir, output_dir, spatiotemporal)
