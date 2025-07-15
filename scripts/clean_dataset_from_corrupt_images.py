import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm.auto import tqdm
import sys
# Assuming PreprocessedSatelliteDataset is defined in your project
from runner import Runner
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

class CustomDataset(Dataset):
    """
    Dataset class for preprocessed satellite imagery.
    """

    def __init__(self, data_path, dataframe=None):
        df = pd.read_csv(dataframe)
            
        self.files = list(df["paths"].apply(lambda x: os.path.join(data_path, x)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        data = np.load(file)
        try:
            image = data["data"].astype(np.float32)
            label = data["labels"].astype(np.float32)
            return_index = -1  # If we get here, the image is not corrupt and we return -1
        except Exception as e:
            return_index = index  # The image is corrupt and we return the actual index
        
        return return_index



def clean_dataset(dataset_name, split, num_workers_default=4):
    # Set up dataset and DataLoader
    rootPath = Runner.get_dataset_root(dataset_name=dataset_name)
    dataframe = os.path.join(rootPath, f'{split}.csv')


    dataset = CustomDataset(data_path=rootPath, dataframe=dataframe)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = num_workers_default * torch.cuda.device_count()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    corrupt_img_list = []
    # Process each batch
    with torch.no_grad():
        for index in tqdm(dataloader):
            # Move index to CUDA
            index = index.to(device=device, non_blocking=True)

            # Add the corrupt indices to the list
            corrupt_img_list.extend(index[index != -1].tolist())
    

    sys.stdout.write(f"Found {len(corrupt_img_list)} corrupt images.\n")
    sys.stdout.write(f"Corrupt indices: {corrupt_img_list}.\n")

    # Open the original csv file with pandas, delete the corrupt images and save the new csv file
    if len(corrupt_img_list) == 0:
        sys.stdout.write("No corrupt images found. Exiting.\n")
        return
    df = pd.read_csv(dataframe)
    old_length = len(df)
    df = df.drop(corrupt_img_list)
    new_length = len(df)
    sys.stdout.write(f"Removed {old_length - new_length} corrupt images.\n")
    df.to_csv(dataframe, index=False)

# Usage example
dataset_name = 'ai4forest_rh98'
split = 'val'
clean_dataset(dataset_name, split)