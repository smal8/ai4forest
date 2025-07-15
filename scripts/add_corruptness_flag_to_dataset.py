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
        image = data["data"].astype(np.float32)

        return image, index



def clean_dataset(dataset_name, split, num_workers_default=4):
    # Set up dataset and DataLoader
    rootPath = Runner.get_dataset_root(dataset_name=dataset_name)
    dataframe = os.path.join(rootPath, f'{split}.csv')


    dataset = CustomDataset(data_path=rootPath, dataframe=dataframe)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = num_workers_default * torch.cuda.device_count()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True)

    corrupt_img_list = []
    # Process each batch
    with torch.no_grad():
        for image, index in tqdm(dataloader):
            image = image.to(device=device, non_blocking=True)
            index = index.to(device=device, non_blocking=True)
            # Drop the first 4 channels of each sample
            image = image[:, 4:]

            # Flatten the image
            image = image.flatten(start_dim=2)

            # Sum up the pixels of each channel
            image = torch.sum(image, dim=2)

            # Convert to boolean tensor
            image = image == 0

            # Get the indices where at least one channel is all zero
            image = torch.any(image, dim=1)

            # Add the indices to the list
            corrupt_img_list.extend(index[image].tolist())

    sys.stdout.write(f"Found {len(corrupt_img_list)} corrupt images.\n")
    sys.stdout.write(f"Corrupt indices: {corrupt_img_list}.\n")

    # Open the original csv file with pandas, delete the corrupt images and save the new csv file
    if len(corrupt_img_list) == 0:
        sys.stdout.write("No corrupt images found. Exiting.\n")
        return
    df = pd.read_csv(dataframe)

    # Create a new column named "has_corrupt_s2_channel_flag" and set it to False everywhere, except for the indices in corrupt_img_list
    df["has_corrupt_s2_channel_flag"] = False
    df.loc[corrupt_img_list, "has_corrupt_s2_channel_flag"] = True
    df.to_csv(dataframe, index=False)

# Usage example
dataset_name = 'ai4forest_rh98'
split = 'train'
clean_dataset(dataset_name, split)