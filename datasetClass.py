from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import glob
import os
import ast
import torch
import numpy as np
import pandas as pd
import pdb
from torch.utils.data.dataloader import default_collate
import sys
from typing import Optional
from torchvision.transforms import transforms
import torch.nn.functional as F

import numpy as np
import pandas as pd
import pdb
from torch.utils.data.dataloader import default_collate
import sys
from typing import Optional
from torch.utils.data import Dataset
from pyproj import Transformer

class SatelliteImageDataset(Dataset):
    """
    Dataset class for preprocessed satellite imagery.
    """

    def __init__(self, data_path: str,
                 shift_year: int, 
                 collapse_months: bool = False, 
                 single_month_scaling: bool = True, 
                 scale_adjustments: dict = None, 
                 is_3d_model: bool = False,
                 time_mode: Optional[str] = None,
                 has_sentinel_1: bool = False,
                 geo_encoding_type: str = 'per_pixel',
                 geo_encoding_location: str = 'last'):
        
        self.shift_year = shift_year
        assert collapse_months in [True, False], "collapse_months must be either True or False."
        self.collapse_months = collapse_months
        self.is_3d_model = is_3d_model
        self.single_month_scaling = single_month_scaling
        if self.single_month_scaling and self.collapse_months:
            sys.stderr.write("Warning: Cannot use single_month_scaling = True with collapse_months = True, setting single_month_scaling to False.\n")
            self.single_month_scaling = False
        
        if self.is_3d_model:
            if not self.single_month_scaling:
                sys.stderr.write("Warning: single_month_scaling must be True for 3D models, setting single_month_scaling to True to avoid NaN values.\n")
                self.single_month_scaling = True
        
        # Assert that the time mode is valid
        assert time_mode in ['channel', 'rescale', None, 'None', 'none'], "time_mode must be either 'channel', 'rescale', or None."
        if time_mode in ['None', 'none']:
            self.time_mode = None
        else:
            self.time_mode = time_mode

        self.has_sentinel_1 = has_sentinel_1
        self.geo_encoding_type = geo_encoding_type
        self.geo_encoding_location = geo_encoding_location
        # Print part of the configuration
        sys.stdout.write(f"Single month scaling: {self.single_month_scaling}.\n")
        sys.stdout.write(f"Collapse months: {self.collapse_months}.\n")
        sys.stdout.write(f"Is 3D model: {self.is_3d_model}.\n")
        sys.stdout.write(f"Time mode: {self.time_mode}.\n")

        df = pd.read_csv(os.path.join(data_path, "metadata.csv"))

        # Step 1: Join the 'sentinel_file' column with 'data_path'
        df["files"] = df["tile"].apply(lambda x: os.path.join(data_path, 'samples', x, x))

        # Step 2: Append '_' to the end of the file name
        df["files"] = df["files"] + "_"

        # Step 3: Append the value in column 'sample_id' to the end of the file name
        df["files"] = df["files"] + df["sample_id"].astype(str)

        # Step 4: Add '.npz' to the end of the file name
        df["files"] = df["files"] + ".npz"
        
        self.bounds = df["bounds"]
        
        df["zone"] = df.apply(
            lambda row : int(row["tile"][1:3]), 
            axis = 1
        )
        self.zone = df["zone"]
        
        df["is_northern"] = df.apply(
            lambda row : row["tile"][3] >= 'N',
            axis = 1
        )
        self.is_northern = df["is_northern"]

        # Assign the result to self.files
        self.files = np.array(df["files"]).astype(np.bytes_)
        self.year_data = df["year"].values.astype(np.float32)
        

        self.scale_adjustments = scale_adjustments or {}
        self.scaling_dict = self.get_adjusted_scaling_dict()

    def get_adjusted_scaling_dict(self):
        base_dict = {
            (1, 2, 3, 4): (0, 2000),
            (6, 7, 8, 9): (0, 6000),
            (0,): (0, 1000),
            (5, 10, 11): (0, 4000),
        }

        adjusted_dict = {}
        for channels, (min_val, max_val) in base_dict.items():
            if channels == (1, 2, 3, 4):
                adjustment = self.scale_adjustments.get('scale_adjust_1234', 0.0)
            elif channels == (6, 7, 8, 9):
                adjustment = self.scale_adjustments.get('scale_adjust_6789', 0.0)
            elif channels == (0,):
                adjustment = self.scale_adjustments.get('scale_adjust_0', 0.0)
            elif channels == (5, 10, 11):
                adjustment = self.scale_adjustments.get('scale_adjust_51011', 0.0)
                
            else:
                adjustment = 0.0
            
            if adjustment is None:
                adjustment = 0.0

            adjusted_max = max_val * (1 + adjustment)
            adjusted_dict[channels] = (min_val, adjusted_max)
        sys.stdout.write(f"Adjusted scaling dict: {adjusted_dict}.\n")

        return adjusted_dict
        

    def __len__(self):
        return len(self.files)

    def utm_bounds_to_latlon_grids(self, min_e, min_n, max_e, max_n, zone, is_northern, height=256, width=256):
        """
        Given UTM bounds and zone info, returns 256x256 arrays of latitude and longitude in WGS84.
        
        Args:
            min_e (float): Minimum easting (meters, SW corner)
            min_n (float): Minimum northing (meters, SW corner)
            max_e (float): Maximum easting (meters, NE corner)
            max_n (float): Maximum northing (meters, NE corner)
            zone (int): UTM zone number
            is_northern (bool): True if northern hemisphere, False if southern
            height (int): Output grid height (default 256)
            width (int): Output grid width (default 256)
        
        Returns:
            lat_grid (np.ndarray): (height, width) array of latitudes (degrees)
            lon_grid (np.ndarray): (height, width) array of longitudes (degrees)
        """
        # Create 1D arrays for easting and northing
        eastings = np.linspace(min_e, max_e, width)
        northings = np.linspace(min_n, max_n, height)
        # Create 2D meshgrid
        easting_grid, northing_grid = np.meshgrid(eastings, northings)
        # Set up transformer
        epsg_code = 32600 + zone if is_northern else 32700 + zone
        transformer = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)
        # Transform all points
        lat_grid, lon_grid = transformer.transform(easting_grid, northing_grid)
        return lat_grid, lon_grid

    def __getitem__(self, index):
        file = self.files[index]
        data = np.load(file)    # Columns: sentinel_data (6x12x256x256), gedi_data (3x256x256, first three channels are heights, slope is not given for now). if has_sentinel_1 is True, then sentinel_data is split as seen below
        if self.has_sentinel_1:
            image_s1 = data["sentinel1_data"].astype(np.float32)    # 4x256x256
            
            # Replace any NaN values with 0
            image_s1[np.isnan(image_s1)] = 0

            image_s2 = data["sentinel2_data"].astype(np.float32)    # 6x12x256x256

            # Sentinel 1 channels are given in range -50 to 1 and should be scaled to 0 to 1
            image_s1 = ((image_s1 + 50) / 51)

            # Create the image by concatenating the first 4 channels of image_s1 and the second 12 channels of image_s2, resulting in a tensor of shape 6x16x256x256. Therefore, the s1 channels must be repeated first, to bring the shape to 6x4x256x256
            image_s1 = np.repeat(image_s1[np.newaxis, ...], repeats=image_s2.shape[0], axis=0)  # 6x4x256x256
            image = np.concatenate((image_s2, image_s1), axis=1)  # 6x16x256x256, concatenate first the s2 channels, then the s1 channels, such that the scaling is correctly applied
        else:
            image = data["sentinel_data"].astype(np.float32)
        
        min_e, min_n, max_e, max_n = ast.literal_eval(self.bounds[index])
        lat_grid, lon_grid = self.utm_bounds_to_latlon_grids(min_e, min_n, max_e, max_n, self.zone[index], self.is_northern[index])
        
        lat_grid = lat_grid[np.newaxis, ...]  # (1, 256, 256)
        lon_grid = lon_grid[np.newaxis, ...]  # (1, 256, 256)
        
        lat_grid = np.radians(lat_grid)
        lon_grid = np.radians(lon_grid)
        
        if self.geo_encoding_type == 'per_pixel':
            geo_encoding = np.concatenate([np.sin(np.pi * lat_grid / 180), np.cos(np.pi * lat_grid / 180), np.sin(np.pi * lon_grid / 180), np.cos(np.pi * lon_grid / 180)], axis=0)    
        elif self.geo_encoding_type == 'mean':
            geo_encoding = np.mean(np.concatenate([np.sin(np.pi * lat_grid / 180), np.cos(np.pi * lat_grid / 180), np.sin(np.pi * lon_grid / 180), np.cos(np.pi * lon_grid / 180)], axis=1), axis=0)
        else: # 'none'
            geo_encoding = None
    
        if self.geo_encoding_location == 'bottleneck':
            # Convert NumPy to PyTorch tensor
            geo_encoding = torch.from_numpy(geo_encoding).unsqueeze(0).float()  # shape: (1, 4, 256, 256)
            # Interpolate
            geo_encoding = F.interpolate(
                geo_encoding,
                size=(16, 16),
                mode='bilinear',
                align_corners=False
            )
            # Remove batch dimension and convert back to NumPy
            geo_encoding = geo_encoding.squeeze(0).numpy()  # shape: (4, 16, 16)
        
        if self.collapse_months:
            # Collapse the first dimension by computing the median along this dimension
            original_shape_of_month = image.shape[1:]
            image = image.reshape(image.shape[0], -1)
            image = np.median(image, axis=0)
            # Reshape back to the original shape of the month
            image = image.reshape(*original_shape_of_month)
        else:
            # Potentially apply single month scaling
            if self.single_month_scaling:
                for channels, (min_val, max_val) in self.scaling_dict.items():
                    for channel in channels:
                        image[:, channel] = np.clip(image[:, channel], min_val, max_val)
                        image[:, channel] = (image[:, channel] - min_val) / (max_val - min_val)
            if not self.is_3d_model:
                # Just join the first two dimensions into one, e.g. the resulting shape is either (6*12)x256x256 or (12*12)x256x256 (or use 16 instead of 12 if has_sentinel_1 is True)
                image = image.reshape(image.shape[0] * image.shape[1], *image.shape[2:])
            else:
                # Swap month and channel dimensions: (months, channels, H, W) -> (channels, months, H, W)
                image = image.transpose(1, 0, 2, 3) # This is equivalent to image.permute(1, 0, 2, 3) and should also create a view instead of a copy
        
        # Transform the image into a torch tensor without swapping dimensions
        # Remark: Suboptimal solution. It works and avoids the dimension swapping of transforms.ToTensor(), but it also does not meet the complexity of ToTensor()
        image = torch.from_numpy(image).contiguous()

        # Concatenate the year variable as an additional channel if time_mode is 'channel'
        year = self.year_data[index] - self.shift_year
        year = np.array([year], dtype=np.float32)  # Ensure the correct shape
        if self.time_mode == 'channel':            
            if self.is_3d_model:
                year_channel = np.full((1, image.shape[-3], image.shape[-2], image.shape[-1]), year)
            else:
                year_channel = np.full((1, image.shape[-2], image.shape[-1]), year)
            
            image = np.concatenate((image, year_channel), axis=0)   

        year = torch.tensor(year)
        
        label = data["gedi_data"].astype(np.float32)
        label = torch.from_numpy(label).contiguous()

        return image, label, year, geo_encoding