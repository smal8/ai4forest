import h5py
import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm

def create_hdf5_dataset(data_path, dataframe_path, hdf5_path):
    df = pd.read_csv(dataframe_path)
    files = df["paths"].apply(lambda x: os.path.join(data_path, x))

    with h5py.File(hdf5_path, 'w') as hdf:
        for i, file in enumerate(tqdm(files)):
            data = np.load(file)
            images = data['data'].astype(np.float32)
            labels = data['labels'].astype(np.float32)
            hdf.create_dataset(f'image_{i}', data=images, compression='gzip', dtype=np.float16, chunks=True)
            hdf.create_dataset(f'label_{i}', data=labels, compression='gzip', dtype=np.float16, chunks=True)

# Example usage
data_path = '/home/htc/mzimmer/SCRATCH/ai4forest_debug'
split = 'train'

dataframe_path = os.path.join(data_path, f'{split}.csv')
hdf5_path = os.path.join(data_path, f'{split}.hdf5')
create_hdf5_dataset(data_path=data_path, dataframe_path=dataframe_path, hdf5_path=hdf5_path)
