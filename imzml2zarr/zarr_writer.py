import numpy as np
import os
import tempfile
from tqdm import tqdm
from pyimzml.ImzMLParser import ImzMLParser
import zarr
import sparse

class ZarrWriter:
    def __init__(self, zarr_store_path, unique_mz_values, pixel_coords):
        self.zarr_store_path = zarr_store_path
        self.unique_mz_values = unique_mz_values
        self.pixel_coords = pixel_coords
        self.mz_index_map = {mz: i for i, mz in enumerate(unique_mz_values)}
        self.shape = (len(pixel_coords), len(unique_mz_values))

    def create_zarr_store(self, chunks=(1000, 1000)):
        # Open or create the root of the Zarr store
        z = zarr.open_group(self.zarr_store_path, mode='w')
        
        # Create a 'data' group for storing data arrays
        data_group = z.create_group('data')
        
        # Create a Zarr array within the 'data' group
        data_group.zeros(
            'data_array', shape=self.shape, chunks=chunks, dtype='float32'
        )
        
        # Create a 'metadata' group for storing metadata
        z.create_group('metadata')
        
        return z

    def write_chunks(self, file_parser, chunk_size=1000):
        z = self.create_zarr_store()
        data_array = z['data/data_array']  # Access the Zarr array for writing

        num_pixels = len(self.pixel_coords)
        num_chunks = (num_pixels + chunk_size - 1) // chunk_size

        with tqdm(total=num_pixels, desc="Overall Progress") as pbar:
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, num_pixels)
                chunk_pixels = range(start_idx, end_idx)

                coords = [[], []]
                data = []

                for idx_in_chunk, idx in enumerate(chunk_pixels):
                    mz_values, intensities = file_parser.read_spectrum(idx)
                    for mz, intensity in zip(mz_values, intensities):
                        column_index = self.mz_index_map.get(mz, -1)
                        if column_index >= 0:
                            coords[0].append(idx_in_chunk)
                            coords[1].append(column_index)
                            data.append(intensity)

                    pbar.update(1)

                if data:
                    data = np.array(data)
                    coords = np.array(coords)
                    chunk_shape = (len(chunk_pixels), len(self.unique_mz_values))
                    coo_array = sparse.COO(coords, data, shape=chunk_shape)

                    # Assigning values to the Zarr array using slices
                    row_indices = coo_array.coords[0] + start_idx
                    col_indices = coo_array.coords[1]

                    # Write data to the Zarr array at the specified indices
                    data_array.vindex[row_indices, col_indices] = coo_array.data

        # # Add metadata to the Zarr store after writing data
        # z.attrs['pixel_coordinates'] = self.pixel_coords
        # z.attrs['mass_axis'] = self.unique_mz_values

        return z
