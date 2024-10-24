import zarr
import sparse
import numpy as np
from tqdm import tqdm
import xarray as xr

class ZarrWriter:
    def __init__(self, zarr_store_path, unique_mz_values, pixel_coords):
        self.zarr_store_path = zarr_store_path
        self.unique_mz_values = unique_mz_values
        self.pixel_coords = pixel_coords
        self.mz_index_map = {mz: i for i, mz in enumerate(unique_mz_values)}
        self.num_pixels = len(pixel_coords)
        self.num_mz = len(unique_mz_values)
        self.shape = (self.num_pixels, self.num_mz)

    def create_zarr_store(self, chunks=(16384, 5000)):
        """
        Create a Zarr store for writing the data.
        """
        z = zarr.zeros(shape=self.shape, chunks=chunks, dtype=np.uint32, store=self.zarr_store_path, overwrite=True)
        return z

    def write_data_in_chunks(self, parser, chunk_size=1000):
        """
        Writes mass spectrometry imaging data to a Zarr array in chunks using the given parser.
        """
        z = self.create_zarr_store()

        # Process data in chunks
        total_pixels = self.num_pixels
        with tqdm(total=total_pixels, desc="Overall Progress", unit="pixel") as pbar:
            for chunk_data in parser.collect_data_in_chunks(chunk_size):
                self._process_chunk(z, chunk_data, pbar)

        self._add_zarr_metadata(z)
        print(f"Zarr array created successfully at {self.zarr_store_path}.")

    def _process_chunk(self, z, chunk_data, pbar):
        """
        Processes a chunk of pixels and writes the data to the Zarr array.
        """
        coords = [[], []]  # Coordinates for non-zero values (row index within chunk, column index)
        data = []

        for idx_in_chunk, (mz_values, intensities) in enumerate(chunk_data):
            for mz, intensity in zip(mz_values, intensities):
                column_index = self.mz_index_map.get(mz, -1)
                if column_index >= 0:
                    coords[0].append(idx_in_chunk)
                    coords[1].append(column_index)
                    data.append(intensity)

            pbar.update(1)

        # Write sparse COO array to Zarr
        if data:
            self._write_sparse_array_to_zarr(coords, data, z)

    def _write_sparse_array_to_zarr(self, coords, data, z):
        """
        Writes a sparse COO array to the Zarr array.
        """
        coords = np.array(coords)
        data = np.array(data)
        chunk_shape = (len(data), self.num_mz)
        coo_array = sparse.COO(coords, data, shape=chunk_shape)

        # Write data to Zarr array
        z.set_coordinate_selection((coo_array.coords[0], coo_array.coords[1]), coo_array.data)

    def _add_zarr_metadata(self, z):
        """
        Adds metadata to the Zarr store.
        """
        z.attrs['description'] = "Mass spectrometry imaging data in Zarr format"
        z.attrs['num_pixels'] = self.num_pixels
        z.attrs['num_mz'] = self.num_mz
        z.attrs['mz_range'] = (self.unique_mz_values[0], self.unique_mz_values[-1]) if self.unique_mz_values.size > 0 else (None, None)
        z.attrs['pixel_coordinates'] = self.pixel_coords.tolist()
        z.attrs['mass_axis'] = self.unique_mz_values.tolist()
