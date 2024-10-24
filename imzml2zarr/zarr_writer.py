import zarr
import sparse
import numpy as np
from tqdm import tqdm

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
        # Create Zarr array with chunk sizes
        z = zarr.zeros(shape=self.shape, chunks=chunks, dtype=np.uint32, store=self.zarr_store_path, mode='w', overwrite=True)
        return z

    def write_data_in_chunks(self, imzml_parser):
        """
        Writes mass spectrometry imaging data to a Zarr array in chunks using the given ImzMLParser.
        """
        z = self.create_zarr_store()

        # Process data in chunks
        chunk_size = z.chunks[0]  # Use the row chunk size from Zarr array
        num_chunks = (self.num_pixels + chunk_size - 1) // chunk_size

        # Create the main progress bar for the entire dataset
        with tqdm(total=self.num_pixels, desc="Overall Progress", unit="pixel") as pbar:
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, self.num_pixels)

                # Process each chunk and write to Zarr
                self._process_chunk(imzml_parser, z, pbar, chunk_idx, start_idx, end_idx)

        self._add_zarr_metadata(z)
        print(f"Zarr array created successfully at {self.zarr_store_path}.")

    def _process_chunk(self, imzml_parser, z, pbar, chunk_idx, start_idx, end_idx):
        """
        Processes a chunk of pixels and writes the data to the Zarr array.
        """
        chunk_pixels = range(start_idx, end_idx)
        chunk_length = end_idx - start_idx

        # Initialize data structures for the current chunk
        coords = [[], []]  # Coordinates for non-zero values (row index within chunk, column index)
        data = []

        # Process each pixel in the current chunk
        for idx_in_chunk, idx in enumerate(tqdm(chunk_pixels, desc=f"Processing chunk {chunk_idx + 1}", leave=False)):
            mz_values, intensities = imzml_parser.parser.getspectrum(idx)
            for mz, intensity in zip(mz_values, intensities):
                column_index = self.mz_index_map.get(mz, -1)
                if column_index >= 0:
                    coords[0].append(idx_in_chunk)  # row index within chunk
                    coords[1].append(column_index)  # column index
                    data.append(intensity)

            # Update the main progress bar
            pbar.update(1)

        # Write sparse COO array to Zarr
        if data:
            self._write_sparse_array_to_zarr(coords, data, chunk_length, z, start_idx)
        else:
            print(f"No data in chunk {chunk_idx}")

    def _write_sparse_array_to_zarr(self, coords, data, chunk_length, z, start_idx):
        """
        Writes a sparse COO array to the Zarr array.
        """
        coords = np.array(coords)
        data = np.array(data)
        chunk_shape = (chunk_length, self.num_mz)
        coo_array = sparse.COO(coords, data, shape=chunk_shape)

        # Adjust row indices to absolute indices
        row_indices = coo_array.coords[0] + start_idx
        col_indices = coo_array.coords[1]

        # Write data to Zarr array
        z.set_coordinate_selection((row_indices, col_indices), coo_array.data)

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
