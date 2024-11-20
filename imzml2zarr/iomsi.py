import numpy as np
from pyimzml.ImzMLParser import ImzMLParser as PyImzMLParser
import os
import tempfile
from tqdm import tqdm
import heapq
import xarray as xr

class FileParser:
    def __init__(self, file_path):
        """
        Initializes the FileParser with the given file path.

        Args:
            file_path (str): Path to the file to be parsed.
        """
        self.file_path = file_path

    def collect_metadata(self):
        """
        Collects metadata such as m/z values and pixel coordinates.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def collect_data(self):
        """
        Collects data like spectra and intensities for each pixel.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

class ProcessedImzMLParser(FileParser):
    def __init__(self, file_path):
        """
        Initializes the ProcessedImzMLParser with the given file path and sets up the ImzML parser.

        Args:
            file_path (str): Path to the imzML file to be parsed.
        """
        super().__init__(file_path)
        self.parser = PyImzMLParser(file_path)
        self.num_pixels = len(self.parser.coordinates)

    def collect_metadata(self, batch_size=100000):
        """
        Collects unique m/z values across the dataset, writing to temporary files in batches to avoid memory issues.

        Args:
            batch_size (int): Number of unique m/z values to collect before writing to the temporary file.

        Returns:
            tuple: A tuple containing:
                - unique_mz_values (numpy.ndarray): Array of unique m/z values.
                - pixel_coords (numpy.ndarray): Array of pixel coordinates.
        """
        with tempfile.NamedTemporaryFile(delete=False) as temp_mz_file, \
                tempfile.NamedTemporaryFile(delete=False) as temp_unique_mz_file:
            try:
                # Collect and write temporary m/z values incrementally
                self._write_mz_values_to_tempfile(temp_mz_file, batch_size)
                temp_mz_file.close()
                
                # Merge temporary files into a single sorted, deduplicated array
                self._merge_sorted_chunks([temp_mz_file.name], temp_unique_mz_file.name)
                temp_unique_mz_file.close()
                
                # Load the final unique m/z values
                self.unique_mz_values = np.fromfile(temp_unique_mz_file.name, dtype=np.float64)
                pixel_coords = np.array(self.parser.coordinates)
                return self.unique_mz_values, pixel_coords

            finally:
                self._cleanup_temp_files([temp_mz_file.name, temp_unique_mz_file.name])

    def collect_data(self, chunk_x=128, chunk_y=128):
        unique_mz_values = self.unique_mz_values
        x_coords = sorted(set(coord[0] for coord in self.parser.coordinates))
        y_coords = sorted(set(coord[1] for coord in self.parser.coordinates))

        x_map = {val: idx for idx, val in enumerate(x_coords)}
        y_map = {val: idx for idx, val in enumerate(y_coords)}

        for x_start in tqdm(range(0, len(x_coords), chunk_x), desc="Processing x-coordinate chunks"):
            x_end = min(x_start + chunk_x, len(x_coords))
            for y_start in range(0, len(y_coords), chunk_y):
                y_end = min(y_start + chunk_y, len(y_coords))

                # Initialize chunk_data with full mz_channel size
                chunk_data = np.zeros(
                    (x_end - x_start, y_end - y_start, len(unique_mz_values)),
                    dtype=np.float32
                )

                # Fill chunk_data
                for idx in range(self.num_pixels):
                    mz_values, intensities = self.parser.getspectrum(idx)
                    x, y, _ = self.parser.coordinates[idx]

                    x_idx = x_map[x]
                    y_idx = y_map[y]

                    if x_start <= x_idx < x_end and y_start <= y_idx < y_end:
                        mz_indices = np.searchsorted(unique_mz_values, mz_values)
                        valid_idx = (mz_indices < len(unique_mz_values))
                        chunk_data[x_idx - x_start, y_idx - y_start, mz_indices[valid_idx]] = intensities[valid_idx]

                # Create block_ds
                chunk_ds = xr.Dataset(
                    {
                        "intensity": (
                            ("x_coordinate", "y_coordinate", "mz_channel"), chunk_data
                        )
                    },
                    coords={
                        "x_coordinate": x_coords[x_start:x_end],
                        "y_coordinate": y_coords[y_start:y_end],
                        "mz_channel": unique_mz_values,
                    }
                )

                yield chunk_ds



    def _write_mz_values_to_tempfile(self, temp_mz_file, batch_size):
        """
        Writes m/z values to a temporary file in batches to avoid memory overload.

        Args:
            temp_mz_file (file object): Temporary file object to write m/z values.
            batch_size (int): Number of unique m/z values to collect before writing to the file.
        """
        mz_set = set()

        for idx in tqdm(range(self.num_pixels), desc="Collecting Metadata"):
            mz_values, _ = self.parser.getspectrum(idx)
            mz_set.update(mz_values)

            if len(mz_set) >= batch_size:
                sorted_batch = sorted(mz_set)
                np.array(sorted_batch, dtype=np.float64).tofile(temp_mz_file)
                mz_set.clear()

        if mz_set:
            sorted_batch = sorted(mz_set)
            np.array(sorted_batch, dtype=np.float64).tofile(temp_mz_file)

    def _merge_sorted_chunks(self, input_filenames, output_filename, chunk_size=1_000_000):
        """
        Merges sorted chunks of m/z values from multiple files into a single output file.

        Args:
            input_filenames (list of str): List of input file names containing sorted m/z values.
            output_filename (str): Name of the output file to write merged m/z values.
            chunk_size (int): Number of m/z values to read from each file at a time.
        """
        with open(output_filename, 'wb') as f_out:
            open_files = [open(filename, 'rb') for filename in input_filenames]

            def read_chunk_generator(file):
                while True:
                    chunk = np.fromfile(file, dtype=np.float64, count=chunk_size)
                    if not chunk.size:
                        break
                    yield from chunk

            iterators = [read_chunk_generator(f) for f in open_files]
            last_value = None
            for value in heapq.merge(*iterators):
                if value != last_value:
                    np.array([value], dtype=np.float64).tofile(f_out)
                    last_value = value

            for f in open_files:
                f.close()

    def _cleanup_temp_files(self, filenames):
        """
        Removes temporary files after processing.

        Args:
            filenames (list of str): List of temporary file names to be removed.
        """
        for filename in filenames:
            if os.path.exists(filename):
                os.remove(filename)