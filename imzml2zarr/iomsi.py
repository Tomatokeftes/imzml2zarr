import numpy as np
from pyimzml.ImzMLParser import ImzMLParser as PyImzMLParser
import os
import tempfile
from tqdm import tqdm
import heapq

class FileParser:
    def __init__(self, file_path):
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
        super().__init__(file_path)
        self.parser = PyImzMLParser(file_path)
        self.num_pixels = len(self.parser.coordinates)

    def collect_metadata(self):
        """
        Collect unique m/z values and pixel coordinates from an imzML file.

        Returns:
            tuple: Unique sorted m/z values (1D numpy array) and pixel coordinates (2D numpy array).
        """
        # Create temporary files for m/z values
        with tempfile.NamedTemporaryFile(delete=False) as temp_mz_file, \
             tempfile.NamedTemporaryFile(delete=False) as temp_unique_mz_file:

            # Write m/z values to a temporary file
            self._write_mz_values_to_tempfile(temp_mz_file)

            # Deduplicate m/z values using hashes
            self._merge_sorted_chunks([temp_mz_file.name], temp_unique_mz_file.name)

            # Read unique m/z values from the deduplicated file
            unique_mz_values = np.fromfile(temp_unique_mz_file.name, dtype=np.float64)
            unique_mz_values = np.sort(unique_mz_values)

        # Clean up temporary files after ensuring they're closed
        self._cleanup_temp_files([temp_mz_file.name, temp_unique_mz_file.name])

        # Collect pixel coordinates
        pixel_coords = np.array(self.parser.coordinates)

        return unique_mz_values, pixel_coords

    def collect_data(self):
        """
        Collect spectra and intensity data for each pixel.
        This method iterates through the pixels and retrieves the data.
        """
        data = []
        for idx in tqdm(range(self.num_pixels), desc="Collecting Data"):
            mz_values, intensities = self.parser.getspectrum(idx)
            data.append((mz_values, intensities))
        return data
    
    def _write_mz_values_to_tempfile(self, temp_mz_file, batch_size=1000):
        """
        Writes sorted m/z values from each pixel in the imzML file to a temporary file in sorted batches.
        """
        mz_set = set()

        for idx in tqdm(range(self.num_pixels), desc="First Pass: Collecting Metadata"):
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
        Merges multiple sorted chunks of m/z values into a single sorted output.
        """
        with open(output_filename, 'wb') as f_out:
            # Open all input files
            open_files = [open(filename, 'rb') for filename in input_filenames]

            def read_chunk(file):
                return np.fromfile(file, dtype=np.float64, count=chunk_size)

            # Read initial chunks from all files
            iterators = [iter(read_chunk(f)) for f in open_files]

            # Perform n-way merge using heapq.merge
            for value in heapq.merge(*iterators):
                np.array([value], dtype=np.float64).tofile(f_out)

            # Close all input files
            for f in open_files:
                f.close()

    def _cleanup_temp_files(self, filenames):
        """
        Removes temporary files after processing.
        """
        for filename in filenames:
            if os.path.exists(filename):
                os.remove(filename)

    def collect_data_in_chunks(self, chunk_size):
        """
        Yields chunks of spectra data in the form of (m/z values, intensities).
        This method yields chunk_size pixels at a time to avoid memory overload.
        """
        for start_idx in range(0, self.num_pixels, chunk_size):
            end_idx = min(start_idx + chunk_size, self.num_pixels)
            chunk_data = []

            for idx in range(start_idx, end_idx):
                mz_values, intensities = self.parser.getspectrum(idx)
                chunk_data.append((mz_values, intensities))

            yield chunk_data
