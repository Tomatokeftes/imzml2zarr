import numpy as np
from pyimzml.ImzMLParser import ImzMLParser as PyImzMLParser
import os
import tempfile
from tqdm import tqdm
import zarr
import heapq

class FileParser:
    def __init__(self, file_path):
        self.file_path = file_path

    def collect_data(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def collect_metadata(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
class ProcessedImzMLParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.parser = PyImzMLParser(file_path)
        self.num_pixels = len(self.parser.coordinates)

    def collect_metadata(self):
        """
        Collect unique m/z values and pixel coordinates from an imzML file.

        Returns:
            tuple: Unique sorted m/z values (1D numpy array) and pixel coordinates (2D numpy array).
        """
        # Create temporary files for m/z values
        temp_mz_file = tempfile.NamedTemporaryFile(delete=False)
        temp_unique_mz_file = tempfile.NamedTemporaryFile(delete=False)

        # Write m/z values to a temporary file
        self._write_mz_values_to_tempfile(temp_mz_file)

        # Close temp_mz_file to release it for reading
        temp_mz_file.close()

        # Deduplicate m/z values using hashes
        self._merge_sorted_chunks([temp_mz_file.name], temp_unique_mz_file.name)

        # Close temp_unique_mz_file after deduplication
        temp_unique_mz_file.close()

        # Read unique m/z values from the deduplicated file
        unique_mz_values = np.fromfile(temp_unique_mz_file.name, dtype=np.float64)
        unique_mz_values = np.sort(unique_mz_values)

        # Collect pixel coordinates
        pixel_coords = np.array(self.parser.coordinates)

        # Clean up temporary files
        self._cleanup_temp_files([temp_mz_file.name, temp_unique_mz_file.name])

        return unique_mz_values, pixel_coords

    def _write_mz_values_to_tempfile(self, temp_mz_file, batch_size=1000):
        """
        Writes sorted m/z values from each pixel in the imzML file to a temporary file in sorted batches.
        """
        mz_set = set()

        for idx in tqdm(range(self.num_pixels), desc="First Pass: Collecting Metadata"):
            mz_values, _ = self.parser.getspectrum(idx)
            mz_set.update(mz_values)

            if len(mz_set) >= batch_size:
                # Sort the batch
                sorted_batch = sorted(mz_set)
                # Write sorted m/z values to the temp file using np.tofile (binary)
                np.array(sorted_batch, dtype=np.float64).tofile(temp_mz_file)
                mz_set.clear()

        # Write any remaining sorted m/z values to the file
        if mz_set:
            sorted_batch = sorted(mz_set)
            np.array(sorted_batch, dtype=np.float64).tofile(temp_mz_file)

        temp_mz_file.close()

    def _merge_sorted_chunks(self, input_filenames, output_filename, chunk_size=1_000_000):
        """
        Merges multiple sorted chunks of m/z values into a single sorted output.
        """
        # Open all input files
        open_files = [open(filename, 'rb') for filename in input_filenames]

        def read_chunk(file):
            """Helper function to read a chunk from a file."""
            return np.fromfile(file, dtype=np.float64, count=chunk_size)

        # Read initial chunks from all files
        iterators = [iter(read_chunk(f)) for f in open_files]

        # Perform n-way merge using heapq.merge (memory efficient merge)
        with open(output_filename, 'wb') as f_out:
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
            os.remove(filename)