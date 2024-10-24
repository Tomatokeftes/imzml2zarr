import numpy as np
import os
import tempfile
from tqdm import tqdm
from pyimzml.ImzMLParser import ImzMLParser
import zarr
import sparse
import heapq


def collect_metadata(imzml_file):
    """
    Collect unique m/z values and pixel coordinates from an imzML file.
    
    Args:
        imzml_file (str): Path to the imzML file.
    
    Returns:
        tuple: Unique sorted m/z values (1D numpy array) and pixel coordinates (2D numpy array).
    """
    parser = ImzMLParser(imzml_file)
    num_pixels = len(parser.coordinates)

    # Create temporary files for m/z values
    temp_mz_file = tempfile.NamedTemporaryFile(delete=False)
    temp_unique_mz_file = tempfile.NamedTemporaryFile(delete=False)

    # Write m/z values to a temporary file
    write_mz_values_to_tempfile(parser, temp_mz_file, num_pixels)

    # Close temp_mz_file to release it for reading
    temp_mz_file.close()

    # Deduplicate m/z values using hashes
    merge_sorted_chunks([temp_mz_file.name], temp_unique_mz_file.name)

    # Close temp_unique_mz_file after deduplication
    temp_unique_mz_file.close()

    # Read unique m/z values from the deduplicated file
    unique_mz_values = np.fromfile(temp_unique_mz_file.name, dtype=np.float64)
    unique_mz_values = np.sort(unique_mz_values)

    # Collect pixel coordinates
    pixel_coords = np.array(parser.coordinates)

    # Clean up temporary files
    cleanup_temp_files([temp_mz_file.name, temp_unique_mz_file.name])

    return unique_mz_values, pixel_coords



def write_mz_values_to_tempfile(parser, temp_mz_file, num_pixels, batch_size=1000):
    """
    Writes sorted m/z values from each pixel in the imzML file to a temporary file in sorted batches.
    
    Args:
        parser (ImzMLParser): The ImzMLParser object for parsing the imzML file.
        temp_mz_file (file object): Temporary file object to write sorted m/z values.
        num_pixels (int): Total number of pixels to process.
        batch_size (int): Number of unique m/z values to collect before writing to the file.
    """
    mz_set = set()
    
    for idx in tqdm(range(num_pixels), desc="First Pass: Collecting Metadata"):
        mz_values, _ = parser.getspectrum(idx)
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


def merge_sorted_chunks(input_filenames, output_filename, chunk_size=1_000_000):
    """
    Merges multiple sorted chunks of m/z values into a single sorted output.
    
    Args:
        input_filenames (list of str): Paths to the temporary files containing sorted m/z values.
        output_filename (str): Path to the file where the merged and sorted m/z values will be written.
        chunk_size (int): The size of chunks to read during merging.
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
    
    print("Merging completed successfully.")

def cleanup_temp_files(filenames):
    """
    Removes temporary files after processing.
    
    Args:
        filenames (list of str): List of file paths to remove.
    """
    for filename in filenames:
        os.remove(filename)


def write_data_in_chunks(imzml_file, zarr_store_path, unique_mz_values, pixel_coords):
    """
    Writes mass spectrometry imaging data to a Zarr array in chunks.
    
    Args:
        imzml_file (str): Path to the imzML file.
        zarr_store_path (str): Path to the Zarr store where the data will be written.
        unique_mz_values (numpy array): Array of unique m/z values.
        pixel_coords (numpy array): Array of pixel coordinates.
    """

    parser = ImzMLParser(imzml_file)
    mz_index_map = {mz: i for i, mz in enumerate(unique_mz_values)}
    num_pixels = len(pixel_coords)
    num_mz = len(unique_mz_values)
    shape = (num_pixels, num_mz)

    # Create Zarr array with chunk sizes
    z = zarr.zeros(shape=shape, chunks=(16384, 5000), dtype=np.uint32, store=zarr_store_path, mode='w', overwrite=True)

    # Process data in chunks
    chunk_size = z.chunks[0]  # Use the row chunk size from Zarr array
    num_chunks = (num_pixels + chunk_size - 1) // chunk_size

    # Create the main progress bar for the entire dataset
    with tqdm(total=num_pixels, desc="Overall Progress", unit="pixel") as pbar:
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, num_pixels)

            # Process each chunk and write to Zarr
            process_chunk(parser, mz_index_map, z, pbar, chunk_idx, start_idx, end_idx, num_mz)

    # Add metadata to the Zarr store
    add_zarr_metadata(z, imzml_file, num_pixels, num_mz, unique_mz_values, pixel_coords)


    print(f"Zarr array shape: {z.shape}")
    print(f"Zarr array chunks: {z.chunks}")
    print(f"Zarr array dtype: {z.dtype}")
    print(f"Zarr array created successfully.")


def process_chunk(parser, mz_index_map, z, pbar, chunk_idx, start_idx, end_idx, num_mz):
    """
    Processes a chunk of pixels and writes the data to the Zarr array.
    
    Args:
        parser (ImzMLParser): The ImzMLParser object for parsing the imzML file.
        mz_index_map (dict): Map of m/z values to their corresponding indices.
        z (zarr.Array): Zarr array object to write the data to.
        pbar (tqdm): Progress bar object.
        chunk_idx (int): Index of the current chunk.
        start_idx (int): Starting index of the current chunk.
        end_idx (int): Ending index of the current chunk.
        num_mz (int): Total number of m/z values.
    """
    chunk_pixels = range(start_idx, end_idx)
    chunk_length = end_idx - start_idx

    # Initialize data structures for the current chunk
    coords = [[], []]  # Coordinates for non-zero values (row index within chunk, column index)
    data = []

    # Process each pixel in the current chunk
    for idx_in_chunk, idx in enumerate(tqdm(chunk_pixels, desc=f"Processing chunk {chunk_idx + 1}", leave=False)):
        mz_values, intensities = parser.getspectrum(idx)
        for mz, intensity in zip(mz_values, intensities):
            column_index = mz_index_map.get(mz, -1)
            if column_index >= 0:
                coords[0].append(idx_in_chunk)  # row index within chunk
                coords[1].append(column_index)  # column index
                data.append(intensity)

        # Update the main progress bar
        pbar.update(1)

    # Write sparse COO array to Zarr
    if data:
        write_sparse_array_to_zarr(coords, data, chunk_length, num_mz, z, start_idx)
    else:
        print(f"No data in chunk {chunk_idx}")


def write_sparse_array_to_zarr(coords, data, chunk_length, num_mz, z, start_idx):
    """
    Writes a sparse COO array to the Zarr array.
    
    Args:
        coords (list of lists): Coordinates for the non-zero values.
        data (list): Intensity values for the non-zero elements.
        chunk_length (int): Length of the chunk being processed.
        num_mz (int): Total number of m/z values.
        z (zarr.Array): Zarr array object to write the data to.
        start_idx (int): Starting index of the chunk in the overall array.
    """
    coords = np.array(coords)
    data = np.array(data)
    chunk_shape = (chunk_length, num_mz)
    coo_array = sparse.COO(coords, data, shape=chunk_shape)
    print(coo_array.shape)

    # Adjust row indices to absolute indices
    row_indices = coo_array.coords[0] + start_idx
    col_indices = coo_array.coords[1]

    # Write data to Zarr array
    z.set_coordinate_selection((row_indices, col_indices), coo_array.data)
    


def add_zarr_metadata(z, imzml_file, num_pixels, num_mz, unique_mz_values, pixel_coords):
    """
    Adds metadata to the Zarr store.
    
    Args:
        z (zarr.Array): Zarr array object.
        imzml_file (str): Path to the imzML file.
        num_pixels (int): Number of pixels.
        num_mz (int): Number of m/z values.
        unique_mz_values (numpy array): Array of unique m/z values.
        pixel_coords (numpy array): Array of pixel coordinates.
    """
    z.attrs['description'] = "Mass spectrometry imaging data in Zarr format"
    z.attrs['imzml_file'] = imzml_file
    z.attrs['num_pixels'] = num_pixels
    z.attrs['num_mz'] = num_mz
    z.attrs['mz_range'] = (unique_mz_values[0], unique_mz_values[-1]) if unique_mz_values.size > 0 else (None, None)
    z.attrs['pixel_coordinates'] = pixel_coords.tolist()  # Optionally improve this
    z.attrs['mass_axis'] = unique_mz_values.tolist()  # Optionally improve this


def main():
    imzml_file = r"C:\Users\tvisv\Downloads\ZarrConvert\tests\data\test_processed.imzML"
    zarr_store_path = r'C:\Users\tvisv\Downloads\test_processed.zarr'
    unique_mz_values, pixel_coords = collect_metadata(imzml_file)
    write_data_in_chunks(imzml_file, zarr_store_path, unique_mz_values, pixel_coords)
    


if __name__ == "__main__":
    main()
