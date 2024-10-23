import numpy as np
import os
import tempfile
from tqdm import tqdm
from pyimzml.ImzMLParser import ImzMLParser
import zarr
import sparse

def collect_metadata(imzml_file):
    parser = ImzMLParser(imzml_file)
    num_pixels = len(parser.coordinates)

    # Create temporary files for m/z values
    temp_mz_file = tempfile.NamedTemporaryFile(delete=False)
    temp_unique_mz_file = tempfile.NamedTemporaryFile(delete=False)

    # Write m/z values to temporary file
    for idx in tqdm(range(num_pixels), desc="First Pass: Collecting Metadata"):
        mz_values, _ = parser.getspectrum(idx)
        mz_values.tofile(temp_mz_file)

    temp_mz_file.close()

    # Deduplicate m/z values using hashes
    deduplicate_mz_values_with_hashes(temp_mz_file.name, temp_unique_mz_file.name)

    # Close the unique m/z file before reading it and deleting it
    temp_unique_mz_file.close()

    # Read unique m/z values from the deduplicated file
    unique_mz_values = np.fromfile(temp_unique_mz_file.name, dtype=np.float64)

    unique_mz_values = np.sort(unique_mz_values)

    # Collect pixel coordinates
    pixel_coords = np.array(parser.coordinates)

    # Clean up temporary files
    os.remove(temp_mz_file.name)
    os.remove(temp_unique_mz_file.name)

    # Return unique m/z values and pixel coordinates (array)
    return unique_mz_values, pixel_coords



def deduplicate_mz_values_with_hashes(input_filename, output_filename):

    chunk_size = 1_000_000  # Adjust based on available memory
    hash_set = set()

    with open(input_filename, 'rb') as f_in, open(output_filename, 'wb') as f_out:
        while True:
            mz_values = np.fromfile(f_in, dtype=np.float64, count=chunk_size)
            if mz_values.size == 0:
                break
            unique_mz_values = []
            for mz in mz_values:
                h = hash(mz)
                if h not in hash_set:
                    hash_set.add(h)
                    unique_mz_values.append(mz)
            # Write unique m/z values to output file
            if unique_mz_values:
                np.array(unique_mz_values, dtype=np.float64).tofile(f_out)



def write_data_in_chunks(imzml_file, zarr_store_path, unique_mz_values, pixel_coords):
    parser = ImzMLParser(imzml_file)
    mz_index_map = {mz: i for i, mz in enumerate(unique_mz_values)}
    num_pixels = len(pixel_coords)  # Now pixel_coords is an array
    num_mz = len(unique_mz_values)
    shape = (num_pixels, num_mz)

    # Create Zarr array with desired chunk sizes
    z = zarr.open(zarr_store_path, mode='w', shape=shape, chunks=(16384, 5000), dtype=np.uint32)

    # Process data in chunks
    chunk_size = z.chunks[0]  # Use the row chunk size from Zarr array
    num_chunks = (num_pixels + chunk_size - 1) // chunk_size

    # Create the main progress bar for the entire dataset
    with tqdm(total=num_pixels, desc="Overall Progress", unit="pixel") as pbar:
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, num_pixels)
            chunk_pixels = range(start_idx, end_idx)
            chunk_length = end_idx - start_idx

            # Initialize data structures for the current chunk
            coords = [[], []]  # Coordinates for non-zero values (row index within chunk, column index)
            data = []

            # Process each pixel in the current chunk
            for idx_in_chunk, idx in enumerate(tqdm(chunk_pixels, desc=f"Processing chunk {chunk_idx + 1}/{num_chunks}", leave=False)):
                mz_values, intensities = parser.getspectrum(idx)
                for mz, intensity in zip(mz_values, intensities):
                    column_index = mz_index_map.get(mz, -1)
                    if column_index >= 0:
                        coords[0].append(idx_in_chunk)  # row index within chunk
                        coords[1].append(column_index)  # column index
                        data.append(intensity)

                # Update the main progress bar for each pixel processed
                pbar.update(1)

            # Create sparse COO array for the chunk
            if data:
                data = np.array(data)
                coords = np.array(coords)
                chunk_shape = (chunk_length, num_mz)
                coo_array = sparse.COO(coords, data, shape=chunk_shape)

                # Adjust row indices to absolute indices
                row_indices = coo_array.coords[0] + start_idx
                col_indices = coo_array.coords[1]

                # Write data to Zarr array
                z.set_coordinate_selection((row_indices, col_indices), coo_array.data)
            else:
                print(f"No data in chunk {chunk_idx}")

    # Add metadata to the Zarr store
    z.attrs['description'] = "Mass spectrometry imaging data in Zarr format"
    z.attrs['imzml_file'] = imzml_file
    z.attrs['num_pixels'] = num_pixels
    z.attrs['num_mz'] = num_mz
    z.attrs['mz_range'] = (unique_mz_values[0], unique_mz_values[-1]) if unique_mz_values.size > 0 else (None, None)
    z.attrs['pixel_coordinates'] = pixel_coords.tolist() # Find a better way to save these values
    z.attrs['mass_axis'] = unique_mz_values.tolist() # Find a better way to save these values
    print(f"Zarr array shape: {z.shape}")
    print(f"Zarr array chunks: {z.chunks}")
    print(f"Zarr array dtype: {z.dtype}")
    print(f"Zarr array created successfully.")


def main():
    imzml_file = 'path_to_imzml_file'
    zarr_store_path = 'path_to_zarr_store'
    unique_mz_values, pixel_coords = collect_metadata(imzml_file)
    write_data_in_chunks(imzml_file, zarr_store_path, unique_mz_values, pixel_coords)

if __name__ == "__main__":
    main()
