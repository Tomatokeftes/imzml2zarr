import numpy as np
import sparse
import zarr
from tqdm import tqdm
from pyimzml.ImzMLParser import ImzMLParser
import zarr
print("Zarr version:", zarr.__version__)


# First pass: Read imzML to collect metadata and structure information
def collect_metadata(imzml_file):
    parser = ImzMLParser(imzml_file)
    
    # Store unique m/z values and pixel coordinates
    unique_mz_values = set()
    pixel_coords = parser.coordinates
    
    # For processed format, we track unique m/z values per pixel
    for idx in tqdm(range(len(pixel_coords)), desc="First Pass: Collecting Metadata"):
        x, y, z = parser.coordinates[idx]
        mz_values, _ = parser.getspectrum(idx)
        unique_mz_values.update(mz_values)
    
    unique_mz_values = sorted(unique_mz_values)
    
    return unique_mz_values, pixel_coords

# Second pass: Write to COO sparse array and then to Zarr

def write_coo_to_zarr(imzml_file, zarr_store_path, unique_mz_values, pixel_coords):
    parser = ImzMLParser(imzml_file)
    
    # Build an m/z index map to place m/z values in the right z-index
    mz_index_map = {mz: i for i, mz in enumerate(unique_mz_values)}
    
    coords = [[], []]  # Coordinates for non-zero values (pixel index, mz index)
    data = []
    num_pixels = len(pixel_coords)
    num_mz = len(unique_mz_values)
    shape = (num_pixels, num_mz)
    # Modify the calculation of min and max coordinates
    max_x = max(pixel_coords, key=lambda coord: coord[0])[0]
    max_y = max(pixel_coords, key=lambda coord: coord[1])[1]
    max_z = max(pixel_coords, key=lambda coord: coord[2])[2]

    min_x = min(pixel_coords, key=lambda coord: coord[0])[0]
    min_y = min(pixel_coords, key=lambda coord: coord[1])[1]
    min_z = min(pixel_coords, key=lambda coord: coord[2])[2]


    # Second pass: read each pixel and store data in COO format
    for idx in tqdm(range(len(pixel_coords)), desc="Second Pass: Writing Data"):
        mz_values, intensities = parser.getspectrum(idx)
        
        for mz, intensity in zip(mz_values, intensities):
            column_index = mz_index_map.get(mz, -1)  # Column index representing the m/z value
            if column_index >= 0:
                coords[0].append(idx)  # Pixel index (row)
                coords[1].append(column_index)  # m/z index (column)
                data.append(intensity)
            else:
                print(f"Skipping m/z value {mz} not found in the unique m/z values.")
    
    # Step 1: Create the COO sparse array
    data = np.array(data)
    coo_array = sparse.COO(coords, data, shape=shape)
    
    # Step 2: Create a Zarr array with matching shape and chunking and save it to the working directory
    z = zarr.open(zarr_store_path, mode='w', shape=coo_array.shape, chunks=(1000, 10000), dtype=np.uint32)

    # Step 3: Store the non-zero values in the Zarr array using reconstructed coordinates
    z.set_coordinate_selection(tuple(coo_array.coords), coo_array.data)

    # Step 4: Add metadata to the Zarr store
    # Global metadata
    z.attrs['description'] = "Mass spectrometry imaging data in Zarr format"
    z.attrs['imzml_file'] = imzml_file
    z.attrs['num_pixels'] = num_pixels
    z.attrs['num_mz'] = num_mz
    z.attrs['mz_range'] = (unique_mz_values[0], unique_mz_values[-1]) if unique_mz_values else (None, None)
    z.attrs['pixel_coordinates'] = pixel_coords
    z.attrs['mass_axis'] = unique_mz_values
    z.attrs['min_coords'] = (min_x, min_y, min_z)
    z.attrs['max_coords'] = (max_x, max_y, max_z)
    print(f"Zarr array shape: {z.shape}")
    print(f"Zarr array chunks: {z.chunks}")
    print(f"Zarr array dtype: {z.dtype}")
    print(f"Zarr array compressor: {z.compressor}")
    print(f"Zarr array path: {z.store}")
    print("Zarr array created successfully.")

# Example usage
unique_mz_values, pixel_coords = collect_metadata(r"C:\Users\tvisv\OneDrive\Desktop\Taste of MSI\rsc Taste of MSI\Ingredient Classification MALDI\Original\20240605_pea_pos.imzML")
write_coo_to_zarr(r"C:\Users\tvisv\OneDrive\Desktop\Taste of MSI\rsc Taste of MSI\Ingredient Classification MALDI\Original\20240605_pea_pos.imzML", r"C:\Users\tvisv\Downloads\20240605_pea_pos.zarr", unique_mz_values, pixel_coords)
