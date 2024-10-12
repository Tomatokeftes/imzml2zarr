# import zarr
# import numpy as np
# import sparse
# from tqdm import tqdm
# from pyimzml.ImzMLParser import ImzMLParser  # Assuming you're using pyimzML to read the imzML files

# # First pass: Read imzML to collect metadata and structure information
# def collect_metadata(imzml_file):
#     parser = ImzMLParser(imzml_file)
    
#     # Store unique m/z values and pixel coordinates
#     unique_mz_values = set()
#     all_pixel_coords = set()
    
#     # For processed format, we track unique m/z values per pixel
#     for idx in tqdm(range(len(parser.coordinates)), desc="First Pass: Collecting Metadata"):
#         x, y, _ = parser.coordinates[idx]  # Ignoring z coordinate for 2D images
#         print(x, y)
#         mz_values, intensities = parser.getspectrum(idx)
#         print(mz_values)
#         print(intensities)
        
#         all_pixel_coords.add((x, y))
#         unique_mz_values.update(mz_values)
    
#     unique_mz_values = sorted(unique_mz_values)  # Sort m/z values for use as z-axis
#     num_mz = len(unique_mz_values)
    
#     # Find the maximum pixel coordinates
#     max_x = max(coord[0] for coord in all_pixel_coords)
#     max_y = max(coord[1] for coord in all_pixel_coords)

#     print(f"Number of unique m/z values: {num_mz}")
#     print(f"Maximum X coordinate: {max_x}")
#     print(f"Maximum Y coordinate: {max_y}")
    
#     return all_pixel_coords, unique_mz_values, num_mz, max_x, max_y

# # Second pass: Write to COO sparse array and then to Zarr
# def write_coo_to_zarr(imzml_file, zarr_store, unique_mz_values, max_x, max_y):
#     parser = ImzMLParser(imzml_file)
    
#     # Build an m/z index map to place m/z values in the right z-index
#     mz_index_map = {mz: i for i, mz in enumerate(unique_mz_values)}
    
#     coords = [[], []]  # Coordinates for non-zero values (pixel index, mz index)
#     data = []
    
#     # Second pass: read each pixel and store data in COO format
#     for idx in tqdm(range(len(parser.coordinates)), desc="Second Pass: Writing Data"):
#         x, y, _ = parser.coordinates[idx]
#         mz_values, intensities = parser.getspectrum(idx)
        
#         pixel_index = y * (max_x + 1) + x  # Calculate linear pixel index

#         for mz, intensity in zip(mz_values, intensities):
#             z_index = mz_index_map.get(mz, -1)
#             if z_index >= 0:
#                 coords[0].append(pixel_index)  # Pixel index
#                 coords[1].append(z_index)       # m/z index
#                 data.append(intensity)
    
#     shape = ((max_x + 1) * (max_y + 1), len(unique_mz_values))
    
#     # Step 1: Create the COO sparse array
#     data = np.array(data)
#     coo_array = sparse.COO(coords, data, shape=shape)
    
#     # Step 2: Create a Zarr array with matching shape and chunking
#     z = zarr.zeros(shape=coo_array.shape, chunks=(1, 1000), dtype=coo_array.dtype, compressor=None)
    
#     # Step 3: Store the non-zero values in the Zarr array using reconstructed coordinates
#     z.set_coordinate_selection(tuple(coo_array.coords), coo_array.data)

# # Example usage
# imzml_file = r"C:\Users\tvisv\OneDrive\Desktop\Taste of MSI\rsc Taste of MSI\Ingredient Classification MALDI\Original\20240605_pea_pos.imzML"
# zarr_store = r"C:\Users\tvisv\Downloads\20240605_pea_pos_2d.zarr"

# all_pixel_coords, unique_mz_values, num_mz, max_x, max_y = collect_metadata(imzml_file)
# write_coo_to_zarr(imzml_file, zarr_store, unique_mz_values, max_x, max_y)

import numpy as np
import sparse
import zarr
from tqdm import tqdm
from pyimzml.ImzMLParser import ImzMLParser

# First pass: Read imzML to collect metadata and structure information
def collect_metadata(imzml_file):
    parser = ImzMLParser(imzml_file)
    
    # Store unique m/z values and pixel coordinates
    unique_mz_values = set()
    
    # For processed format, we track unique m/z values per pixel
    for idx in tqdm(range(len(parser.coordinates)), desc="First Pass: Collecting Metadata"):
        mz_values, _ = parser.getspectrum(idx)
        unique_mz_values.update(mz_values)
    
    unique_mz_values = sorted(unique_mz_values)
    
    return unique_mz_values

# Second pass: Write to COO sparse array and then to Zarr

def write_coo_to_zarr(imzml_file, zarr_store_path, unique_mz_values):
    parser = ImzMLParser(imzml_file)
    
    # Build an m/z index map to place m/z values in the right z-index
    mz_index_map = {mz: i for i, mz in enumerate(unique_mz_values)}
    
    coords = [[], []]  # Coordinates for non-zero values (pixel index, mz index)
    data = []
    num_pixels = len(parser.coordinates)
    num_mz = len(unique_mz_values)
    shape = (num_pixels, num_mz)

    # Second pass: read each pixel and store data in COO format
    for idx in tqdm(range(len(parser.coordinates)), desc="Second Pass: Writing Data"):
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
    z = zarr.open(zarr_store_path, mode='w', shape=coo_array.shape, chunks=(10000, 1000), dtype=np.uint32)

    # Step 3: Store the non-zero values in the Zarr array using reconstructed coordinates
    z.set_coordinate_selection(tuple(coo_array.coords), coo_array.data)

    print(f"Zarr array shape: {z.shape}")
    print(f"Zarr array chunks: {z.chunks}")
    print(f"Zarr array dtype: {z.dtype}")
    print(f"Zarr array compressor: {z.compressor}")
    print(f"Zarr array path: {z.path}")
    print(f"Zarr array info: {z.info}")

unique_mz_values = collect_metadata(r"C:\Users\tvisv\OneDrive\Desktop\Taste of MSI\rsc Taste of MSI\Ingredient Classification MALDI\Original\20240605_pea_pos.imzML")
write_coo_to_zarr(r"C:\Users\tvisv\OneDrive\Desktop\Taste of MSI\rsc Taste of MSI\Ingredient Classification MALDI\Original\20240605_pea_pos.imzML", r"C:\Users\tvisv\Downloads\20240605_pea_pos.zarr", unique_mz_values)