# import numpy as np
# from pyimzml.ImzMLParser import ImzMLParser

# def analyze_imzml_file(file_path):
#     # Step 1: Load the imzML file
#     parser = ImzMLParser(file_path)
    
#     # Step 2: Extract coordinates and count pixels
#     total_pixels = 0
#     empty_pixels = 0
#     pixel_coordinates = []
#     missing_pixels = 0
    
#     # Extract all pixel coordinates and analyze the spectra
#     for i, (x, y, z) in enumerate(parser.coordinates):
#         total_pixels += 1
#         mzs, intensities = parser.getspectrum(i)
        
#         if len(intensities) == 0 or np.all(intensities == 0):
#             empty_pixels += 1
            
#         # Collect the coordinates for missing pixel analysis
#         pixel_coordinates.append((x, y, z))
    
#     # Convert to a numpy array for easier missing pixel analysis
#     pixel_coordinates = np.array(pixel_coordinates)
    
#     # Step 3: Check for missing coordinates (assuming z=1 is constant)
#     max_x = np.max(pixel_coordinates[:, 0])
#     max_y = np.max(pixel_coordinates[:, 1])
    
#     # Create a set of all valid coordinates within the range
#     full_coordinate_set = set((x, y) for x in range(1, max_x+1) for y in range(1, max_y+1))
#     observed_coordinate_set = set((x, y) for x, y, z in pixel_coordinates)
    
#     # Missing coordinates
#     missing_coordinates = full_coordinate_set - observed_coordinate_set
#     missing_pixels = len(missing_coordinates)
    
#     # Return the analysis results
#     return {
#         "total_pixels": total_pixels,
#         "empty_pixels": empty_pixels,
#         "missing_pixels": missing_pixels,
#         "missing_coordinates": missing_coordinates
#     }

# # Example usage
# file_path = r"C:\Users\tvisv\Downloads\Eva Data\20240826_xenium_0041899.imzML"
# results = analyze_imzml_file(file_path)
# print("Total Pixels:", results["total_pixels"])
# print("Empty Pixels:", results["empty_pixels"])
# print("Missing Pixels:", results["missing_pixels"])
# print("Missing Coordinates:", results["missing_coordinates"])

# import dask.array as da
# import zarr

# def check_empty_pixels(zarr_file_path):
#     # Step 1: Load the Zarr store lazily using Dask
#     zarr_store = zarr.open(zarr_file_path, mode='r')
    
#     # Assuming your Zarr store has a group for the mass spectrometry data in 2D (rows: pixels, columns: m/z values)
#     # For example, let's say the dataset is stored under the 'mass_spectrum/data' key
#     mz_data = da.from_zarr(zarr_store)

#     # Step 2: Check for completely empty pixels (rows)
#     # We check if all values in each row are zero
#     empty_pixels = da.all(mz_data == 0, axis=1)

#     # Step 3: Compute the number of empty pixels lazily
#     total_empty_pixels = empty_pixels.sum().compute()  # Use compute() to evaluate lazily

#     # Step 4: Optionally, get the indices of empty pixels if needed
#     empty_pixel_indices = da.where(empty_pixels)[0].compute()

#     return {
#         "total_empty_pixels": total_empty_pixels,
#         "empty_pixel_indices": empty_pixel_indices
#     }

# # Example usage
# zarr_file_path = r"C:\Users\tvisv\Downloads\Eva Data\20240826_xenium_0041899.zarr"
# empty_pixel_results = check_empty_pixels(zarr_file_path)
# print("Total Empty Pixels:", empty_pixel_results["total_empty_pixels"])
# print("Indices of Empty Pixels:", empty_pixel_results["empty_pixel_indices"])

import dask.array as da
import zarr

def count_zero_only_rows(zarr_file_path):
    # Step 1: Load the Zarr store lazily using Dask
    zarr_store = zarr.open(zarr_file_path, mode='r')

    # Assuming your Zarr store has a group for the mass spectrometry data in 2D (rows: pixels, columns: m/z values)
    # For example, let's assume the dataset is stored under the 'mass_spectrum/data' key
    mz_data = da.from_zarr(zarr_store)

    # Step 2: Check for rows (pixels) where all m/z values are zero
    zero_only_rows = da.all(mz_data == 0, axis=1)  # This checks each row for all zeros

    # Step 3: Compute the total number of zero-only rows lazily
    total_zero_only_rows = zero_only_rows.sum().compute()

    # Step 4: Optionally, print the indices of zero-only rows if needed
    zero_only_row_indices = da.where(zero_only_rows)[0].compute()

    return {
        "total_zero_only_rows": total_zero_only_rows,
        "zero_only_row_indices": zero_only_row_indices
    }


# Example usage
zarr_file_path = r"C:\Users\tvisv\Downloads\Eva Data\20240826_xenium_0041899.zarr"
zero_only_results = count_zero_only_rows(zarr_file_path)

print(f"Total Zero-Only Rows: {zero_only_results['total_zero_only_rows']}")
print(f"Indices of Zero-Only Rows: {zero_only_results['zero_only_row_indices']}")
