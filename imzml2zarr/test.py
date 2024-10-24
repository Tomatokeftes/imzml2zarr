from iomsi import ProcessedImzMLParser
from zarr_writer import ZarrWriter


def main():
    imzml_file = r"C:\Users\tvisv\Downloads\ZarrConvert\tests\data\test_processed.imzML"
    zarr_store_path = r'C:\Users\tvisv\Downloads\test_processed.zarr'
    
    # Step 1: Parse imzML file
    imzml_parser = ProcessedImzMLParser(imzml_file)
    unique_mz_values, pixel_coords = imzml_parser.collect_metadata()

    # Step 2: Create Zarr writer and write data
    zarr_writer = ZarrWriter(zarr_store_path, unique_mz_values, pixel_coords)
    zarr_writer.write_data_in_chunks(imzml_parser)

if __name__ == "__main__":
    main()

# import xarray as xr
# import numpy as np
# import matplotlib.pyplot as plt

# # Sample dimensions
# x_coords = np.arange(0, 128)  # 128 pixels in x
# y_coords = np.arange(0, 128)  # 128 pixels in y
# mz_values = np.linspace(100, 1000, 500)  # 500 m/z values ranging from 100 to 1000

# # Create a 3D random data array to simulate intensities
# data = np.random.rand(128, 128, 500)  # Random intensities

# # Create the xarray DataArray
# da = xr.DataArray(data, dims=['x', 'y', 'mz'], coords={'x': x_coords, 'y': y_coords, 'mz': mz_values})

# # Plotting a slice of the data
# plt.figure(figsize=(10, 6))
# da.isel(mz=250).plot(cmap='viridis')  # Slice at the middle of the m/z dimension
# plt.title('Intensity Image at m/z = {:.2f}'.format(mz_values[250]))
# plt.show()
