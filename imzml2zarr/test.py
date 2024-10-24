from iomsi import ProcessedImzMLParser
from zarr_writer import ZarrWriter


def main():
    imzml_file_path = r"C:\Users\tvisv\OneDrive\Desktop\Taste of MSI\rsc Taste of MSI\Ingredient Classification MALDI\Original\20240605_pea_pos.imzML"
    zarr_store_path = r'C:\Users\tvisv\Downloads\test_processed.zarr'
    
    # Initialize the parser
    parser = ProcessedImzMLParser(file_path=imzml_file_path)

    # Collect metadata (unique m/z values and pixel coordinates)
    unique_mz_values, pixel_coords = parser.collect_metadata()

    # Initialize the Zarr writer
    writer = ZarrWriter(zarr_store_path=zarr_store_path, unique_mz_values=unique_mz_values, pixel_coords=pixel_coords)

    # Write the data to Zarr in chunks
    chunk_size = 1000  # Adjust the chunk size based on memory capacity and performance needs
    writer.write_data_in_chunks(parser=parser, chunk_size=chunk_size)

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
