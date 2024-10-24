import dask.array as da
import zarr
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
from dask.distributed import Client, LocalCluster
import dask.array as da
import zarr
import numpy as np
import matplotlib.pyplot as plt

def plot_total_ion_image(zarr_store_path, cmap="viridis"):
    # Open the Zarr store with Dask for lazy loading
    z = zarr.open_group(zarr_store_path, mode='r')
    
    # Load the intensity data lazily with Dask
    intensities = da.from_zarr(z['intensities'])
    
    # Sum the intensities across the m/z axis (axis=1) to get the total ion image
    total_ion_image = intensities.sum(axis=1)

    # Load the pixel coordinates (they are small, so can be loaded into memory)
    pixel_coords = z['pixel_coords'][:]

    # Extract x, y coordinates from the pixel coordinates
    x_coords = pixel_coords[:, 0]
    y_coords = pixel_coords[:, 1]

    # Create an empty image array (filled with NaN initially)
    max_x, max_y = np.max(x_coords), np.max(y_coords)
    ion_image = np.full((max_y + 1, max_x + 1), np.nan)

    # Compute the total ion image (this loads data lazily)
    total_intensities = total_ion_image.compute()

    # Place the intensity values at the correct (x, y) positions
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        ion_image[y, x] = total_intensities[i]

    # Plot the total ion image using matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(ion_image, cmap=cmap, origin="lower")
    plt.colorbar(label="Total Ion Intensity")
    plt.title("Total Ion Image")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()


if __name__ == '__main__':
    cluster = LocalCluster()
    client = Client(cluster)
    print(f"Dask dashboard link: {client.dashboard_link}")
    
    # Open the Zarr array with Dask using lazy loading
    zarr_store = zarr.open(r"C:\Users\tvisv\Downloads\test_processed.zarr", mode='r')
    dask_array = da.from_zarr(zarr_store)

    # Access metadata from the Zarr store
    mz_axis = zarr_store.attrs['mass_axis']
    num_mz = zarr_store.attrs['num_mz']
    num_pixels = zarr_store.attrs['num_pixels']

    print("Number of Pixels:", num_pixels)
    print("Number of MZ:", num_mz)
    print("MZ Range:", zarr_store.attrs['mz_range'])

    # Calculate the mean mass spectrum across all spatial pixels
    average_mass_spectrum = dask_array.mean(axis=0)  # Mean over spatial dimensions

    # # Use progress bar during computation
    # with ProgressBar():
    #     average_mass_spectrum_computed = average_mass_spectrum.compute()

    # # Plot using the mass axis for the x-axis
    # plt.plot(mz_axis, average_mass_spectrum_computed)
    # plt.xlabel('m/z')
    # plt.ylabel('Intensity')
    # plt.title('Average Mass Spectrum')
    # plt.show()

    total_ion_image = dask_array.sum(axis=1)
    print(total_ion_image.shape)

    # Load the pixel coordinates (they are small, so can be loaded into memory)
    pixel_coords = np.array(zarr_store.attrs['pixel_coordinates'])

    # Extract x, y coordinates from the pixel coordinates
    x_coords = pixel_coords[:, 0]
    y_coords = pixel_coords[:, 1]

    # Create an empty image array (filled with NaN initially)
    max_x, max_y = np.max(x_coords), np.max(y_coords)
    ion_image = np.full((max_y + 1, max_x + 1), np.nan)
    print(ion_image.shape)

    # Compute the total ion image (this loads data lazily)
    total_intensities = total_ion_image.compute()

    # Place the intensity values at the correct (x, y) positions
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        ion_image[y, x] = total_intensities[i]


    # Flip the image array vertically
    ion_image = np.flipud(ion_image)
    print(ion_image.shape)

    # Plot the total ion image using matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(ion_image, cmap='viridis', origin="lower")
    plt.colorbar(label="Total Ion Intensity")
    plt.title("Total Ion Image")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()
