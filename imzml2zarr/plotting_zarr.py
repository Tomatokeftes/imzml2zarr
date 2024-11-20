import xarray as xr
import dask.array as da
import matplotlib.pyplot as plt

def verify_output(zarr_store_path):
    # Open the Zarr store as a Dataset using xarray
    dataset = xr.open_zarr(zarr_store_path, chunks={'pixel': 1000, 'mz': 1000})

    # Access the 'intensity' DataArray within the Dataset
    intensity_data = dataset['intensity']

    # Compute Total Ion Image (sum over m/z axis)
    total_ion_image = intensity_data.sum(dim='mz').compute()

    # Compute Total Mass Spectrum (sum over pixel axis)
    total_mass_spectrum = intensity_data.sum(dim='pixel').compute()

    # Plot the Total Ion Image
    plt.figure(figsize=(8, 6))
    plt.imshow(total_ion_image.values.reshape((-1, 1)), aspect='auto', cmap='hot')
    plt.colorbar(label='Intensity')
    plt.title('Total Ion Image')
    plt.xlabel('Pixel')
    plt.ylabel('Intensity')
    plt.show()

    # Plot the Total Mass Spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(dataset['mz'], total_mass_spectrum.values)
    plt.xlabel('m/z')
    plt.ylabel('Intensity')
    plt.title('Total Mass Spectrum')
    plt.show()

if __name__ == "__main__":
    zarr_store_path = r"C:\Users\tvisv\Downloads\test_processed.zarr"
    verify_output(zarr_store_path)
