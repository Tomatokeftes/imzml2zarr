# import numpy as np
# import xarray as xr

# # Set up the synthetic data dimensions and values
# # Example dimensions: y (14), x (26), mz_channel (4 unique m/z values)
# y_dim, x_dim = 14, 26
# mz_values = np.array([100, 200, 300, 400])  # Define 4 m/z channels
# intensity_values = {100: 50, 200: 60, 300: 70, 400: 90}  # Map each m/z to an intensity

# # Define the base intensity map, ensuring each region has the right m/z and intensity
# intensity_map = np.zeros((y_dim, x_dim, len(mz_values)), dtype=np.float32)


# # Define the base array structure with known regions
# array_multi_color = np.array([
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 4, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 3, 3, 3, 0, 0],
#     [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 0, 2, 2, 0, 0, 0, 3, 3, 3, 0, 0],
#     [0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 0, 3, 3, 3, 0, 0],
#     [0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 3, 3, 3, 0, 0],
#     [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 3, 3, 3, 0, 0],
#     [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 3, 3, 3, 0, 0],
#     [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 3, 3, 3, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# ])

# # Populate intensity_map based on `array_multi_color`
# label_to_mz = {1: 100, 2: 200, 3: 300, 4: 400}  # Mapping of labels to m/z
# for y in range(y_dim):
#     for x in range(x_dim):
#         label = array_multi_color[y, x]
#         if label in label_to_mz:
#             mz_idx = np.where(mz_values == label_to_mz[label])[0][0]
#             intensity_map[y, x, mz_idx] = intensity_values[label_to_mz[label]]

# # Create an xarray DataArray with coordinates matching imzML structure
# zarr_dataarray = xr.DataArray(
#     intensity_map,
#     dims=("y", "x", "mz_channel"),
#     coords={"y": np.arange(y_dim), "x": np.arange(x_dim), "mz_channel": mz_values},
#     name="data"
# )

# # Save DataArray to Zarr format
# zarr_dataarray.to_zarr("test_data.zarr")




import xarray as xr
import matplotlib.pyplot as plt

# Specify the full path to the 'data' group inside the Zarr store
data = xr.open_zarr(r"C:\Users\tvisv\Downloads\test_processed.zarr\data", consolidated=False)

# Calculate the Total Ion Image (TII) by summing over the mz_channel axis
total_ion_image = data["intensity"].sum(dim="mz_channel")

# Calculate the Average Mass Spectrum by averaging over the x and y dimensions
average_mass_spectrum = data["intensity"].mean(dim=["x_coordinate", "y_coordinate"])

# Plot the Total Ion Image (TII)
plt.figure(figsize=(8, 6))
total_ion_image.plot(cmap="viridis")
plt.title("Total Ion Image")
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.show()

# Plot the Average Mass Spectrum as a stem plot
plt.figure(figsize=(8, 6))
plt.stem(
    average_mass_spectrum["mz_channel"], 
    average_mass_spectrum, 
    linefmt='b-', 
    markerfmt='bo', 
    basefmt=" ",  # No baseline
)
plt.title("Average Mass Spectrum")
plt.xlabel("m/z")
plt.ylabel("Intensity")
plt.grid(True)
plt.show()


