import numpy as np
import sparse
import zarr

# Non-zero data values
data = np.array([10, 20, 30, 40, 50, 60])

# Indices of the non-zero values along the uncompressed axis (columns in this case)
indices = np.array([[0, 2, 1, 3, 1, 0],  # y-coordinates (2nd dimension)
                    [1, 4, 3, 2, 0, 2]])  # z-coordinates (3rd dimension)

# Index pointer array (indptr) for the compressed axis (x-axis in this case)
indptr = np.array([0, 2, 4, 6])  # First 2 non-zeros are in x=0, next 2 in x=1, last 2 in x=2

# Shape of the 3D array (4 x 3 x 5) -> (x, y, z)
shape = (4, 4, 5)

# Step 1: Create the GCXS sparse array
gcxs_array = sparse.GCXS((data, indices, indptr), shape=shape)

# Step 2: Reconstruct the coordinates for each axis
x_coords = []
y_coords = gcxs_array.indices[0]  # y-coordinates
z_coords = gcxs_array.indices[1]  # z-coordinates

# Reconstruct x-coordinates from indptr
for x_idx in range(len(gcxs_array.indptr) - 1):
    start = gcxs_array.indptr[x_idx]
    end = gcxs_array.indptr[x_idx + 1]
    x_coords.extend([x_idx] * (end - start))

# Convert to numpy arrays for use in set_coordinate_selection
x_coords = np.array(x_coords)
y_coords = np.array(y_coords)
z_coords = np.array(z_coords)

# Debugging: Print the coordinates and check bounds
print("Shape of Zarr array:", gcxs_array.shape)
print("x_coords:", x_coords)
print("y_coords:", y_coords)
print("z_coords:", z_coords)

# Check bounds for x, y, z coordinates
assert np.all(x_coords < shape[0]), f"x_coords out of bounds: {x_coords}"
assert np.all(y_coords < shape[1]), f"y_coords out of bounds: {y_coords}"
assert np.all(z_coords < shape[2]), f"z_coords out of bounds: {z_coords}"

# Step 3: Create a Zarr array with matching shape and chunking
z = zarr.zeros(shape=gcxs_array.shape, chunks=(1, 1, 1), dtype=gcxs_array.dtype, compressor=None)

# Step 4: Store the non-zero values in the Zarr array using reconstructed coordinates
z.set_coordinate_selection((x_coords, y_coords, z_coords), gcxs_array.data)

# Step 5: Verify the result by printing the full array from Zarr
print(z[...])
