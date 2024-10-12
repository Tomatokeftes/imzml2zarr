import zarr

# Open the Zarr array in read mode
z = zarr.open(r"C:\Users\tvisv\Downloads\20240605_pea_pos.zarr", mode='r')

# Access and print global metadata
print("Description:", z.attrs['description'])
print("IMZML File:", z.attrs['imzml_file'])
print("Number of Pixels:", z.attrs['num_pixels'])
print("Number of MZ:", z.attrs['num_mz'])
print("MZ Range:", z.attrs['mz_range'])
print("Pixel Coordinates:", z.attrs['pixel_coordinates'])
print("Mass Axis:", z.attrs['mass_axis'])
print("Minimum Coordinates:", z.attrs['min_coords'])
print("Maximum Coordinates:", z.attrs['max_coords'])
print("Zarr Array Shape:", z.shape)
print("Zarr Array Chunks:", z.chunks)
print("Zarr Array Data Type:", z.dtype)
print("Zarr Array Compressor:", z.compressor)
print("Zarr Array Path:", z.store)

