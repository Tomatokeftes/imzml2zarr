import xarray as xr
import dask.array as da
import numpy as np

class ZarrWriter:
    def __init__(self, zarr_store_path, mass_axis, x_coords, y_coords, metadata):
        self.zarr_store_path = zarr_store_path
        self.mass_axis = mass_axis
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.num_x = len(x_coords)
        self.num_y = len(y_coords)
        self.num_mz = len(mass_axis)
        # Create mappings from coordinate values to indices
        self.x_coord_map = {coord: idx for idx, coord in enumerate(self.x_coords)}
        self.y_coord_map = {coord: idx for idx, coord in enumerate(self.y_coords)}


    def create_store(self):
        """
        Initialize the Zarr store with an empty dataset and predefined metadata.
        """

        # Create a dummy DataArray to initialize the Zarr store
        dummy_data = da.zeros(
            (self.num_x, self.num_y, self.num_mz), 
            chunks=(64, 64, min(50000, self.num_mz)),  # Chunk sizes to balance memory and speed
            dtype=np.float32
        )

        # Wrap in an xarray DataArray
        data = xr.DataArray(
            dummy_data,
            dims=('x_coordinate', 'y_coordinate', 'mz_channel'),
            coords={
                'x_coordinate': np.arange(self.num_x),
                'y_coordinate': np.arange(self.num_y),
                'mz_channel': self.mass_axis
            },
            name='intensity'
        )

        # Write metadata only to Zarr store
        data.to_zarr(self.zarr_store_path, mode='w', consolidated=True, compute=False, group='data')

    def write_block(self, block_ds):
        start_x_coord = block_ds.coords['x_coordinate'].values[0]
        start_y_coord = block_ds.coords['y_coordinate'].values[0]

        start_x_idx = self.x_coord_map[start_x_coord]
        start_y_idx = self.y_coord_map[start_y_coord]

        # Since mz_channel dimension matches, we can set start and end indices directly
        mz_start_idx = 0
        end_mz_idx = self.num_mz

        region = {
            "x_coordinate": slice(start_x_idx, start_x_idx + block_ds.sizes["x_coordinate"]),
            "y_coordinate": slice(start_y_idx, start_y_idx + block_ds.sizes["y_coordinate"]),
            "mz_channel": slice(mz_start_idx, end_mz_idx)
        }

        block_ds.to_zarr(
            self.zarr_store_path,
            mode='a',
            region=region,
            group='data'
        )
