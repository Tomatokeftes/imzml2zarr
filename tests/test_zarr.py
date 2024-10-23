import pytest
import numpy as np
import zarr
from unittest.mock import MagicMock
from imzml2zarr.zarr_writer import ZarrWriter  # Assuming your ZarrWriter is in zarr_writer.py


@pytest.fixture
def mock_parser():
    # This will mock the behavior of the file parser (e.g., ImzMLParser)
    mock = MagicMock()
    
    # Define mock m/z and intensity values for 3 pixels
    mock.read_spectrum.side_effect = [
        ([100, 200], [10, 20]),  # Pixel 1 data
        ([150, 250], [15, 25]),  # Pixel 2 data
        ([100, 300], [12, 30])   # Pixel 3 data
    ]
    
    return mock


@pytest.fixture
def zarr_writer(tmp_path):
    unique_mz_values = [100, 150, 200, 250, 300]
    pixel_coords = [(1, 2), (3, 4), (5, 6)]
    zarr_store_path = tmp_path / "test.zarr"
    return ZarrWriter(str(zarr_store_path), unique_mz_values, pixel_coords)


def test_create_zarr_store(zarr_writer):
    # Create the Zarr store
    z = zarr_writer.create_zarr_store(chunks=(1000, 1000))

    # Test that the Zarr store is created correctly
    # z is a group, not an array, so we check group attributes
    assert isinstance(z, zarr.hierarchy.Group)

    # Assert that 'data' and 'metadata' groups exist
    assert 'data' in z
    assert 'metadata' in z


def test_write_chunks(zarr_writer, mock_parser):
    # Write data into the Zarr store using the provided function
    zarr_writer.write_chunks(mock_parser, chunk_size=2)

    # Re-open the Zarr store to ensure metadata and data are read from disk
    z = zarr.open_group(zarr_writer.zarr_store_path, mode='r')

    # Ensure the data has been written correctly to the 'data' group
    expected_data = np.array([
        [10, 0, 20, 0, 0],  # Pixel 1
        [0, 15, 0, 25, 0],  # Pixel 2
        [12, 0, 0, 0, 30]   # Pixel 3
    ])
    
    # Assuming the data array is named 'data_array' inside the 'data' group
    np.testing.assert_array_equal(z['data/data_array'][:, :], expected_data)

    # Test that the metadata has been written correctly to the 'metadata' group
    # metadata_group = z['metadata']
    # assert metadata_group.attrs['mass_axis'] == [100, 150, 200, 250, 300]
    # Uncomment if 'pixel_coordinates' are also saved in the metadata
    # assert metadata_group.attrs['pixel_coordinates'] == [(1, 2), (3, 4), (5, 6)]
