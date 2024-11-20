import pytest
import zarr
from pathlib import Path
from your_module.convertor import convert_to_store, _get_xarray_axes

@pytest.fixture
def continuous_imzml_files():
    """Fixture pointing to a real continuous imzML file."""
    # Replace with the actual path to your continuous sample files
    imzml_path = Path("tests/data/sample_continuous.imzML")
    ibd_path = Path("tests/data/sample_continuous.ibd")
    
    if not imzml_path.exists() or not ibd_path.exists():
        pytest.skip("Continuous imzML sample files not found.")
    
    return imzml_path, ibd_path

@pytest.fixture
def processed_imzml_files():
    """Fixture pointing to a real processed imzML file."""
    # Replace with the actual path to your processed sample files
    imzml_path = Path("tests/data/sample_processed.imzML")
    ibd_path = Path("tests/data/sample_processed.ibd")
    
    if not imzml_path.exists() or not ibd_path.exists():
        pytest.skip("Processed imzML sample files not found.")
    
    return imzml_path, ibd_path

def test_convert_continuous_to_zarr(continuous_imzml_files, tmp_path):
    """Test conversion of a real continuous imzML file to Zarr format."""
    imzml_path, ibd_path = continuous_imzml_files
    zarr_dest = tmp_path / "continuous_output.zarr"
    
    # Perform the conversion
    convert_to_store("test_continuous", imzml_path.parent, zarr_dest)

    # Open Zarr store and check structure
    zarr_root = zarr.open(zarr_dest, mode="r")
    assert "0" in zarr_root
    assert "labels/mzs/0" in zarr_root

    # Check metadata and shapes
    assert zarr_root.attrs["multiscales"][0]["name"] == "test_continuous"
    assert _get_xarray_axes(zarr_root) == ['c', 'z', 'y', 'x']
    
    # Additional checks: shape validation, content verification, etc.
    # assert zarr_root["0"].shape == expected_shape

def test_convert_processed_to_zarr(processed_imzml_files, tmp_path):
    """Test conversion of a real processed imzML file to Zarr format."""
    imzml_path, ibd_path = processed_imzml_files
    zarr_dest = tmp_path / "processed_output.zarr"
    
    # Perform the conversion
    convert_to_store("test_processed", imzml_path.parent, zarr_dest)

    # Open Zarr store and check structure
    zarr_root = zarr.open(zarr_dest, mode="r")
    assert "0" in zarr_root
    assert "labels/mzs/0" in zarr_root
    assert "labels/lengths/0" in zarr_root

    # Check metadata and shapes
    assert zarr_root.attrs["multiscales"][0]["name"] == "test_processed"
    assert _get_xarray_axes(zarr_root) == ['c', 'z', 'y', 'x']
    
    # Additional checks: shape validation, content verification, etc.
    # assert zarr_root["0"].shape == expected_shape
