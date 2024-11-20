from iomsi import ProcessedImzMLParser
from zarr_writer import ZarrWriter
from tqdm import tqdm

def main():
    imzml_file_path = r"C:\Users\tvisv\Downloads\ZarrConvert\tests\data\test_processed.imzML"
    zarr_store_path = r'C:\Users\tvisv\Downloads\pea_processed.zarr'
    
    # Initialize the parser
    parser = ProcessedImzMLParser(file_path=imzml_file_path)

    # Collect metadata (unique m/z values and pixel coordinates)
    unique_mz_values, pixel_coords = parser.collect_metadata()
    print("Metadata collected successfully.")
    # Extract x and y coordinates separately from pixel_coords
    x_coords = sorted(set(coord[0] for coord in pixel_coords))
    y_coords = sorted(set(coord[1] for coord in pixel_coords))
    metadata = {"description": "Processed imzML data converted to Zarr format"}

    # Initialize the Zarr writer with separate x and y coordinates
    writer = ZarrWriter(zarr_store_path=zarr_store_path, mass_axis=unique_mz_values, x_coords=x_coords, y_coords=y_coords, metadata=metadata)
    writer.create_store()
    print("Zarr store created successfully.")
    # Process data in chunks and write each chunk to the Zarr store
    for block_ds in parser.collect_data():
        writer.write_block(block_ds)


    print(f"DataArray saved successfully at {zarr_store_path}.")

if __name__ == "__main__":
    main()
