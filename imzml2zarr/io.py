import numpy as np
from pyimzml.ImzMLParser import ImzMLParser
import os
import tempfile
from tqdm import tqdm
import zarr

class FileParser:
    def __init__(self, file_path):
        self.file_path = file_path

    def collect_data(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def collect_metadata(self):
        raise NotImplementedError("Subclasses must implement this method.")
    

class ImzMLParser(FileParser):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.parser = ImzMLParser(file_path)
        self.num_pixels = len(self.parser.coordinates)
    
    def collect_metadata(self):
        unique_mz_values = set()
        pixel_coords = []
        for idx in range(self.num_pixels):
            mz_values, _ = self.parser.getspectrum(idx)
            unique_mz_values.update(mz_values)
            pixel_coords.append(self.parser.coordinates[idx])
        
        return np.array(sorted(unique_mz_values)), np.array(pixel_coords)
    
    def read_spectrum(self, idx):
        return self.parser.getspectrum(idx)
    

    