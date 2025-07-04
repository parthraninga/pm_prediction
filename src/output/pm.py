#To get visuals of a specific file
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import cartopy.crs as ccrs
from datetime import datetime
import os
import cartopy.feature as cfeature

file_path = 'prediction_2025-04-01T05.h5'

with h5py.File(file_path, 'r') as f:
    # Explore the keys
    print("Available datasets:")
    def print_keys(name):
        print(name)
    f.visit(print_keys)

    # Example: Read AOD and coordinates (check exact key names from output)
    pm = f['/PM2.5'][:]  # replace with exact path like '/Geophysical Data/AOD'
    lat = f['/latitude'][:]
    lon = f['/longitude'][:]