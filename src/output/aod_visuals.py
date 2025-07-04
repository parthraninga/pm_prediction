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

# --- File path ---
file_path = 'prediction_2025-04-01T05.h5'

# --- Load data ---
with h5py.File(file_path, 'r') as f:
    aod = f['PM2.5'][0, :, :]
    lat = f['latitude'][:]
    lon = f['longitude'][:]

lat2d, lon2d = lat, lon

# --- Custom colormap ---
plasma = plt.colormaps['plasma']
colors = plasma(np.linspace(0, 1, 256))
colors[0] = [1, 1, 1, 1]
white_plasma = mcolors.ListedColormap(colors)

# --- Parse filename for title and image name ---
filename = os.path.basename(file_path)
parts = filename.split('_')
date_str = parts[1]
time_str = parts[2]
dt = datetime.strptime(date_str + time_str, '%d%b%Y%H%M')
plot_title = f'INSAT-3DR AOD on {dt.strftime("%d %B %Y, %H:%M UTC")}'
image_filename = f"AOD_{dt.strftime('%Y%m%d_%H%M')}.png"

# --- Output directory ---
output_dir = 'AOD_Plots'
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, image_filename)

# --- Plot ---
plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
plot = ax.pcolormesh(lon2d, lat2d, aod, cmap=white_plasma, vmin=0, vmax=1.5, shading='auto')
ax.add_feature(cfeature.BORDERS, linewidth=1)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
ax.set_extent([65, 100, 5, 40], crs=ccrs.PlateCarree())
cbar = plt.colorbar(plot, orientation='vertical', pad=0.05, aspect=30)
cbar.set_label('AOD @ 650 nm')
plt.title(plot_title)

# --- Save and show ---
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
