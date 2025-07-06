import h5py
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from matplotlib import cm
from matplotlib.colors import ListedColormap
from scipy.interpolate import griddata

file_path = "prediction_2025-04-01T05.h5"

# Load data
with h5py.File(file_path, 'r') as f:
    lat = f['latitude'][:]
    lon = f['longitude'][:]
    pm25 = f['PM2.5'][:]  # <- Change to PM2.5

# Remove missing/zero values
valid_mask = (~np.isnan(pm25)) & (pm25 > 0)
lat_valid = lat[valid_mask]
lon_valid = lon[valid_mask]
pm25_valid = pm25[valid_mask]

# Define finer grid for interpolation
grid_lon = np.linspace(lon.min(), lon.max(), 400)
grid_lat = np.linspace(lat.min(), lat.max(), 400)
grid_lon2d, grid_lat2d = np.meshgrid(grid_lon, grid_lat)

# Interpolate to smooth surface
grid_pm25 = griddata(
    points=np.array([lon_valid, lat_valid]).T,
    values=pm25_valid,
    xi=(grid_lon2d, grid_lat2d),
    method='cubic'
)

masked_pm25 = np.ma.masked_where((grid_pm25 == 0) | np.isnan(grid_pm25), grid_pm25)

# Custom colormap with white at 0
base_cmap = cm.get_cmap('plasma', 256)
new_colors = base_cmap(np.linspace(0, 1, 256))
new_colors[0] = [1, 1, 1, 1]
custom_cmap = ListedColormap(new_colors)

# Plot
plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

plot = ax.pcolormesh(
    grid_lon2d, grid_lat2d, masked_pm25,
    cmap=custom_cmap,
    shading='auto',
    vmin=0,
    vmax=150,
    transform=ccrs.PlateCarree()
)

# Add map features
ax.add_feature(cfeature.BORDERS, linewidth=0.7)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
ax.set_extent([65, 100, 5, 40], crs=ccrs.PlateCarree())

# Cities
cities = {
    "Delhi": (77.1025, 28.7041),
    "Mumbai": (72.8777, 19.0760),
    "Chennai": (80.2707, 13.0827),
    "Kolkata": (88.3639, 22.5726)
}
for name, (lon_c, lat_c) in cities.items():
    ax.plot(lon_c, lat_c, 'ro', markersize=4)
    ax.text(lon_c + 0.5, lat_c + 0.5, name, fontsize=8)

# Colorbar
cbar = plt.colorbar(plot, orientation='vertical', pad=0.05, aspect=30)
cbar.set_label('Predicted PM2.5 (µg/m³)')

# Save
title = os.path.basename(file_path).replace('.h5', '')
plt.title(f'{title} | PM2.5 Concentration Map')
plt.tight_layout()
plt.savefig(f'{title}_pm25_map.png')
plt.show()
