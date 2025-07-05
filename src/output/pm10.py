import h5py
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from matplotlib import cm
from matplotlib.colors import ListedColormap

file_path = "prediction_2025-04-01T05.h5"

# Load data
with h5py.File(file_path, 'r') as f:
    lat = f['latitude'][:]
    lon = f['longitude'][:]
    pm10 = f['PM10'][:]  # Change from PM2.5 to PM10

# --- Grid reconstruction ---
if len(lat) == len(lon) == len(pm10):
    unique_lat = np.unique(lat)
    unique_lon = np.unique(lon)
    lat2d, lon2d = np.meshgrid(unique_lat, unique_lon, indexing='ij')
    pm10_2d = np.full(lat2d.shape, np.nan)

    for i in range(len(pm10)):
        lat_idx = np.where(unique_lat == lat[i])[0][0]
        lon_idx = np.where(unique_lon == lon[i])[0][0]
        pm10_2d[lat_idx, lon_idx] = pm10[i]
else:
    raise ValueError("Latitude, longitude, and PM10 shapes do not match.")

# --- Mask NaNs and 0s ---
masked_pm10 = np.ma.masked_where((pm10_2d == 0) | np.isnan(pm10_2d), pm10_2d)

# --- Custom colormap where 0 is white ---
base_cmap = cm.get_cmap('plasma', 256)
new_colors = base_cmap(np.linspace(0, 1, 256))
new_colors[0] = [1, 1, 1, 1]  # White for lowest value
custom_cmap = ListedColormap(new_colors)

# --- Plot with smoother rendering ---
plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

plot = ax.imshow(
    masked_pm10,
    extent=(lon2d.min(), lon2d.max(), lat2d.min(), lat2d.max()),
    origin='lower',
    transform=ccrs.PlateCarree(),
    cmap=custom_cmap,
    interpolation='bicubic',  # Smooth the pixel grid
    vmin=0,
    vmax=150
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
cbar.set_label('Predicted PM10 (µg/m³)')

# Save
title = os.path.basename(file_path).replace('.h5', '')
plt.title(f'{title} | PM10 Concentration Map')
plt.tight_layout()
plt.savefig(f'{title}_pm10_map.png')
plt.show()
