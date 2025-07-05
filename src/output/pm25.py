import h5py
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from matplotlib import cm
from matplotlib.colors import ListedColormap

file_path = "prediction_2025-04-01T05.h5"

with h5py.File(file_path, 'r') as f:
    lat = f['latitude'][:]
    lon = f['longitude'][:]
    pm25 = f['PM2.5'][:]

# --- Infer shape ---
if len(lat) == len(lon) == len(pm25):
    unique_lat = np.unique(lat)
    unique_lon = np.unique(lon)
    lat2d, lon2d = np.meshgrid(unique_lat, unique_lon, indexing='ij')
    pm25_2d = np.full(lat2d.shape, np.nan)

    for i in range(len(pm25)):
        lat_idx = np.where(unique_lat == lat[i])[0][0]
        lon_idx = np.where(unique_lon == lon[i])[0][0]
        pm25_2d[lat_idx, lon_idx] = pm25[i]
else:
    raise ValueError("Latitude, longitude, and PM2.5 shapes do not match.")

# --- Mask NaNs and values == 0 ---
masked_pm25 = np.ma.masked_where((pm25_2d == 0) | np.isnan(pm25_2d), pm25_2d)

# --- Custom colormap where 0 is white ---
base_cmap = cm.get_cmap('plasma', 256)
new_colors = base_cmap(np.linspace(0, 1, 256))
new_colors[0] = [1, 1, 1, 1]  # Set lowest value to white
custom_cmap = ListedColormap(new_colors)

# --- Plot ---
plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

plot = ax.pcolormesh(lon2d, lat2d, masked_pm25, cmap=custom_cmap, shading='auto', vmin=0, vmax=100)

ax.add_feature(cfeature.BORDERS, linewidth=0.7)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
ax.set_extent([65, 100, 5, 40], crs=ccrs.PlateCarree())

# City markers
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

title = os.path.basename(file_path).replace('.h5', '')
plt.title(f'{title} | PM2.5 Concentration Map')
plt.tight_layout()
plt.savefig(f'{title}_pm25_map.png')
plt.show()
