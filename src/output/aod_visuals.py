import h5py
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

file_path = "prediction_2025-04-01T05.h5"

with h5py.File(file_path, 'r') as f:
    lat = f['latitude'][:]
    lon = f['longitude'][:]
    pm25 = f['PM2.5'][:]

# --- Infer shape ---
if len(lat) == len(lon) == len(pm25):
    # Find unique lats and lons
    unique_lat = np.unique(lat)
    unique_lon = np.unique(lon)

    # Create 2D grid
    lat2d, lon2d = np.meshgrid(unique_lat, unique_lon, indexing='ij')

    # Create 2D PM2.5 matrix
    pm25_2d = np.full(lat2d.shape, np.nan)

    # Fill PM2.5 matrix
    for i in range(len(pm25)):
        lat_idx = np.where(unique_lat == lat[i])[0][0]
        lon_idx = np.where(unique_lon == lon[i])[0][0]
        pm25_2d[lat_idx, lon_idx] = pm25[i]
else:
    raise ValueError("Latitude, longitude, and PM2.5 shapes do not match.")

# --- Plot ---
plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
plot = ax.pcolormesh(lon2d, lat2d, pm25_2d, cmap='viridis', shading='auto')

ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
ax.set_extent([65, 100, 5, 40], crs=ccrs.PlateCarree())

cbar = plt.colorbar(plot, orientation='vertical', pad=0.05, aspect=30)
cbar.set_label('Predicted PM2.5')

title = os.path.basename(file_path).replace('.h5', '')
plt.title(f'{title} | PM2.5 Concentration')
plt.tight_layout()
plt.savefig(f'{title}_pm25_map.png')
plt.show()
