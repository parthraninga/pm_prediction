import h5py
import joblib
import numpy as np
import xarray as xr
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import os

# Load model
model_pm25 = joblib.load("models/random_forest/rf_model_pm2.5.pkl")
model_pm10 = joblib.load("models/random_forest/rf_model_pm10.pkl")

# Load AOD
with h5py.File("input/aod/3RIMG_01APR2025_0545_L2G_AOD_V02R00.h5", "r") as f:
    aod_data = f["AOD"][0]  # shape (551, 551)
    lat_aod = f["latitude"][:]  # shape (551, 551)
    lon_aod = f["longitude"][:]  # shape (551, 551)

# Load MERRA2 dataset
ds = xr.open_dataset("input/nasa/MERRA2_400.tavg1_2d_flx_Nx.20250401.SUB.nc")

# List available time options
time_options = ds.time.values
print("\nAvailable Time Options:")
for idx, t in enumerate(time_options):
    print(f"{idx}: {str(t)}")

# Ask for time index selection
time_index = int(input("\nSelect time index: "))
selected_time = ds.isel(time=time_index)

# Flatten AOD, lat, lon for mapping
flat_lat = lat_aod.flatten()
flat_lon = lon_aod.flatten()
flat_aod = aod_data.flatten()

def find_nearest_aod(lat_query, lon_query):
    dist = np.sqrt((flat_lat - lat_query) ** 2 + (flat_lon - lon_query) ** 2)
    idx = np.argmin(dist)
    return flat_aod[idx]

# Loop through all (lat, lon) in selected_time
lats = selected_time.lat.values
lons = selected_time.lon.values
results = []

for lat in lats:
    for lon in lons:
        aod_val = find_nearest_aod(lat, lon)
        hflux = float(selected_time["HFLUX"].sel(lat=lat, lon=lon).values)
        pblh = float(selected_time["PBLH"].sel(lat=lat, lon=lon).values)
        qlml = float(selected_time["QLML"].sel(lat=lat, lon=lon).values)
        speed = float(selected_time["SPEED"].sel(lat=lat, lon=lon).values)
        tlml = float(selected_time["TLML"].sel(lat=lat, lon=lon).values)

        features = pd.DataFrame([{
                                        'AOD': aod_val,
                                        'HFLUX': hflux,
                                        'PBLH': pblh,
                                        'QLML': qlml,
                                        'SPEED': speed,
                                        'TLML': tlml
                                    }])

        pm25 = model_pm25.predict(features)[0]
        pm10 = model_pm10.predict(features)[0]

        results.append([lat, lon, pm25, pm10])

# Convert to arrays
results = np.array(results)
lat_out = results[:, 0]
lon_out = results[:, 1]
pm25_out = results[:, 2]
pm10_out = results[:, 3]

os.makedirs("output", exist_ok=True)
# Save to output HDF5
output_path = f"output/prediction_{str(ds.time.values[time_index])[:13].replace(':', '')}.h5"
with h5py.File(output_path, "w") as f:
    f.create_dataset("latitude", data=lat_out)
    f.create_dataset("longitude", data=lon_out)
    f.create_dataset("PM2.5", data=pm25_out)
    f.create_dataset("PM10", data=pm10_out)

print(f"\n Prediction saved to {output_path}")
