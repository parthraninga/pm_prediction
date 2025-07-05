import h5py
import joblib
import numpy as np
import xarray as xr
import pandas as pd
import logging
import time
import os

from sklearn.ensemble import RandomForestRegressor

# Setup logging to file and console
logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# File handler
file_handler = logging.FileHandler("logs/prediction.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


start_time = time.time()
logging.info("Prediction process started.")

# Load model
model_pm25 = joblib.load("models/random_forest/rf_model_pm2.5.pkl")
model_pm10 = joblib.load("models/random_forest/rf_model_pm10.pkl")

# Load AOD
with h5py.File("input/aod/3RIMG_01APR2025_0545_L2G_AOD_V02R00.h5", "r") as f:
    aod_data = f["AOD"][0]  # shape (551, 551)
    lat_aod = f["latitude"][:]
    lon_aod = f["longitude"][:]

# Load MERRA2 dataset
ds = xr.open_dataset("input/nasa/MERRA2_400.tavg1_2d_flx_Nx.20250401.SUB.nc")

# List time options
time_options = ds.time.values
print("\nAvailable Time Options:")
for idx, t in enumerate(time_options):
    print(f"{idx}: {str(t)}")

# User selects a time index
time_index = int(input("\nSelect time index: "))
selected_time = ds.isel(time=time_index)

# Flatten AOD arrays
flat_lat = lat_aod.flatten()
flat_lon = lon_aod.flatten()
flat_aod = aod_data.flatten()

def find_nearest_aod(lat_query, lon_query):
    dist = np.sqrt((flat_lat - lat_query) ** 2 + (flat_lon - lon_query) ** 2)
    idx = np.argmin(dist)
    return flat_aod[idx]

# Predict
lats = selected_time.lat.values
lons = selected_time.lon.values
results = []

total = len(lats) * len(lons)
count = 0

for lat in lats:
    for lon in lons:
        count += 1
        if count % 10 == 0 or count == total:
            logging.info(f"Processed {count}/{total} points")

        aod_val = find_nearest_aod(lat, lon)
        #logging.info(f"AOD at lat={lat}, lon={lon}: {aod_val}")  
        hflux = float(selected_time["HFLUX"].sel(lat=lat, lon=lon).values)
        pblh = float(selected_time["PBLH"].sel(lat=lat, lon=lon).values)
        qlml = float(selected_time["QLML"].sel(lat=lat, lon=lon).values)
        speed = float(selected_time["SPEED"].sel(lat=lat, lon=lon).values)
        tlml = float(selected_time["TLML"].sel(lat=lat, lon=lon).values)

        if np.isnan(aod_val) or aod_val == -999.0:
            logging.warning(f"AOD is NaN at lat={lat}, lon={lon}. Setting PM2.5 and PM10 to NaN.")
            pm25, pm10 = np.nan, np.nan
        else:
            features = pd.DataFrame([{
                'aod': aod_val,  # Lowercase key if model trained with lowercase
                'HFLUX': hflux,
                'PBLH': pblh,
                'QLML': qlml,
                'SPEED': speed,
                'TLML': tlml
            }])
            pm25 = model_pm25.predict(features)[0]
            pm10 = model_pm10.predict(features)[0]

        results.append([lat, lon, pm25, pm10])

results = np.array(results)
lat_out = results[:, 0]
lon_out = results[:, 1]
pm25_out = results[:, 2]
pm10_out = results[:, 3]

# Save
os.makedirs("output", exist_ok=True)
output_path = f"output/prediction_{str(ds.time.values[time_index])[:13].replace(':', '')}.h5"

with h5py.File(output_path, "w") as f:
    f.create_dataset("latitude", data=lat_out)
    f.create_dataset("longitude", data=lon_out)
    f.create_dataset("PM2.5", data=pm25_out)
    f.create_dataset("PM10", data=pm10_out)

end_time = time.time()
elapsed = round(end_time - start_time, 2)
logging.info(f"Prediction complete. Time taken: {elapsed} seconds")
logging.info(f"Saved to: {output_path}")

print(f"\nPrediction saved to: {output_path}")
print(f"Time taken: {elapsed} seconds. See logs/prediction.log for details.")
