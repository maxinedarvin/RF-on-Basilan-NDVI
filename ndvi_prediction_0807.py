import os
import re
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import r2_score
import xarray as xr

# Set Local File Paths
print("📂 Setting Local File Paths...")
data_folder = "BCC 08-07 NDVI FImputed Cubic Spline s=1"
real_ndvi_folder = "BCC 08-07 NDVI FImputed Cubic Spline s=1"
error_maps_folder = "08-07 NDVI Error Maps"
grib_file = "ERA5.grib"

print("📥 Loading ERA5 GRIB data...")

# Load the main dataset (excluding tp, mx2t, mn2t)
ds_main = xr.open_dataset(grib_file, engine="cfgrib", backend_kwargs={"filter_by_keys": {}})
print("✅ Loaded ERA5 Weather Variables (excluding tp, mx2t, mn2t):", list(ds_main.data_vars.keys()))

# Try to load total precipitation (tp) separately
try:
    ds_tp = xr.open_dataset(grib_file, engine="cfgrib", backend_kwargs={"filter_by_keys": {"shortName": "tp"}})
    print("✅ Successfully loaded total precipitation (tp).")
except:
    ds_tp = None
    print("⚠️ Warning: Could not find tp in the GRIB file!")

# Try to load maximum temperature (mx2t) separately
try:
    ds_mx2t = xr.open_dataset(grib_file, engine="cfgrib", backend_kwargs={"filter_by_keys": {"shortName": "mx2t"}})
    print("✅ Successfully loaded maximum temperature (mx2t).")
except:
    ds_mx2t = None
    print("⚠️ Warning: Could not find mx2t in the GRIB file!")

# Try to load minimum temperature (mn2t) separately
try:
    ds_mn2t = xr.open_dataset(grib_file, engine="cfgrib", backend_kwargs={"filter_by_keys": {"shortName": "mn2t"}})
    print("✅ Successfully loaded minimum temperature (mn2t).")
except:
    ds_mn2t = None
    print("⚠️ Warning: Could not find mn2t in the GRIB file!")

# Merge the datasets if available
if ds_tp is not None:
    ds_tp = ds_tp.assign_coords({"time": ds_main["time"]})  # Align time coordinates
    ds_tp = ds_tp.drop_vars("step", errors="ignore")  # Remove conflicting step variable
else:
    ds_tp = ds_main  # Use only the main dataset if tp is missing

if ds_mx2t is not None:
    ds_mx2t = ds_mx2t.assign_coords({"time": ds_main["time"]})
    ds_mx2t = ds_mx2t.drop_vars("step", errors="ignore")
else:
    ds_mx2t = ds_main  # Use only the main dataset if mx2t is missing

if ds_mn2t is not None:
    ds_mn2t = ds_mn2t.assign_coords({"time": ds_main["time"]})
    ds_mn2t = ds_mn2t.drop_vars("step", errors="ignore")
else:
    ds_mn2t = ds_main  # Use only the main dataset if mn2t is missing

# Now merge all datasets (main, tp, mx2t, mn2t) if available
ds = xr.merge([ds_main, ds_tp, ds_mx2t, ds_mn2t], compat="override")
print("✅ Successfully merged ERA5 dataset with tp, mx2t, and mn2t.")

# Print final available variables
print("📊 Final Loaded Variables in ERA5 Dataset:", list(ds.data_vars.keys()))

# Extract relevant ERA5 variables (ensure they have the same shape)
prec = ds["tp"] if "tp" in ds else None  # Handle missing tp
mx2t = ds["mx2t"] if "mx2t" in ds else None  # Handle missing mx2t
mn2t = ds["mn2t"] if "mn2t" in ds else None  # Handle missing mn2t
tcrw = ds["tcrw"]
temp_2m = ds["t2m"]
dew = ds["d2m"]

# Convert time to a Pandas datetime format
era5_dates = pd.to_datetime(ds["time"].values)

# Flatten spatial dimensions while ensuring all variables have the same time dimension
time_steps = len(era5_dates)

prec_values = prec.values.reshape(time_steps, -1).mean(axis=1) if prec is not None else [None] * time_steps
mx2t_values = mx2t.values.reshape(time_steps, -1).mean(axis=1) if mx2t is not None else [None] * time_steps
mn2t_values = mn2t.values.reshape(time_steps, -1).mean(axis=1) if mn2t is not None else [None] * time_steps
tcrw_values = tcrw.values.reshape(time_steps, -1).mean(axis=1)
temp_2m_values = temp_2m.values.reshape(time_steps, -1).mean(axis=1)
dew_values = dew.values.reshape(time_steps, -1).mean(axis=1)

# **Create new features before defining era5_df**
E_dew = 6.112 * np.exp((17.67*dew_values) / (dew_values + 243.5))
E_air = 6.112 * np.exp((17.67*temp_2m_values) / (temp_2m_values + 243.5))
hum = (E_dew / E_air) * 100
prec_temp = prec_values / temp_2m_values
diff_temp = mx2t_values - mn2t_values

# Now, create the era5_df DataFrame **without** min_temp and max_temp
era5_df = pd.DataFrame({
    "time": era5_dates,
    "prec": prec_values,
    "tcrw": tcrw_values,
    "temp": temp_2m_values,
    "hum": hum,
    "prec_temp": prec_temp,
    "diff_temp": diff_temp
})

# Create the error maps folder if it doesn't exist
if not os.path.exists(error_maps_folder):
    os.makedirs(error_maps_folder)
    print(f"📁 Created folder: {error_maps_folder}")

# Function to Extract Dates from Filenames (New Format "0807ndvi_YYYYMMDD_imputed_spline3.tif")
def extract_date_from_filename(filename):
    match = re.search(r'0807ndvi_(\d{8})\_imputed_spline3.tif$', filename)  # Adjusted for new format
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d")
    return None

# Load raster files from a directory
def load_rasters_from_folder(folder_path):
    print(f"📥 Loading raster data from: {folder_path}...")
    raster_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    valid_files = [(f, extract_date_from_filename(f)) for f in raster_files]
    valid_files = [(f, d) for f, d in valid_files if d is not None]
    valid_files.sort(key=lambda x: x[1])  # Sort by extracted date

    raster_data = []
    dates = []

    for i, (file, date) in enumerate(valid_files):
        with rasterio.open(os.path.join(folder_path, file)) as dataset:
            raster_data.append(dataset.read(1))  # Read first band
        dates.append(date)
        if (i + 1) % 10 == 0 or i == len(valid_files) - 1:
            print(f"✅ Loaded {i+1}/{len(valid_files)} files...")

    return np.array(raster_data), dates

# Load Training Data
ndvi_data, dates = load_rasters_from_folder(data_folder)

# Function to Load a Specific NDVI File by Date (For Lagging)
def load_ndvi_for_date(date, primary_folder, fallback_folder):
    """
    Loads NDVI for a given date.
    - Tries primary_folder (data_folder) first.
    - Falls back to fallback_folder (real_ndvi_folder) if not found.
    """
    filename = f"0807ndvi_{date.strftime('%Y%m%d')}_imputed_spline3.tif"

    # Check primary folder (data_folder)
    file_path_primary = os.path.join(primary_folder, filename)
    if os.path.exists(file_path_primary):
        with rasterio.open(file_path_primary) as dataset:
            return dataset.read(1)  # Return NDVI as numpy array
    
    # Check fallback folder (real_ndvi_folder)
    file_path_fallback = os.path.join(fallback_folder, filename)
    if os.path.exists(file_path_fallback):
        print(f"⚠️ Using NDVI from fallback folder ({fallback_folder}) for {date.strftime('%Y-%m-%d')}")
        with rasterio.open(file_path_fallback) as dataset:
            return dataset.read(1)  

    # Data not found in either folder
    print(f"⚠️ No NDVI data available for {date.strftime('%Y-%m-%d')} in either folder.")
    return None


# Feature Engineering: Temporal Features and Lagged NDVI
print("🛠️ Creating temporal and lagged NDVI features...")

def create_temporal_features(dates, ndvi_data=None, era5_data=None, grib_file=None):
    print("🛠️ Generating temporal and weather features...")

    # ✅ Temporal Features
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])  
    month = np.array([d.month for d in dates]) 
    year = np.array([d.year for d in dates]) 

    temporal_features = np.column_stack([day_of_year, month, year])
    print(f"📊 Temporal Features Shape: {temporal_features.shape}") 

    if era5_data is not None:
        era5_data["date"] = pd.to_datetime(era5_data["time"]).dt.date 

    ndvi_dates_df = pd.DataFrame({"date": [d.date() for d in dates]})
    merged_era5 = ndvi_dates_df.merge(era5_data, on="date", how="left")  

    # ✅ Replace 0.0 values with NaN before interpolation
    numeric_cols = ["prec", "tcrw", "temp", "hum", "diff_temp", "prec_temp"]
    merged_era5[numeric_cols] = merged_era5[numeric_cols].replace(0.0, np.nan)

    # ✅ Interpolate missing values row-wise, and backfill any remaining NaNs
    merged_era5[numeric_cols] = merged_era5[numeric_cols].interpolate(method="linear", axis=0).bfill()
    era5_features = merged_era5[numeric_cols].values  
    era5_features = era5_features[:len(dates)]  

    print(f"📊 ERA5 Features Shape: {era5_features.shape}")  

    era5_filled = pd.DataFrame(era5_features, columns=numeric_cols, dtype=float).interpolate(method="linear", axis=0).bfill()

    lag_30_era5 = np.roll(era5_filled.values, shift=6, axis=0)  
    lag_60_era5 = np.roll(era5_filled.values, shift=12, axis=0)  
    lag_30_era5[:6, :] = era5_filled.values[:6, :]  
    lag_60_era5[:12, :] = era5_filled.values[:12, :]

    era5_lagged = np.hstack([era5_filled, lag_30_era5, lag_60_era5])
    print(f"📊 ERA5 Features Shape (after lagging): {era5_lagged.shape}")

    # ✅ 25-Day Rolling Average for ERA5 Variables
    rolling_avg_25_era5 = era5_filled.rolling(window=5).mean().values
    era5_lagged = np.hstack([era5_lagged, rolling_avg_25_era5])
    print(f"📊 ERA5 Features Shape (after adding rolling averages): {era5_lagged.shape}")

    if ndvi_data is not None:
        if len(ndvi_data.shape) == 3:
            ndvi_reshaped = ndvi_data.reshape(ndvi_data.shape[0], -1)
        elif len(ndvi_data.shape) == 2 and ndvi_data.shape[0] == 1:
            ndvi_reshaped = np.repeat(ndvi_data, repeats=len(dates), axis=0)
        else:
            ndvi_reshaped = ndvi_data

        ndvi_filled = pd.DataFrame(ndvi_reshaped).interpolate(method="linear", axis=0).bfill()
        ndvi_filled[ndvi_filled == 0.0] = np.nan

        lag_30 = np.roll(ndvi_filled.values, shift=6, axis=0) 
        lag_60 = np.roll(ndvi_filled.values, shift=12, axis=0)  
        lag_30[:6, :] = ndvi_filled.values[:6, :]  
        lag_60[:12, :] = ndvi_filled.values[:12, :]

        rolling_avg_25 = ndvi_filled.rolling(window=5).mean().values
        ndvi_features = np.hstack([lag_30, lag_60, rolling_avg_25])
        num_pixels = ndvi_filled.shape[1]  
    else:
        print(f"⚠️ ERROR: NDVI data is missing! Cannot create lagged NDVI features.")
        exit()

    print(f"📊 NDVI Features Shape: {ndvi_features.shape}")  

    all_features = np.hstack([temporal_features, era5_lagged, ndvi_features])
    print(f"📊 All Features Shape: {all_features.shape}")  # expect 6069+6=6075 with 08-07
    return all_features, temporal_features, ndvi_features, num_pixels

X_full, temporal_features_repeated, ndvi_features, num_pixels = create_temporal_features(dates, ndvi_data, era5_df)

# Train-Test Split
print("📊 Splitting dataset into training and testing...")
split_date = datetime(2023, 8, 7)
train_mask = np.array([d < split_date for d in dates])  # Training data before split date
test_mask = np.array([d >= split_date for d in dates])  # Test data after split date

X_train, X_test = X_full[train_mask], X_full[test_mask]

print(f"NDVI Data Shape: {ndvi_data.shape}")
print(f"Train Mask Shape: {train_mask.shape}")
print(f"Test Mask Shape: {test_mask.shape}")

print(f"Total elements in train data: {ndvi_data[train_mask].size}")  # Should match 368 * 2112
print(f"Total elements in test data: {ndvi_data[test_mask].size}")    # Should match 136 * 2112


# Check the number of pixels to reshape correctly
temporal_features_count = 3
era5_features_count = 24
ndvi_features_per_pixel = 3  # Each pixel gets 2 lagged NDVI features

# Calculate the number of pixels
num_pixels = (X_train.shape[1] - temporal_features_count - era5_features_count) // ndvi_features_per_pixel
print(f"Number of pixels: {num_pixels}")

y_train = ndvi_data[train_mask].reshape(-1, num_pixels)  # Reshape into (samples, pixels)
y_test = ndvi_data[test_mask].reshape(-1, num_pixels)    # Reshape into (samples, pixels)

# CHECK SHAPES
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"X_train size (total elements): {X_train.size}")
print(f"y_train size (total elements): {y_train.size}")

# Create DataFrame for X_train
df_X_train = pd.DataFrame(X_train)

# Feature Names
feature_names = []

# Add Temporal Features
feature_names.extend(["Day of Year", "Month", "Year"])

# Add ERA5 Features (Current, 30-day lag, 60-day lag)
era5_feature_names = ["prec", "tcrw", "temp", "hum", "diff_temp", "prec_temp"]
for time_lag in ["", "_lag30", "_lag60", "_rolling25"]:
    for era5_feature in era5_feature_names:
        feature_names.append(f"{era5_feature}{time_lag}")

# Add NDVI Lag Features Per Pixel
for pixel in range(num_pixels):
    feature_names.append(f"ndvi_pixel{pixel}_lag30")
    feature_names.append(f"ndvi_pixel{pixel}_lag60")
    feature_names.append(f"ndvi_pixel{pixel}_rolling25")

# Assign column names to X_train
df_X_train.columns = feature_names[:X_train.shape[1]]
# Create DataFrame for y_train
df_y_train = pd.DataFrame(y_train, columns=[f"ndvi_pixel{i}" for i in range(y_train.shape[1])])
# Concatenate Features & Target
df_train = pd.concat([df_X_train, df_y_train], axis=1)
# Save to CSV
train_csv_path = "training_data_0807.csv"
df_train.to_csv(train_csv_path, index=False)
print(f"✅ Training data saved to training_data_0807.csv")

if np.any(np.isnan(y_train)):
    print("⚠️ Found NaN values in y_train. Filling NaN values with zeros.")
    y_train = pd.DataFrame(y_train).fillna(0).values

if np.any(np.isnan(y_test)):
    print("⚠️ Found NaN values in y_test. Filling NaN values with zeros.")
    y_test = pd.DataFrame(y_test).fillna(0).values

# Define parameter grid
param_grid = {
    "n_estimators": [25, 50, 100, 150],
    "max_depth": [5, 10, 15, 20, 25, None],
    "max_features": [30, 40, "sqrt", "log2"],
    "min_samples_leaf": [1, 2, 5, 10],
    "min_samples_split": [2, 4, 6, 8, 10]
}

# 🚀 **Hyperparameter Tuning with GridSearchCV**
print("🔍 Performing hyperparameter tuning with GridSearchCV...")

# Define 5-fold time series split
print("📊 Performing 5-fold Time Series Cross-Validation...")
tscv = TimeSeriesSplit(n_splits=5)
rf = RandomForestRegressor(random_state=42)

# # Perform Grid Search
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=tscv, scoring="r2", n_jobs=-1, verbose=2)
# grid_search.fit(X_train, y_train)

# # Get best hyperparameters
# best_params = grid_search.best_params_
# print(f"✅ Best Hyperparameters: {best_params}")

# # Train Final Model with Best Parameters
# print("🚀 Training the final Random Forest model...")

# # Perform cross-validation with TimeSeriesSplit
# cv_scores = cross_val_score(rf, X_train, y_train, cv=tscv, scoring="r2", n_jobs=-1)
# print(f"📊 Mean CV R²: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
# mean_r2_score = np.mean(cv_scores)

# BEST PARAMS EXAMPLE
best_params = {
    "n_estimators": 100,  
    "max_depth": 5,
    "max_features": 30,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "bootstrap": True
}

# Train final model on the entire training set
print("🔄 Training final model on full training set...")
best_model = RandomForestRegressor(**best_params, oob_score=True, random_state=42, n_jobs=-1, verbose=2)
best_model.fit(X_train, y_train)

print("✅ Final model training complete!")

# Display Feature Importance
print("📊 Calculating feature importance...")

# Get the feature importance from the trained model
feature_importance = best_model.feature_importances_

# Create DataFrame for feature importance
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importance
})

# Sort features by importance
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Top 210 most important features
top_10_features = importance_df.head(10)

# Plot feature importance
plt.figure(figsize=(12, 8))
plt.barh(top_10_features["Feature"], top_10_features["Importance"], color='#FFA500')
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance Plot for Brgy. Baluno")
plt.gca().invert_yaxis()
plt.show()

print("✅ Feature importance calculation and visualization completed!")

ndvi_bins = np.arange(-1, 1.1, 0.10)  # NDVI range bins from -1 to 1 with 0.10 intervals
ndvi_colors = [
    "#000000", "#000000", "#000000", "#000000", "#000000", "#000000", "#000000", "#000000", "#000000", "#000000",
    "#5B2C06", "#5B2C06",  # Brown for the range 0.0-0.1, 0.1-0.2
    "#E71D1D",  # Red for range 0.2–0.3
    "#E76F1D",  # Orange for range 0.3–0.4
    "#F7D04A",  # Yellow for range 0.4–0.5
    "#FFE75A",  # Light Yellow for range 0.5–0.6
    "#63E063",  # Light Green for range 0.6–0.7
    "#73F5E6",  # Cyan for range 0.7–0.8
    "#3682F5",  # Blue for range 0.8–0.9
    "#0A24F5",  # Dark Blue for range 0.9–1
]

# Ensure the color list matches the number of bins (there should be 21 colors for 21 bins)
expanded_colors = [
    "#000000"] * 10 + ["#5B2C06"] * 2 + ["#E71D1D", "#E76F1D", "#F7D04A", "#FFE75A", "#63E063", "#73F5E6", "#3682F5", "#0A24F5"]  # Repeat brown 12 times and add other colors

# Create the colormap from the expanded list of colors
ndvi_cmap = LinearSegmentedColormap.from_list("ndvi_gradient", expanded_colors, N=256)

# Define a function to save NDVI heatmaps as PNGs
def save_ndvi_heatmap(data, title, filename, cmap=ndvi_cmap):
    data = np.ma.masked_invalid(data)  # Mask NaN values
    
    # Apply the colormap directly without normalization
    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap=cmap, vmin=-1, vmax=1)  # Use fixed vmin and vmax to ensure color consistency
    plt.colorbar(label="NDVI Value")
    plt.title(title)
    plt.axis("off")  # Hide axes
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Start date for predictions
date_start = datetime(2023, 8, 7)

# Manually add the special dates without prediction
special_dates = [
    datetime(2023, 7, 23),
    datetime(2023, 7, 28),
    datetime(2023, 8, 2)
]

# Generate prediction dates in 5-day intervals up to 2 months after the start date
prediction_dates = [date_start + timedelta(days=5*i) for i in range(0, 16)]  # 12 intervals -> 60 days, +3 intervals before testing
# Combine special dates and prediction dates
prediction_dates = sorted(set(prediction_dates + special_dates))

# Generate Future Predictions Using Historical NDVI for Lagging
print("📈 Generating future predictions...")

predictions = []
for pred_date in prediction_dates:
    print(f"🔄 Processing prediction for {pred_date.strftime('%Y-%m-%d')}")

    if pred_date in special_dates:
        print(f"⚠️ Skipping prediction for {pred_date.strftime('%Y-%m-%d')}, generating only the actual NDVI PNG.")
        
        actual_filename = os.path.join(real_ndvi_folder, f"0807ndvi_{pred_date.strftime('%Y%m%d')}_imputed_spline3.tif")
        if os.path.exists(actual_filename):
            with rasterio.open(actual_filename) as dataset:
                actual_ndvi = dataset.read(1)

            # Save actual NDVI heatmap
            actual_png = os.path.join(error_maps_folder, f"{pred_date.strftime('%Y%m%d')}_ndvi_actual.png")
            save_ndvi_heatmap(actual_ndvi, f"Actual NDVI {pred_date.strftime('%Y-%m-%d')}", actual_png)

            print(f"🖼️ Saved PNG for actual NDVI for {pred_date.strftime('%Y-%m-%d')}: {actual_png}")
        else:
            print(f"⚠️ No actual NDVI data available for {pred_date.strftime('%Y-%m-%d')} in {real_ndvi_folder}")
        
        # Skip prediction for this date
        continue

    # Load lagged NDVI from data_folder, fallback to real_ndvi_folder
    lag_30_date = pred_date - timedelta(days=30)
    lag_60_date = pred_date - timedelta(days=60)

    lag_30_ndvi = load_ndvi_for_date(lag_30_date, data_folder, real_ndvi_folder)
    lag_60_ndvi = load_ndvi_for_date(lag_60_date, data_folder, real_ndvi_folder)

    if lag_30_ndvi is None or lag_60_ndvi is None:
        print(f"⚠️ NDVI missing for {pred_date.strftime('%Y-%m-%d')}, interpolating...")
        
        # Convert NDVI data to DataFrame for interpolation
        ndvi_df = pd.DataFrame(ndvi_data.reshape(ndvi_data.shape[0], -1))

        # Interpolate missing values along the time axis
        ndvi_interpolated = ndvi_df.interpolate(method="linear", axis=0).bfill().ffill()

        # Use the last available interpolated NDVI for missing lags
        lag_30 = ndvi_interpolated.iloc[-6].values.reshape(1, -1)  # Approximate last 30-day lag
        lag_60 = ndvi_interpolated.iloc[-12].values.reshape(1, -1)  # Approximate last 60-day lag
    else:
        lag_30 = lag_30_ndvi.reshape(1, -1)
        lag_60 = lag_60_ndvi.reshape(1, -1)

    # Ensure correct NDVI input shape
    future_ndvi = np.hstack([lag_30, lag_60])  # Combine both lags

    future_features, _, _, _ = create_temporal_features([pred_date], np.array([future_ndvi]), era5_df)
    print(f"future_features shape: {future_features.shape}")

    # Ensure future_features has the same shape as X_train (6357 features) # 3045+6 = 3051 upon coding 08-07
    expected_feature_count = 2781
    if future_features.shape[1] > expected_feature_count:
        future_features = future_features[:, :expected_feature_count]
    elif future_features.shape[1] < expected_feature_count:
        # If future_features has fewer features than expected, pad with zeros
        future_features = np.pad(future_features, ((0, 0), (0, expected_feature_count - future_features.shape[1])), 'constant', constant_values=0)
    
    # ✅ Ensure X_test matches X_train in feature count
    X_test = X_test[:, :2781]  # Trim excess features if needed
    print(f"future_features shape after adjustment: {future_features.shape}")
    # ✅ Predict NDVI values for the test set
    y_pred_test = best_model.predict(X_test)

    # Make predictions
    prediction = best_model.predict(future_features)
    predictions.append(prediction)


# Flatten arrays for comparison
y_test_flat = y_test.flatten()
y_pred_test_flat = y_pred_test.flatten()

# Mask out zero NDVI pixels
valid_mask = y_test_flat > 0  # Only consider nonzero pixels
actual_nonzero = y_test_flat[valid_mask]
predicted_nonzero = y_pred_test_flat[valid_mask]

# Compute mean and standard deviation only for vegetated areas
actual_mean, actual_std = actual_nonzero.mean(), actual_nonzero.std()
predicted_mean, predicted_std = predicted_nonzero.mean(), predicted_nonzero.std()

print(f"📊 Nonzero Actual NDVI - Mean: {actual_mean:.4f}, Std: {actual_std:.4f}")
print(f"📊 Nonzero Predicted NDVI - Mean: {predicted_mean:.4f}, Std: {predicted_std:.4f}")

actual_mean, actual_std = y_test_flat.mean(), y_test_flat.std()
predicted_mean, predicted_std = y_pred_test_flat.mean(), y_pred_test_flat.std()

print(f"📊 Actual NDVI - Mean: {actual_mean:.4f}, Std: {actual_std:.4f}")
print(f"📊 Predicted NDVI - Mean: {predicted_mean:.4f}, Std: {predicted_std:.4f}")


# # Average R² score of 5 CV folds
# mean_r2_score = np.mean(cv_scores)
# print(f"📊 Average Cross-Validation R² Score: {mean_r2_score:.4f}")

# OOB Score
oob_r2 = best_model.oob_score_
print(f"📊 Out-of-Bag (OOB) R² Score: {oob_r2:.4f}")
# Predict NDVI values
y_pred_test = best_model.predict(X_test)
# Test Set R² Score
test_r2 = r2_score(y_test, y_pred_test)
print(f"📊 Test Set R² Score: {test_r2:.4f}")

# # Overfitting Check
# if mean_r2_score > test_r2 + 0.05:
#     print("⚠️ WARNING: Possible Overfitting in Hyperparameter Optimization! GridSearchCV R² is much higher than Test R².")
#     print("   🔹 Try increasing `cv=10` in GridSearchCV.")
#     print("   🔹 Consider reducing model complexity (e.g., `max_depth`, `min_samples_split`).")

if oob_r2 > test_r2 + 0.05:
    print("⚠️ WARNING: Possible Overfitting in Final Random Forest! OOB Score is much higher than Test Set R².")
    print("   🔹 Try increasing `min_samples_split` to reduce overfitting.")
    print("   🔹 Reduce `max_depth` to prevent deep trees.")
    print("   🔹 Consider reducing the number of trees (`n_estimators`).")

# Convert training and test sets into DataFrames
train_df = pd.DataFrame(X_train)
test_df = pd.DataFrame(X_test)

# Compute mean and standard deviation for each feature
train_stats = train_df.describe().transpose()
test_stats = test_df.describe().transpose()

# Compare feature means and standard deviations
comparison = pd.DataFrame({
    "Train Mean": train_stats["mean"],
    "Test Mean": test_stats["mean"],
    "Train Std": train_stats["std"],
    "Test Std": test_stats["std"],
})

# Display the feature distribution differences
print(comparison)

# Reshape predictions to match NDVI data
height, width = ndvi_data.shape[1], ndvi_data.shape[2]
predictions = np.array(predictions).reshape(len(predictions), height, width)

print(f"📊 Predicted NDVI statistics: Min={np.min(predictions):.4f}, Max={np.max(predictions):.4f}, Mean={np.mean(predictions):.4f}")

# Process predictions and save PNGs (skip special dates for prediction and error maps)
for i, pred in enumerate(predictions):
    date_str = prediction_dates[i].strftime('%Y-%m-%d')

    if prediction_dates[i] in special_dates:
        # For special dates, only save the actual NDVI heatmap (no prediction or error map)
        actual_filename = os.path.join(real_ndvi_folder, f"0807ndvi_{prediction_dates[i].strftime('%Y%m%d')}_imputed_spline3.tif")
        
        if os.path.exists(actual_filename):
            with rasterio.open(actual_filename) as dataset:
                actual_ndvi = dataset.read(1)

            # Save actual NDVI heatmap for special dates
            actual_png = os.path.join(error_maps_folder, f"{prediction_dates[i].strftime('%Y%m%d')}_ndvi_actual.png")
            save_ndvi_heatmap(actual_ndvi, f"Actual NDVI {date_str}", actual_png)
            print(f"🖼️ Saved PNG for actual NDVI for special date {date_str}: {actual_png}")
        else:
            print(f"⚠️ No actual NDVI data available for special date {date_str} in {real_ndvi_folder}")
        
        # Skip prediction and error map generation for special dates
        continue  # Skip the rest of the loop for this date
    
    # For non-special dates, proceed with prediction and error map generation
    actual_filename = os.path.join(real_ndvi_folder, f"0807ndvi_{prediction_dates[i].strftime('%Y%m%d')}_imputed_spline3.tif")
    
    if os.path.exists(actual_filename):
        with rasterio.open(actual_filename) as dataset:
            actual_ndvi = dataset.read(1)

        pred = np.where(pred == 0, np.nan, pred)
        error_map = pred - actual_ndvi

        # File paths for saving
        actual_png = os.path.join(error_maps_folder, f"{prediction_dates[i].strftime('%Y%m%d')}_ndvi_actual.png")
        predicted_png = os.path.join(error_maps_folder, f"{prediction_dates[i].strftime('%Y%m%d')}_ndvi_predicted.png")
        error_png = os.path.join(error_maps_folder, f"{prediction_dates[i].strftime('%Y%m%d')}_error_map.png")

        # Save actual NDVI heatmap
        save_ndvi_heatmap(actual_ndvi, f"Actual NDVI {date_str}", actual_png)

        # Save predicted NDVI heatmap
        save_ndvi_heatmap(pred, f"Predicted NDVI {date_str}", predicted_png)

        # Save error map (NDVI difference)
        plt.figure(figsize=(8, 6))
        plt.imshow(error_map, cmap='RdBu', vmin=-0.5, vmax=0.5)  # Red-Blue for errors
        plt.colorbar(label="Prediction Error (NDVI)")
        plt.title(f"Error Map {date_str}")
        plt.axis("off")
        plt.savefig(error_png, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"🖼️ Saved PNGs for {date_str}:\n  ✅ {actual_png}\n  ✅ {predicted_png}\n  ✅ {error_png}")
    else:
        print(f"⚠️ No actual NDVI data available for {date_str} in {real_ndvi_folder}")
