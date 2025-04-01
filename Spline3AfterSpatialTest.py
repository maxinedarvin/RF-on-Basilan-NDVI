import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
import geopandas as gpd
from scipy.interpolate import UnivariateSpline
from datetime import datetime, timedelta
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import mean_squared_error

# Define NDVI colormap
ndvi_bins = np.arange(-1, 1.1, 0.10)
ndvi_colors = ["#000000"] * 10 + ["#5B2C06"] * 2 + ["#E71D1D", "#E76F1D", "#F7D04A", "#FFE75A", "#63E063", "#73F5E6", "#3682F5", "#0A24F5"]
ndvi_cmap = LinearSegmentedColormap.from_list("ndvi_gradient", ndvi_colors, N=256)

# Function to save NDVI heatmaps
def save_ndvi_heatmap(data, title, filename, cmap=ndvi_cmap):
    data = np.ma.masked_invalid(data)  # Mask NaN values
    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar(label="NDVI Value")
    plt.title(title)
    plt.axis("off")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

# Function to save the imputed raster
def save_imputed_raster(original_raster, data, output_path):
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=original_raster.crs,
        transform=original_raster.transform
    ) as dst:
        dst.write(data, 1)

# Define paths
clusters = ["02-22", "04-03", "08-07", "08-07 mini"]
base_folder = "/Darvin/BCC {cluster} NDVI FImputedIter/"
output_folder_raster = "/Darvin/BCC {cluster} NDVI FImputed Cubic Spline s=0/"
output_folder_heatmap = "/Darvin/{cluster} NDVI FImputed Cubic Spline s=0/"
filename_prefixes = {
    "02-22": "0222ndvi",
    "04-03": "0403ndvi",
    "08-07": "0807ndvi",
    "08-07 mini": "m0807ndvi"
}
target_date = "20180209"

# Get non-empty pixel coordinates for all clusters
non_empty_pixel_coords = {}
default_shapes = {}
for cluster in clusters:
    raster_file = os.path.join(base_folder.format(cluster=cluster), f"{filename_prefixes[cluster]}_{target_date}_imputed.tif")
    if os.path.exists(raster_file):
        with rasterio.open(raster_file) as src:
            ndvi_array = src.read(1).astype(float)
            non_empty_coords = np.argwhere(~np.isnan(ndvi_array))
            non_empty_pixel_coords[cluster] = non_empty_coords.tolist()
            default_shapes[cluster] = ndvi_array.shape
    else:
        print(f"Raster file not found for cluster {cluster} on date {target_date}.")

# Define date range
start_date = datetime(2018, 1, 5)
end_date = datetime(2024, 12, 29)
date_list = [start_date + timedelta(days=x) for x in range(0, (end_date - start_date).days + 1, 5)]
formatted_dates = [date.strftime("%Y%m%d") for date in date_list]

# Verify the number of dates generated
expected_num_dates = 511
actual_num_dates = len(formatted_dates)
print(f"Expected number of dates: {expected_num_dates}")
print(f"Actual number of dates generated: {actual_num_dates}")

if actual_num_dates != expected_num_dates:
    raise ValueError(f"Number of generated dates ({actual_num_dates}) does not match the expected count ({expected_num_dates}).")

for cluster in clusters:
    print(f"\n🔍 Processing cluster: {cluster}")

    os.makedirs(output_folder_raster.format(cluster=cluster), exist_ok=True)
    os.makedirs(output_folder_heatmap.format(cluster=cluster), exist_ok=True)

    ndvi_data = []
    missing_dates = []
    first_existing_file = None
    first_existing_raster = None
    default_shape = default_shapes.get(cluster, (0, 0))  # Default shape for NaN arrays

    coords_within_boundary = non_empty_pixel_coords.get(cluster, [])
    print(f"Number of entries in coords within boundary for cluster {cluster}: {len(coords_within_boundary)}")

    # First loop: add existing rasters and create NaN rasters
    for target_date in formatted_dates:
        raster_file = os.path.join(base_folder.format(cluster=cluster), f"{filename_prefixes[cluster]}_{target_date}_imputed.tif")
        if os.path.exists(raster_file):
            with rasterio.open(raster_file) as src:
                ndvi_array = src.read(1).astype(float)
                ndvi_data.append(ndvi_array)
                output_png = os.path.join(output_folder_heatmap.format(cluster=cluster), f"{filename_prefixes[cluster]}_{target_date}_imputed_spline3.png")
                output_raster = os.path.join(output_folder_raster.format(cluster=cluster), f"{filename_prefixes[cluster]}_{target_date}_imputed_spline3.tif")
                save_ndvi_heatmap(ndvi_array, f"NDVI Raster - {cluster} ({target_date})", output_png)
                save_imputed_raster(src, ndvi_array, output_raster)
                print(f"📷 Heatmap saved: {output_png}")
                print(f"🗃 Raster saved: {output_raster}")
                if first_existing_file is None:
                    first_existing_file = raster_file
                    first_existing_raster = src
        else:
            print(f"⚠ No raster file found for {target_date} in {cluster}. Added to missing date list.")
            missing_dates.append(target_date)
            nan_array = np.full(default_shape, np.nan)
            ndvi_data.append(nan_array)

    ndvi_stack = np.stack(ndvi_data, axis=0)
    original_values = ndvi_stack.copy()
    height, width = ndvi_stack.shape[1], ndvi_stack.shape[2]

    # Ensure ndvi_stack has the correct number of entries
    if ndvi_stack.shape[0] != expected_num_dates:
        raise ValueError(f"ndvi_stack.shape[0] ({ndvi_stack.shape[0]}) does not match the expected count ({expected_num_dates}).")

    print(f"Missing dates for cluster {cluster}: {missing_dates}")

    # Second loop: perform imputation for missing data
    for target_date in missing_dates:
        t = formatted_dates.index(target_date)

        # Proceed with imputation for the missing data
        for i, j in coords_within_boundary:

            pixel_series = ndvi_stack[:, i, j]

            masked_pixel_series = pixel_series[~np.isnan(pixel_series)] # not NaN
            m = len(masked_pixel_series)

            X_train = np.array([k for k in range(len(pixel_series)) if not np.isnan(pixel_series[k])]) # not NaN
            y_train = pixel_series[~np.isnan(pixel_series)] # not NaN

            # Fit the smoothing spline with a positive smoothing factor and boundary conditions
            spline = UnivariateSpline(X_train, y_train, s=0,  k=3, ext=3)

            # Generate interpolated values only for the NaN positions
            nan_indices = np.where(np.isnan(pixel_series))[0]
            imputed_values = spline(nan_indices)

            # Ensure non-NaN values remain unchanged
            imputed_series = pixel_series.copy()
            imputed_series[nan_indices] = imputed_values

            # Only update NaN values in the NDVI stack for pixel (i, j)
            ndvi_stack[:, i, j] = imputed_series
        print(f"✅ Imputed data for missing date: {target_date}")

        # Save imputed rasters and heatmaps immediately after imputation
        if t < ndvi_stack.shape[0]:
            imputed_array = ndvi_stack[t]
            output_png = os.path.join(output_folder_heatmap.format(cluster=cluster), f"{filename_prefixes[cluster]}_{target_date}_imputed_spline3.png")
            output_raster = os.path.join(output_folder_raster.format(cluster=cluster), f"{filename_prefixes[cluster]}_{target_date}_imputed_spline3.tif")

            with rasterio.open(first_existing_file) as src:
                save_ndvi_heatmap(imputed_array, f"NDVI Raster - {cluster} ({target_date})", output_png)
                save_imputed_raster(src, imputed_array, output_raster)

            print(f"📷 Imputed Heatmap saved: {output_png}")
            print(f"🗃 Imputed Raster saved: {output_raster}")