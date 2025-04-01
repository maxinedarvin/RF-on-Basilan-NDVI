import rasterio
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import glob
import os
from rasterio.features import geometry_mask
from matplotlib.colors import LinearSegmentedColormap
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.ndimage import generic_filter
from datetime import datetime, timedelta

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
folders = {
    "02-22": "/Darvin/BCC 02-22 NDVI Filtered/",
    "04-03": "/Darvin/BCC 04-03 NDVI Filtered/",
    "08-07": "/Darvin/BCC 08-07 NDVI Filtered/",
    "08-07 mini": "/Darvin/BCC 08-07 Mini NDVI Filtered/"
}

boundaries = {
    "02-22": "/Darvin/Cluster Shapefiles/0222 pixel shape.shp",
    "04-03": "/Darvin/Cluster Shapefiles/0403 pixel shape.shp",
    "08-07": "/Darvin/Cluster Shapefiles/0807 pixel shape.shp",
    "08-07 mini": "/Darvin/Cluster Shapefiles/0807 pixel shape mini.shp"
}

filename_prefixes = {
    "02-22": "0222ndvi",
    "04-03": "0403ndvi",
    "08-07": "0807ndvi",
    "08-07": "m0807ndvi"
}

def extract_neighborhood(array, boundary_mask, size=5):
    def process_window(window):
        valid_pixels = window[~np.isnan(window)]  # Keep only valid NDVI pixels
        if len(valid_pixels) < 1:  
            return np.nan  # If no valid pixels, return NaN
        return np.nanmean(valid_pixels)  # Return mean of valid pixels
    
    # Apply filter to get neighborhood values inside boundary
    neighborhood_values = generic_filter(array, process_window, size=size, mode='constant', cval=np.nan)

    # Apply filter to count valid pixels in the neighborhood
    count_valid = generic_filter(boundary_mask.astype(float), np.sum, size=size, mode='constant', cval=0)

    return neighborhood_values, count_valid  # Return values and valid pixel counts

# Generate all dates from 01-5-2018 to 12-29-2024 (5-day interval)
start_date = datetime(2018, 1, 5)
end_date = datetime(2024, 12, 29)
date_list = [start_date + timedelta(days=x) for x in range(0, (end_date - start_date).days + 1, 5)]

# Convert date format to match filenames (e.g., 20180130)
formatted_dates = [date.strftime("%Y%m%d") for date in date_list]

# Store metadata
for target_date in formatted_dates:
    for area, folder_path in folders.items():
        print(f"\n🔍 Processing raster in: {folder_path} for date {target_date}")

        raster_files = glob.glob(os.path.join(folder_path, f"*{target_date}.tif"))
        if not raster_files:
            print(f"⚠ No raster file found for {target_date} in {area}. Skipping...")
            continue

        raster_file = raster_files[0]
        print(f"✅ Found raster file: {raster_file}")

        with rasterio.open(raster_file) as src:
            ndvi_array = src.read(1).astype(float)
            ndvi_array = np.where(ndvi_array == 0, np.nan, ndvi_array)  # Replace 0s with NaN

        # Load boundary shapefile
        boundary_file = boundaries[area]
        boundary = gpd.read_file(boundary_file)
        boundary = boundary.to_crs(src.crs)

        # Generate rasterized mask for exact boundary
        boundary_mask = geometry_mask(
            [geom for geom in boundary.geometry],
            transform=src.transform,
            invert=True,
            out_shape=ndvi_array.shape
        )

        # Apply mask to the raster
        ndvi_within_boundary = np.where(boundary_mask, ndvi_array, np.nan)

        # Count Total Pixels & NaN Pixels Within Boundary
        total_pixels_within_boundary = np.count_nonzero(boundary_mask)
        nan_pixels_within_boundary = np.count_nonzero(np.isnan(ndvi_within_boundary) & boundary_mask)
        
        print(f"🟢 {area} ({target_date}): Total Pixels Within Boundary = {total_pixels_within_boundary}")
        print(f"🟡 {area} ({target_date}): NaN Pixels Within Boundary Before Imputation = {nan_pixels_within_boundary}")

        initial_nan_percentage = (nan_pixels_within_boundary / total_pixels_within_boundary) * 100

        output_folder_raster = f"/Darvin/BCC {area} NDVI FImputed/"
        output_folder_heatmap = f"/Darvin/{area} NDVI FImputed Heatmaps/"
        os.makedirs(output_folder_raster, exist_ok=True)
        os.makedirs(output_folder_heatmap, exist_ok=True)

        filename_prefix = filename_prefixes[area]  
        output_png = os.path.join(output_folder_heatmap, f"{filename_prefix}_{target_date}_imputed.png")
        output_raster = os.path.join(output_folder_raster, f"{filename_prefix}_{target_date}_imputed.tif")

        # If No NaN Pixels Exist, Save the Unmodified Image
        if nan_pixels_within_boundary == 0:
            print(f"✅ {area} ({target_date}): No NaN pixels detected within boundary. Saving unmodified heatmap and raster.")
            save_ndvi_heatmap(ndvi_within_boundary, f"NDVI Raster - {area} ({target_date}, {initial_nan_percentage:.1f}% Imputed)", output_png)
            save_imputed_raster(src, ndvi_within_boundary, output_raster)
            continue

        # Spatial Imputation (Within boundary)
        iteration = 0
        valid_pixel_requirements = [8, 6, 5]  # 30%, 25%, 20%
        current_requirement_index = 0
        previous_nan_count = nan_pixels_within_boundary
        
        while nan_pixels_within_boundary > 0:
            iteration += 1
            print(f"🟠 Iteration {iteration}: Starting imputation with valid pixel requirement {valid_pixel_requirements[current_requirement_index]}...")

            neighborhood_values, count_valid_pixels = extract_neighborhood(ndvi_within_boundary, boundary_mask, size=5)
            valid_rows = count_valid_pixels >= valid_pixel_requirements[current_requirement_index]
            valid_indices = np.where(valid_rows & np.isnan(ndvi_within_boundary))
            valid_values = neighborhood_values[valid_indices].reshape(-1, 1)

            if len(valid_values) > 0:
                imputer = IterativeImputer(max_iter=10, random_state=42)
                imputed_values = imputer.fit_transform(valid_values)
                ndvi_within_boundary[valid_indices] = imputed_values.flatten()

            nan_pixels_within_boundary = np.count_nonzero(np.isnan(ndvi_within_boundary) & boundary_mask)
            print(f"🔵 Iteration {iteration}: NaN Pixels Remaining = {nan_pixels_within_boundary}")

            if nan_pixels_within_boundary == previous_nan_count:
                # No change in NaN count, adjust valid pixel requirement
                if current_requirement_index < len(valid_pixel_requirements) - 1:
                    current_requirement_index += 1
                    print(f"⚠ Iteration {iteration}: No change in NaN count. Reducing valid pixel requirement to {valid_pixel_requirements[current_requirement_index]}.")
                else:
                    # If already at the lowest requirement, break to avoid infinite loop
                    print(f"❌ Iteration {iteration}: No further reduction possible. Exiting loop.")
                    break
            previous_nan_count = nan_pixels_within_boundary

        # Final Clipping: Apply Boundary Mask Again
        ndvi_within_boundary = np.where(boundary_mask, ndvi_within_boundary, np.nan)

        # Calculate the percentage of imputed pixels
        imputed_pixels_count = np.count_nonzero(~np.isnan(ndvi_within_boundary) & np.isnan(ndvi_array) & boundary_mask)
        percentage_imputed = (imputed_pixels_count / total_pixels_within_boundary) * 100

        # Save heatmap and raster
        save_ndvi_heatmap(ndvi_within_boundary, f"NDVI Raster - {area} ({target_date}, {initial_nan_percentage:.1f}% Imputed)", output_png)
        save_imputed_raster(src, ndvi_within_boundary, output_raster)
        
        print(f"📷 Imputed Heatmap saved: {output_png}")
        print(f"🗃 Imputed Raster saved: {output_raster}")