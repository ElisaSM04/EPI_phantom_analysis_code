# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 01:26:09 2026

@author: Elisa Spiteri
"""
import os
import pydicom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# USER SETTINGS
# ============================================================

parent_folder = r"path_to_dicom_folder"

time_series_number = 8
volume_used_start = 1
volume_used_end = 200

# Set this to 0 if using all volumes.
# If later you decide to discard first 5 volumes, change this to 5.
discard_initial_volumes = 0

# ROI settings for 80 x 80 EPI images
roi_centre_x = 40
roi_centre_y = 40
roi_radius = 12

output_excel = os.path.join(parent_folder, "Temporal_stability_results.xlsx")
output_timecourse_plot = os.path.join(parent_folder, "Temporal_stability_timecourse.png")
output_roi_plot = os.path.join(parent_folder, "Temporal_stability_ROI_placement.png")


# ============================================================
# FUNCTIONS
# ============================================================

def load_series_by_number(parent_folder_path, series_number):
    selected_files = []

    for root, dirs, files in os.walk(parent_folder_path):
        for file in files:
            file_path = os.path.join(root, file)

            try:
                ds = pydicom.dcmread(file_path, stop_before_pixels=True, force=True)
                current_series = int(getattr(ds, "SeriesNumber", -1))

                if current_series == series_number:
                    instance_number = int(getattr(ds, "InstanceNumber", 0))
                    selected_files.append((instance_number, file_path))

            except Exception:
                pass

    selected_files = sorted(selected_files, key=lambda x: x[0])

    if len(selected_files) == 0:
        raise ValueError(f"No files found for SeriesNumber {series_number}")

    return selected_files


def circular_roi_mask(image_shape, centre_x, centre_y, radius):
    rows, cols = image_shape
    y, x = np.ogrid[:rows, :cols]
    mask = (x - centre_x) ** 2 + (y - centre_y) ** 2 <= radius ** 2
    return mask


def t_value_95_two_sided(df):
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df))
    except Exception:
        t_values = {
            1: 12.706,
            2: 4.303,
            3: 3.182,
            4: 2.776,
            5: 2.571,
            6: 2.447,
            7: 2.365,
            8: 2.306,
            9: 2.262,
            10: 2.228,
        }
        return t_values.get(df, 1.96)


# ============================================================
# LOAD TIME SERIES
# ============================================================

files = load_series_by_number(parent_folder, time_series_number)

print("Number of files found:", len(files))

if len(files) < volume_used_end:
    raise ValueError("Fewer than 200 volumes were found.")

# Read first volume to check shape
first_ds = pydicom.dcmread(files[0][1], force=True)
first_arr = first_ds.pixel_array.astype(float)

print("First volume shape:", first_arr.shape)

if first_arr.ndim != 3:
    raise ValueError("Expected each DICOM file to contain a 3D volume, e.g. (25, 80, 80).")

n_slices = first_arr.shape[0]
central_slice = int(np.ceil(n_slices / 2))
slices_used = list(range(central_slice - 2, central_slice + 3))

print("Slices used:", slices_used)

mask = circular_roi_mask(first_arr[0].shape, roi_centre_x, roi_centre_y, roi_radius)
roi_area = int(np.sum(mask))

# Volumes used
volume_numbers_all = np.arange(1, len(files) + 1)
analysis_files = files[discard_initial_volumes:volume_used_end]
analysis_volume_numbers = volume_numbers_all[discard_initial_volumes:volume_used_end]

print("Volumes used for analysis:", analysis_volume_numbers[0], "to", analysis_volume_numbers[-1])


# ============================================================
# CALCULATE TEMPORAL STABILITY
# ============================================================

slice_metric_rows = []
timecourse_rows = []

all_slice_timecourses = []

for slice_number in slices_used:
    slice_index = slice_number - 1
    signal_timecourse = []

    for volume_number, file_item in zip(analysis_volume_numbers, analysis_files):
        ds = pydicom.dcmread(file_item[1], force=True)
        arr = ds.pixel_array.astype(float)

        img = arr[slice_index]
        mean_signal = np.mean(img[mask])
        signal_timecourse.append(mean_signal)

    signal_timecourse = np.array(signal_timecourse)
    all_slice_timecourses.append(signal_timecourse)

    mean_signal = np.mean(signal_timecourse)
    sd_signal = np.std(signal_timecourse, ddof=1)

    temporal_snr = mean_signal / sd_signal
    cv_percent = (sd_signal / mean_signal) * 100

    # Linear drift across the analysed time series
    x = np.arange(1, len(signal_timecourse) + 1)
    slope, intercept = np.polyfit(x, signal_timecourse, 1)
    fitted_start = slope * x[0] + intercept
    fitted_end = slope * x[-1] + intercept
    drift_percent = ((fitted_end - fitted_start) / mean_signal) * 100

    slice_metric_rows.append({
        "Series number": time_series_number,
        "Slice used": slice_number,
        "Number of volumes analysed": len(signal_timecourse),
        "ROI centre x": roi_centre_x,
        "ROI centre y": roi_centre_y,
        "ROI radius / pixels": roi_radius,
        "ROI area / pixels": roi_area,
        "Mean signal": mean_signal,
        "Temporal SD": sd_signal,
        "Temporal SNR": temporal_snr,
        "Coefficient of variation (%)": cv_percent,
        "Signal drift (%)": drift_percent,
        "Notes": "Temporal metrics calculated from ROI mean signal time course"
    })

    for volume_number, signal in zip(analysis_volume_numbers, signal_timecourse):
        timecourse_rows.append({
            "Slice used": slice_number,
            "Volume number": volume_number,
            "Mean ROI signal": signal,
            "Normalised signal (%)": (signal / mean_signal) * 100
        })


df_slice_metrics = pd.DataFrame(slice_metric_rows)
df_timecourse = pd.DataFrame(timecourse_rows)

# Mean time course across the five slices
all_slice_timecourses = np.array(all_slice_timecourses)
mean_timecourse = np.mean(all_slice_timecourses, axis=0)
normalised_mean_timecourse = (mean_timecourse / np.mean(mean_timecourse)) * 100

df_mean_timecourse = pd.DataFrame({
    "Volume number": analysis_volume_numbers,
    "Mean signal across five slices": mean_timecourse,
    "Normalised mean signal (%)": normalised_mean_timecourse
})


# ============================================================
# SUMMARY WITH 95% CI ACROSS FIVE SLICES
# ============================================================

summary_rows = []

for metric_name, column_name in [
    ("Temporal SNR", "Temporal SNR"),
    ("Coefficient of variation (%)", "Coefficient of variation (%)"),
    ("Signal drift (%)", "Signal drift (%)")
]:
    values = df_slice_metrics[column_name].to_numpy()

    n = len(values)
    mean_value = np.mean(values)
    sd_value = np.std(values, ddof=1)
    se_value = sd_value / np.sqrt(n)
    t_val = t_value_95_two_sided(n - 1)
    ci_half_width = t_val * se_value

    summary_rows.append({
        "Metric": metric_name,
        "Number of slices analysed": n,
        "Mean value": mean_value,
        "Standard deviation": sd_value,
        "Standard error": se_value,
        "t-value": t_val,
        "95% CI half-width": ci_half_width,
        "Lower 95% CI": mean_value - ci_half_width,
        "Upper 95% CI": mean_value + ci_half_width,
        "Reported value": f"{mean_value:.2f} ± {ci_half_width:.2f}",
        "Notes": "Mean and 95% CI calculated across five central slices"
    })

df_summary = pd.DataFrame(summary_rows)


# ============================================================
# SAVE EXCEL
# ============================================================

with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    df_slice_metrics.to_excel(writer, sheet_name="Slice_metrics", index=False)
    df_timecourse.to_excel(writer, sheet_name="Slice_timecourses", index=False)
    df_mean_timecourse.to_excel(writer, sheet_name="Mean_timecourse", index=False)
    df_summary.to_excel(writer, sheet_name="Temporal_summary", index=False)

print("\nSaved temporal stability workbook to:")
print(output_excel)

print("\nTemporal summary:")
print(df_summary)


# ============================================================
# SAVE TIME COURSE PLOT
# ============================================================

plt.figure(figsize=(9, 5))
plt.plot(df_mean_timecourse["Volume number"], df_mean_timecourse["Normalised mean signal (%)"])
plt.xlabel("Volume number")
plt.ylabel("Normalised mean signal (%)")
plt.title("Temporal stability of the 200-volume EPI time series")
plt.tight_layout()
plt.savefig(output_timecourse_plot, dpi=300)
plt.show()

print("\nSaved temporal stability time-course plot to:")
print(output_timecourse_plot)


# ============================================================
# SAVE ROI PLACEMENT FIGURE
# ============================================================

plt.figure(figsize=(6, 6))
plt.imshow(first_arr[central_slice - 1], cmap="gray")

circle = plt.Circle(
    (roi_centre_x, roi_centre_y),
    roi_radius,
    fill=False,
    edgecolor="yellow",
    linewidth=2
)

plt.gca().add_patch(circle)
plt.text(
    roi_centre_x + roi_radius + 1,
    roi_centre_y,
    "Temporal ROI",
    color="yellow",
    fontsize=12,
    fontweight="bold",
    bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=1)
)

plt.title("Temporal stability ROI placement: time-series EPI")
plt.axis("off")
plt.tight_layout()
plt.savefig(output_roi_plot, dpi=300)
plt.show()

print("\nSaved temporal stability ROI placement plot to:")
print(output_roi_plot)