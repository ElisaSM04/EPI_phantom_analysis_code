# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 22:29:02 2026

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

# Use the parent folder, not only 10050000, because some series are split
# between 10050000, 10050001 and 10050002.
parent_folder = r"path_to_dicom_folder"

# Repeat pairs identified from dicom_series_index.xlsx.
# GRAPPA off and SMS 2 are included here so the script can formally validate
# or exclude them if their repeat acquisitions do not match.
snr_pairs = [
    {"Protocol": "Baseline EPI", "Repeat 1": 6, "Repeat 2": 7},
    {"Protocol": "GRAPPA off", "Repeat 1": 9, "Repeat 2": 10},
    {"Protocol": "High bandwidth", "Repeat 1": 13, "Repeat 2": 14},
    {"Protocol": "Short TE", "Repeat 1": 15, "Repeat 2": 16},
    {"Protocol": "Long TE", "Repeat 1": 17, "Repeat 2": 18},
    {"Protocol": "Low flip angle", "Repeat 1": 19, "Repeat 2": 20},
    {"Protocol": "Medium flip angle", "Repeat 1": 21, "Repeat 2": 22},
    {"Protocol": "SMS 2", "Repeat 1": 23, "Repeat 2": 24},
    {"Protocol": "PA direction", "Repeat 1": 27, "Repeat 2": 28},
    {"Protocol": "Distortion correction off", "Repeat 1": 29, "Repeat 2": 30},

    # Optional additional pairs from your DICOM index.
    # Uncomment only if you want to include them in the final thesis analysis.
    # {"Protocol": "GRAPPA 3", "Repeat 1": 11, "Repeat 2": 12},
    # {"Protocol": "Medium flip angle", "Repeat 1": 21, "Repeat 2": 22},
    # {"Protocol": "SMS 3", "Repeat 1": 25, "Repeat 2": 26},
]

# Use the middle volume for all protocols.
volume_used = 15

# ROI settings used previously on the 80 x 80 EPI images.
# x = column direction, y = row direction.
base_matrix_size = 80
base_roi_centre_x = 40
base_roi_centre_y = 40
base_roi_radius = 12

# Output paths
output_excel = os.path.join(parent_folder, "SNR_all_protocols_results.xlsx")
output_plot = os.path.join(parent_folder, "SNR_all_protocols_plot.png")


# ============================================================
# FUNCTIONS
# ============================================================

def load_series_by_number(parent_folder_path, series_number):
    """
    Recursively load all DICOM files belonging to one SeriesNumber
    and sort them by InstanceNumber.
    """
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


def safe_float(value):
    """Convert DICOM values to float where possible; otherwise return np.nan."""
    try:
        return float(value)
    except Exception:
        return np.nan


def get_series_metadata(file_path):
    """Read metadata and pixel data from one DICOM volume."""
    ds = pydicom.dcmread(file_path, force=True)
    arr = ds.pixel_array

    metadata = {
        "SeriesNumber": int(getattr(ds, "SeriesNumber", -1)),
        "SeriesDescription": str(getattr(ds, "SeriesDescription", "")),
        "ProtocolName": str(getattr(ds, "ProtocolName", "")),
        "Rows": int(getattr(ds, "Rows", 0)),
        "Columns": int(getattr(ds, "Columns", 0)),
        "NumberOfFrames": int(getattr(ds, "NumberOfFrames", arr.shape[0] if arr.ndim == 3 else 1)),
        "TR_ms": safe_float(getattr(ds, "RepetitionTime", np.nan)),
        "TE_ms": safe_float(getattr(ds, "EchoTime", np.nan)),
        "FlipAngle": safe_float(getattr(ds, "FlipAngle", np.nan)),
        "SliceThickness": safe_float(getattr(ds, "SliceThickness", np.nan)),
        "PixelSpacing": str(getattr(ds, "PixelSpacing", "")),
        "PixelArrayShape": str(arr.shape),
    }

    return ds, arr, metadata


def values_match(v1, v2, tolerance=1e-6):
    """
    Compare metadata values.
    If both values are NaN, treat them as matching because the metadata
    field was not available in either repeat.
    """
    if isinstance(v1, float) and isinstance(v2, float):
        if np.isnan(v1) and np.isnan(v2):
            return True
        return abs(v1 - v2) <= tolerance

    return v1 == v2


def metadata_match(meta1, meta2):
    """
    Check whether two repeated acquisitions are suitable for subtraction SNR.
    They must have matching image shape and acquisition parameters where available.
    """
    keys_to_check = [
        "Rows",
        "Columns",
        "NumberOfFrames",
        "TR_ms",
        "TE_ms",
        "FlipAngle",
        "SliceThickness",
        "PixelSpacing",
        "PixelArrayShape",
    ]

    reasons = []
    checks = []

    for key in keys_to_check:
        v1 = meta1[key]
        v2 = meta2[key]
        match = values_match(v1, v2)
        checks.append((key, v1, v2, match))

        if not match:
            reasons.append(f"{key} differs: {v1} vs {v2}")

    all_match = len(reasons) == 0
    return all_match, "; ".join(reasons), checks


def circular_roi_mask(image_shape, centre_x, centre_y, radius):
    """Create a circular ROI mask."""
    rows, cols = image_shape
    y, x = np.ogrid[:rows, :cols]
    mask = (x - centre_x) ** 2 + (y - centre_y) ** 2 <= radius ** 2
    return mask


def t_value_95_two_sided(df):
    """Return two-sided 95% t-value."""
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df))
    except Exception:
        # Fallback table for common small samples
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
# MAIN ANALYSIS
# ============================================================

all_raw_rows = []
all_summary_rows = []
validation_rows = []

for pair in snr_pairs:
    protocol = pair["Protocol"]
    series1 = pair["Repeat 1"]
    series2 = pair["Repeat 2"]

    print(f"\nProcessing: {protocol} | Series {series1} and {series2}")

    try:
        files1 = load_series_by_number(parent_folder, series1)
        files2 = load_series_by_number(parent_folder, series2)

        print(f"Repeat 1 files found: {len(files1)}")
        print(f"Repeat 2 files found: {len(files2)}")

        if len(files1) < volume_used or len(files2) < volume_used:
            validation_rows.append({
                "Protocol": protocol,
                "Repeat 1 series": series1,
                "Repeat 2 series": series2,
                "Included": "No",
                "Reason": f"Fewer than {volume_used} volumes/files available"
            })
            continue

        # Select volume 15
        ds1, arr1, meta1 = get_series_metadata(files1[volume_used - 1][1])
        ds2, arr2, meta2 = get_series_metadata(files2[volume_used - 1][1])

        is_match, mismatch_reason, checks = metadata_match(meta1, meta2)

        if not is_match:
            validation_rows.append({
                "Protocol": protocol,
                "Repeat 1 series": series1,
                "Repeat 2 series": series2,
                "Included": "No",
                "Reason": mismatch_reason
            })
            print(f"Excluded: {mismatch_reason}")
            continue

        # Pixel array should be volume data, e.g. (25, 80, 80)
        if arr1.ndim != 3 or arr2.ndim != 3:
            validation_rows.append({
                "Protocol": protocol,
                "Repeat 1 series": series1,
                "Repeat 2 series": series2,
                "Included": "No",
                "Reason": "Pixel array was not 3D volume data"
            })
            continue

        n_slices = arr1.shape[0]

        # Select five central slices.
        # For 25 slices, this gives 11, 12, 13, 14, 15.
        central_slice = int(np.ceil(n_slices / 2))
        slices_used = list(range(central_slice - 2, central_slice + 3))

        # Scale ROI if matrix size is not 80 x 80
        rows, cols = arr1[0].shape
        scale_x = cols / base_matrix_size
        scale_y = rows / base_matrix_size

        roi_centre_x = int(round(base_roi_centre_x * scale_x))
        roi_centre_y = int(round(base_roi_centre_y * scale_y))
        roi_radius = int(round(base_roi_radius * min(scale_x, scale_y)))

        mask = circular_roi_mask(arr1[0].shape, roi_centre_x, roi_centre_y, roi_radius)
        roi_area = int(np.sum(mask))

        snr_values = []

        for slice_number in slices_used:
            slice_index = slice_number - 1

            img1 = arr1[slice_index].astype(float)
            img2 = arr2[slice_index].astype(float)

            subtraction_image = img1 - img2

            mean_signal_1 = np.mean(img1[mask])
            mean_signal_2 = np.mean(img2[mask])
            mean_signal_used = np.mean([mean_signal_1, mean_signal_2])
            sd_subtraction = np.std(subtraction_image[mask], ddof=1)

            snr = (np.sqrt(2) * mean_signal_used) / sd_subtraction
            snr_values.append(snr)

            all_raw_rows.append({
                "Protocol": protocol,
                "Repeat 1 series number": series1,
                "Repeat 2 series number": series2,
                "Repeat 1 series name": meta1["SeriesDescription"],
                "Repeat 2 series name": meta2["SeriesDescription"],
                "Volume used": volume_used,
                "Slice used": slice_number,
                "Rows": rows,
                "Columns": cols,
                "Number of slices": n_slices,
                "ROI centre x": roi_centre_x,
                "ROI centre y": roi_centre_y,
                "ROI radius / pixels": roi_radius,
                "ROI area / pixels": roi_area,
                "Mean signal repeat 1": mean_signal_1,
                "Mean signal repeat 2": mean_signal_2,
                "Mean signal used": mean_signal_used,
                "SD subtraction image": sd_subtraction,
                "SNR": snr,
                "Notes": "Two-image subtraction method"
            })

        snr_values = np.array(snr_values)
        n = len(snr_values)
        mean_snr = np.mean(snr_values)
        sd_snr = np.std(snr_values, ddof=1)
        se_snr = sd_snr / np.sqrt(n)
        t_val = t_value_95_two_sided(n - 1)
        ci_half_width = t_val * se_snr

        all_summary_rows.append({
            "Protocol": protocol,
            "Repeat 1 series number": series1,
            "Repeat 2 series number": series2,
            "Number of slices analysed": n,
            "Mean SNR": mean_snr,
            "Standard deviation": sd_snr,
            "Standard error": se_snr,
            "t-value": t_val,
            "95% CI half-width": ci_half_width,
            "Lower 95% CI": mean_snr - ci_half_width,
            "Upper 95% CI": mean_snr + ci_half_width,
            "Reported SNR": f"{mean_snr:.1f} ± {ci_half_width:.1f}",
            "Percentage change from baseline": np.nan,
            "Notes": "Included in SNR comparison"
        })

        validation_rows.append({
            "Protocol": protocol,
            "Repeat 1 series": series1,
            "Repeat 2 series": series2,
            "Included": "Yes",
            "Reason": "Matching repeat acquisitions"
        })

        print(f"Included. Mean SNR = {mean_snr:.1f} ± {ci_half_width:.1f}")

    except Exception as e:
        validation_rows.append({
            "Protocol": protocol,
            "Repeat 1 series": series1,
            "Repeat 2 series": series2,
            "Included": "No",
            "Reason": str(e)
        })
        print(f"Error/excluded: {e}")


# ============================================================
# SUMMARY TABLE AND PERCENTAGE CHANGE FROM BASELINE
# ============================================================

df_validation = pd.DataFrame(validation_rows)
df_raw = pd.DataFrame(all_raw_rows)
df_summary = pd.DataFrame(all_summary_rows)

if not df_summary.empty:
    baseline_rows = df_summary[df_summary["Protocol"] == "Baseline EPI"]

    if len(baseline_rows) == 1:
        baseline_snr = float(baseline_rows["Mean SNR"].iloc[0])
        df_summary["Percentage change from baseline"] = (
            (df_summary["Mean SNR"] - baseline_snr) / baseline_snr
        ) * 100


# ============================================================
# SAVE EXCEL OUTPUT
# ============================================================

with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    df_validation.to_excel(writer, sheet_name="SNR_pair_validation", index=False)
    df_raw.to_excel(writer, sheet_name="SNR_raw_measurements", index=False)
    df_summary.to_excel(writer, sheet_name="SNR_summary", index=False)

print("\nSaved SNR workbook to:")
print(output_excel)

print("\nPair validation:")
print(df_validation)

print("\nSNR summary:")
print(df_summary)


# ============================================================
# SAVE SUMMARY PLOT
# ============================================================

if not df_summary.empty:
    plt.figure(figsize=(9, 5))
    plt.bar(
        df_summary["Protocol"],
        df_summary["Mean SNR"],
        yerr=df_summary["95% CI half-width"],
        capsize=5
    )
    plt.ylabel("Mean SNR")
    plt.xlabel("Protocol")
    plt.title("Mean SNR across analysed EPI protocols")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    plt.show()

    print("\nSaved SNR plot to:")
    print(output_plot)