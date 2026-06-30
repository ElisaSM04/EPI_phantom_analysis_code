# -*- coding: utf-8 -*-

"""
Automated multi-slice, multi-direction geometric distortion analysis.

This script:
1. Loads DICOM series by SeriesNumber.
2. Segments the phantom/fluid boundary automatically.
3. Measures directional diameters at 0, 30, 60, 90, 120 and 150 degrees.
4. Uses the T1-weighted image as the geometric reference.
5. Calculates mean distortion and 95% confidence intervals.
6. Saves QC segmentation images and one baseline EPI measurement-lines figure.

Author: Elisa Spiteri
"""

import os
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.stats import t


# ============================================================
# USER SETTINGS
# ============================================================

parent_folder = r"path_to_dicom_folder"

# Check this in series_index.xlsx.
# From your current setup this appears to be the T1_REFERENCE series.
t1_series_number = 5

# Only the EPI protocols included in the final geometric distortion analysis are listed below.
# Large FOV, GRAPPA 3, SMS 3, pilot scans, scout scans, repeat scans and the time-series scan
# are intentionally excluded because they were not part of the final primary comparison.
epi_series = [
    {"Protocol": "Baseline EPI", "Series": 6},
    {"Protocol": "GRAPPA off", "Series": 9},
    {"Protocol": "High bandwidth", "Series": 13},
    {"Protocol": "Short TE", "Series": 15},
    {"Protocol": "Long TE", "Series": 17},
    {"Protocol": "Low flip angle", "Series": 19},
    {"Protocol": "Medium flip angle", "Series": 21},
    {"Protocol": "SMS 2", "Series": 23},
    {"Protocol": "PA direction", "Series": 27},
    {"Protocol": "Distortion correction off", "Series": 29},
]

# EPI volume used for geometric analysis.
epi_volume_index = 15

# T1 has one volume, so use volume 1.
t1_volume_index = 1

# Five central slices and six directions.
number_of_central_slices = 5
directions_deg = [0, 30, 60, 90, 120, 150]

# Manual pixel spacing override based on your recorded scan log.
# This is necessary because the exported DICOM PixelSpacing was not read correctly
# for the EPI images in the previous output.
spacing_override_mm = {
    5: (1.0, 1.0),   # T1 reference: 1 x 1 mm in-plane resolution

    6: (3.0, 3.0),   # Baseline EPI
    9: (3.0, 3.0),   # GRAPPA off
    13: (3.0, 3.0),  # High bandwidth
    15: (3.0, 3.0),  # Short TE
    17: (3.0, 3.0),  # Long TE
    19: (3.0, 3.0),  # Low flip angle
    21: (3.0, 3.0),  # Medium flip angle
    23: (3.0, 3.0),  # SMS 2
    27: (3.0, 3.0),  # PA direction
    29: (3.0, 3.0),  # Distortion correction off
}

output_excel = os.path.join(parent_folder, "Geometric_distortion_multislice_results_corrected.xlsx")
output_index = os.path.join(parent_folder, "series_index.xlsx")
qc_folder = os.path.join(parent_folder, "geometric_distortion_QC_images_corrected")
os.makedirs(qc_folder, exist_ok=True)


# ============================================================
# DICOM LOADING FUNCTIONS
# ============================================================

def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def load_series_files(parent_folder_path, series_number):
    selected_files = []

    for root, _, files in os.walk(parent_folder_path):
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


def create_series_index(parent_folder_path):
    rows = []

    for root, _, files in os.walk(parent_folder_path):
        for file in files:
            file_path = os.path.join(root, file)

            try:
                ds = pydicom.dcmread(file_path, stop_before_pixels=True, force=True)

                series_number_raw = getattr(ds, "SeriesNumber", np.nan)

                try:
                    series_number_numeric = int(series_number_raw)
                except Exception:
                    series_number_numeric = np.nan

                rows.append({
                    "SeriesNumber": series_number_numeric,
                    "SeriesDescription": str(getattr(ds, "SeriesDescription", "N/A")),
                    "ProtocolName": str(getattr(ds, "ProtocolName", "N/A")),
                    "InstanceNumber": getattr(ds, "InstanceNumber", "N/A"),
                    "Rows": getattr(ds, "Rows", "N/A"),
                    "Columns": getattr(ds, "Columns", "N/A"),
                    "PixelSpacing_from_DICOM": str(getattr(ds, "PixelSpacing", "N/A")),
                    "SliceThickness": getattr(ds, "SliceThickness", "N/A"),
                    "TR_ms": getattr(ds, "RepetitionTime", "N/A"),
                    "TE_ms": getattr(ds, "EchoTime", "N/A"),
                    "FlipAngle": getattr(ds, "FlipAngle", "N/A"),
                    "FilePath": file_path
                })

            except Exception:
                pass

    df_index = pd.DataFrame(rows)

    if not df_index.empty:
        df_summary = (
            df_index
            .dropna(subset=["SeriesNumber"])
            .groupby(["SeriesNumber", "SeriesDescription", "ProtocolName"])
            .size()
            .reset_index(name="Number_of_files")
            .sort_values("SeriesNumber")
        )

        with pd.ExcelWriter(output_index, engine="openpyxl") as writer:
            df_summary.to_excel(writer, sheet_name="Series_summary", index=False)
            df_index.to_excel(writer, sheet_name="All_files", index=False)

        print(f"Saved DICOM series index to: {output_index}")

    return df_index


def get_spacing_for_series(ds, series_number):
    """
    Returns row and column spacing in mm.

    Manual override is used where available because the EPI exported DICOM
    spacing was not reliable in the previous output.
    """
    if series_number in spacing_override_mm:
        row_spacing, col_spacing = spacing_override_mm[series_number]
        spacing_source = "manual_override_from_scan_log"
    else:
        pixel_spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
        row_spacing = safe_float(pixel_spacing[0])
        col_spacing = safe_float(pixel_spacing[1])
        spacing_source = "DICOM_PixelSpacing"

    return row_spacing, col_spacing, spacing_source


def load_image_stack(parent_folder_path, series_number, volume_index=1):
    files = load_series_files(parent_folder_path, series_number)

    # Read the selected file if each DICOM file contains a 3D volume.
    selected_index = min(volume_index - 1, len(files) - 1)
    ds = pydicom.dcmread(files[selected_index][1], force=True)
    arr = ds.pixel_array.astype(float)

    row_spacing, col_spacing, spacing_source = get_spacing_for_series(ds, series_number)
    dicom_pixel_spacing = str(getattr(ds, "PixelSpacing", "N/A"))

    # If the file contains a full volume, arr should be 3D: slices x rows x columns.
    if arr.ndim == 3:
        stack = arr

    # If the series is exported as one 2D slice per file, stack the files.
    elif arr.ndim == 2:
        slice_arrays = []
        for _, path in files:
            ds_slice = pydicom.dcmread(path, force=True)
            slice_arrays.append(ds_slice.pixel_array.astype(float))
        stack = np.stack(slice_arrays, axis=0)

    else:
        raise ValueError(f"Unexpected pixel array shape for Series {series_number}: {arr.shape}")

    metadata = {
        "SeriesNumber": series_number,
        "SeriesDescription": str(getattr(ds, "SeriesDescription", "")),
        "ProtocolName": str(getattr(ds, "ProtocolName", "")),
        "DICOM_PixelSpacing": dicom_pixel_spacing,
        "Spacing_source_used": spacing_source,
        "row_spacing_mm_used": row_spacing,
        "col_spacing_mm_used": col_spacing,
        "ArrayShape": str(stack.shape)
    }

    return stack, row_spacing, col_spacing, metadata


def central_slice_indices(n_slices, number_of_slices=5):
    central_slice_one_based = int(np.ceil(n_slices / 2))
    start_one_based = central_slice_one_based - number_of_slices // 2
    end_one_based = start_one_based + number_of_slices - 1

    if start_one_based < 1:
        start_one_based = 1
        end_one_based = number_of_slices

    if end_one_based > n_slices:
        end_one_based = n_slices
        start_one_based = n_slices - number_of_slices + 1

    return list(range(start_one_based, end_one_based + 1))


# ============================================================
# IMAGE SEGMENTATION AND DIAMETER MEASUREMENT
# ============================================================

def otsu_threshold(image):
    values = image[np.isfinite(image)].ravel()

    if values.size == 0:
        raise ValueError("Image contained no finite values.")

    counts, bin_edges = np.histogram(values, bins=256)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    total = values.size
    sum_total = np.sum(counts * bin_centres)

    weight_background = 0
    sum_background = 0
    max_variance = -1
    threshold = bin_centres[0]

    for i in range(256):
        weight_background += counts[i]

        if weight_background == 0:
            continue

        weight_foreground = total - weight_background

        if weight_foreground == 0:
            break

        sum_background += counts[i] * bin_centres[i]

        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        between_class_variance = (
            weight_background
            * weight_foreground
            * (mean_background - mean_foreground) ** 2
        )

        if between_class_variance > max_variance:
            max_variance = between_class_variance
            threshold = bin_centres[i]

    return threshold


def segment_phantom(image):
    img = image.astype(float)

    # Otsu threshold separates bright phantom signal from dark background.
    threshold = otsu_threshold(img)
    mask = img > threshold

    # Clean mask.
    mask = ndi.binary_closing(mask, iterations=2)
    mask = ndi.binary_fill_holes(mask)

    # Keep the largest connected component.
    labelled, number = ndi.label(mask)

    if number == 0:
        raise ValueError("No phantom component detected.")

    areas = ndi.sum(mask, labelled, index=np.arange(1, number + 1))
    largest_label = int(np.argmax(areas) + 1)
    mask = labelled == largest_label

    return mask, threshold


def directional_diameter_mm(mask, row_spacing_mm, col_spacing_mm, angle_deg):
    y_pixels, x_pixels = np.nonzero(mask)

    if len(x_pixels) < 10:
        raise ValueError("Too few pixels in phantom mask.")

    # Convert pixel coordinates to physical coordinates in mm.
    x_mm = x_pixels * col_spacing_mm
    y_mm = y_pixels * row_spacing_mm

    theta = np.deg2rad(angle_deg)

    # Directional projection gives the object width at the selected direction.
    projection = x_mm * np.cos(theta) + y_mm * np.sin(theta)

    diameter = projection.max() - projection.min()

    return diameter


def save_qc_image(image, mask, title, output_path):
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap="gray")
    plt.contour(mask, levels=[0.5], colors=["purple"], linewidths=1)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def line_endpoints_in_mask(mask, angle_deg, centre_xy, n_samples=4000):
    """
    Finds the two endpoints of a line passing through the centre of the segmented
    phantom mask at the requested angle.

    This is used to illustrate the measurement directions for Figure 3.1.
    The numerical diameter calculation is still performed by directional projection
    in directional_diameter_mm().
    """
    rows, cols = mask.shape
    cx, cy = centre_xy

    theta = np.deg2rad(angle_deg)

    # Use a line long enough to cross the full image.
    max_len = max(rows, cols) * 2
    line_parameter = np.linspace(-max_len, max_len, n_samples)

    x_float = cx + line_parameter * np.cos(theta)
    y_float = cy + line_parameter * np.sin(theta)

    x_index = np.round(x_float).astype(int)
    y_index = np.round(y_float).astype(int)

    valid = (
        (x_index >= 0) &
        (x_index < cols) &
        (y_index >= 0) &
        (y_index < rows)
    )

    x_float = x_float[valid]
    y_float = y_float[valid]
    x_index = x_index[valid]
    y_index = y_index[valid]

    inside = mask[y_index, x_index]

    if not np.any(inside):
        return None, None

    inside_indices = np.where(inside)[0]
    first_index = inside_indices[0]
    last_index = inside_indices[-1]

    point_1 = (x_float[first_index], y_float[first_index])
    point_2 = (x_float[last_index], y_float[last_index])

    return point_1, point_2


def save_measurement_lines_image(image, mask, directions_deg, title, output_path):
    """
    Saves one illustrative figure showing:
    - the phantom image
    - the segmentation boundary
    - the centre of the segmented phantom
    - the six directional measurement lines
    """
    centre_y, centre_x = ndi.center_of_mass(mask)
    centre_xy = (centre_x, centre_y)

    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap="gray")
    plt.contour(mask, levels=[0.5], colors=["purple"], linewidths=1.2)

    # Centre point.
    plt.plot(centre_x, centre_y, "o", markersize=4)

    for angle in directions_deg:
        point_1, point_2 = line_endpoints_in_mask(
            mask=mask,
            angle_deg=angle,
            centre_xy=centre_xy
        )

        if point_1 is not None and point_2 is not None:
            plt.plot(
                [point_1[0], point_2[0]],
                [point_1[1], point_2[1]],
                linewidth=1.3,
                label=f"{angle}°"
            )

    plt.title(title)
    plt.axis("off")
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0,
        fontsize=8
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# MAIN ANALYSIS
# ============================================================

create_series_index(parent_folder)

raw_rows = []
summary_rows = []
metadata_rows = []

# ------------------------------------------------------------
# Analyse T1 reference
# ------------------------------------------------------------

t1_stack, t1_row_spacing, t1_col_spacing, t1_meta = load_image_stack(
    parent_folder,
    t1_series_number,
    volume_index=t1_volume_index
)

metadata_rows.append(t1_meta)

t1_slices = central_slice_indices(t1_stack.shape[0], number_of_central_slices)

print("\nT1 reference metadata:")
print(t1_meta)
print("T1 central slices used:", t1_slices)

t1_measurements = []

for slice_number in t1_slices:
    slice_index = slice_number - 1
    image = t1_stack[slice_index]

    mask, threshold = segment_phantom(image)

    if slice_number == t1_slices[len(t1_slices) // 2]:
        save_qc_image(
            image,
            mask,
            f"T1 reference segmentation, slice {slice_number}",
            os.path.join(qc_folder, "T1_reference_QC.png")
        )

    for angle in directions_deg:
        diameter = directional_diameter_mm(
            mask,
            t1_row_spacing,
            t1_col_spacing,
            angle
        )

        t1_measurements.append({
            "Reference": "T1",
            "SeriesNumber": t1_series_number,
            "Slice": slice_number,
            "Direction_deg": angle,
            "Diameter_mm": diameter,
            "T1_row_spacing_mm": t1_row_spacing,
            "T1_col_spacing_mm": t1_col_spacing
        })

t1_df = pd.DataFrame(t1_measurements)

# Mean T1 reference diameter for each direction.
t1_reference_by_direction = (
    t1_df
    .groupby("Direction_deg")["Diameter_mm"]
    .mean()
    .to_dict()
)

print("\nT1 reference diameter by direction:")
for angle, value in t1_reference_by_direction.items():
    print(f"{angle} degrees: {value:.2f} mm")


# ------------------------------------------------------------
# Analyse EPI protocols
# ------------------------------------------------------------

for item in epi_series:
    protocol = item["Protocol"]
    series_number = item["Series"]

    print(f"\nProcessing {protocol}, Series {series_number}")

    epi_stack, epi_row_spacing, epi_col_spacing, epi_meta = load_image_stack(
        parent_folder,
        series_number,
        volume_index=epi_volume_index
    )

    metadata_rows.append(epi_meta)

    epi_slices = central_slice_indices(epi_stack.shape[0], number_of_central_slices)

    distortion_values = []
    epi_diameters = []

    for slice_number in epi_slices:
        slice_index = slice_number - 1
        image = epi_stack[slice_index]

        mask, threshold = segment_phantom(image)

        # Save one QC image per protocol and save the measurement-lines
        # figure for the baseline EPI central analysed slice.
        if slice_number == epi_slices[len(epi_slices) // 2]:
            safe_protocol_name = protocol.replace(" ", "_").replace("/", "_")

            save_qc_image(
                image,
                mask,
                f"{protocol} segmentation, slice {slice_number}",
                os.path.join(qc_folder, f"{safe_protocol_name}_QC.png")
            )

            if protocol == "Baseline EPI":
                save_measurement_lines_image(
                    image=image,
                    mask=mask,
                    directions_deg=directions_deg,
                    title=f"{protocol} measurement directions, slice {slice_number}",
                    output_path=os.path.join(qc_folder, "Baseline_EPI_measurement_lines.png")
                )

        for angle in directions_deg:
            epi_diameter = directional_diameter_mm(
                mask,
                epi_row_spacing,
                epi_col_spacing,
                angle
            )

            t1_reference_diameter = t1_reference_by_direction[angle]

            distortion_percent = (
                (epi_diameter - t1_reference_diameter)
                / t1_reference_diameter
            ) * 100

            epi_diameters.append(epi_diameter)
            distortion_values.append(distortion_percent)

            raw_rows.append({
                "Protocol": protocol,
                "SeriesNumber": series_number,
                "EPI_slice": slice_number,
                "Direction_deg": angle,
                "T1_reference_diameter_mm": t1_reference_diameter,
                "EPI_diameter_mm": epi_diameter,
                "Distortion_percent": distortion_percent,
                "EPI_row_spacing_mm": epi_row_spacing,
                "EPI_col_spacing_mm": epi_col_spacing,
                "T1_row_spacing_mm": t1_row_spacing,
                "T1_col_spacing_mm": t1_col_spacing
            })

    distortion_values = np.array(distortion_values)
    epi_diameters = np.array(epi_diameters)

    n = len(distortion_values)
    mean_distortion = np.mean(distortion_values)
    sd_distortion = np.std(distortion_values, ddof=1)
    se_distortion = sd_distortion / np.sqrt(n)
    t_value = t.ppf(0.975, n - 1)
    ci_half_width = t_value * se_distortion

    mean_epi_diameter = np.mean(epi_diameters)
    sd_epi_diameter = np.std(epi_diameters, ddof=1)

    summary_rows.append({
        "Protocol": protocol,
        "SeriesNumber": series_number,
        "Number_of_slices": len(epi_slices),
        "Number_of_directions": len(directions_deg),
        "Number_of_measurements": n,
        "Mean_EPI_diameter_mm": mean_epi_diameter,
        "SD_EPI_diameter_mm": sd_epi_diameter,
        "Mean_distortion_percent": mean_distortion,
        "SD_distortion_percent": sd_distortion,
        "SE_distortion_percent": se_distortion,
        "t_value": t_value,
        "95_CI_half_width_percent": ci_half_width,
        "Lower_95_CI_percent": mean_distortion - ci_half_width,
        "Upper_95_CI_percent": mean_distortion + ci_half_width,
        "Reported_distortion_percent": f"{mean_distortion:.2f} ± {ci_half_width:.2f}"
    })

    print(
        f"{protocol}: mean distortion = "
        f"{mean_distortion:.2f} ± {ci_half_width:.2f}%"
    )


# ============================================================
# SAVE OUTPUTS
# ============================================================

raw_df = pd.DataFrame(raw_rows)
summary_df = pd.DataFrame(summary_rows)
metadata_df = pd.DataFrame(metadata_rows)

with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    metadata_df.to_excel(writer, sheet_name="Metadata_spacing_used", index=False)
    t1_df.to_excel(writer, sheet_name="T1_reference_measurements", index=False)
    raw_df.to_excel(writer, sheet_name="EPI_raw_measurements", index=False)
    summary_df.to_excel(writer, sheet_name="Summary_results", index=False)

print(f"\nSaved corrected geometric distortion results to: {output_excel}")
print(f"Saved QC segmentation images to: {qc_folder}")
print(
    "Saved baseline EPI measurement-lines figure to: "
    f"{os.path.join(qc_folder, 'Baseline_EPI_measurement_lines.png')}"
)

print("\nSummary:")
print(summary_df[["Protocol", "Reported_distortion_percent"]])