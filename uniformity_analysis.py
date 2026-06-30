# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 00:39:58 2026

@author: Elisa Spiteri
"""

import os
import pydicom  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
# ============================================================
# USER SETTINGS
# ============================================================

parent_folder = r"path_to_dicom_folder"

uniformity_series = [
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

volume_used = 15

# Uniformity ROI settings for 80 x 80 EPI images.
# This ROI is larger than the SNR ROI but avoids the bright phantom edge.
uniformity_x = 40
uniformity_y = 40
uniformity_radius = 18

# Local averaging window used to find Smax and Smin.
# This avoids using single noisy pixels.
local_window_size = 5

output_excel = os.path.join(parent_folder, "Uniformity_all_protocols_results.xlsx")
output_plot = os.path.join(parent_folder, "Uniformity_all_protocols_plot.png")
output_roi_plot = os.path.join(parent_folder, "Uniformity_ROI_and_kernel_movement_baseline.png")

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


def local_mean_values_inside_mask(img, mask, window_size):
    """
    Calculate local mean signal values using a small square window.
    Only windows fully inside the phantom ROI are used.
    """
    half = window_size // 2
    values = []

    rows, cols = img.shape

    for y in range(half, rows - half):
        for x in range(half, cols - half):
            window_mask = mask[y - half:y + half + 1, x - half:x + half + 1]

            if np.all(window_mask):
                window = img[y - half:y + half + 1, x - half:x + half + 1]
                values.append(np.mean(window))

    return np.array(values)


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

def save_uniformity_roi_kernel_figure(
    image,
    output_path,
    roi_center_x,
    roi_center_y,
    roi_radius,
    kernel_size
):
    """
    Saves a two-panel figure showing:
    (a) the circular ROI used for PIU analysis
    (b) example 5 x 5 pixel kernel positions within the ROI

    The rectangles in panel (b) are illustrative examples. The actual PIU
    calculation uses all valid 5 x 5 kernel positions fully contained within
    the circular ROI.
    """

    half_kernel = kernel_size // 2

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    # --------------------------------------------------------
    # Panel A: ROI placement
    # --------------------------------------------------------
    axes[0].imshow(image, cmap="gray")

    roi_circle_a = Circle(
        (roi_center_x, roi_center_y),
        roi_radius,
        fill=False,
        edgecolor="yellow",
        linewidth=2
    )
    axes[0].add_patch(roi_circle_a)

    axes[0].plot(
        roi_center_x,
        roi_center_y,
        marker="o",
        color="yellow",
        markersize=4
    )

    axes[0].set_title("(a) Uniformity ROI")
    axes[0].axis("off")

    # --------------------------------------------------------
    # Panel B: example moving kernel positions
    # --------------------------------------------------------
    axes[1].imshow(image, cmap="gray")

    roi_circle_b = Circle(
        (roi_center_x, roi_center_y),
        roi_radius,
        fill=False,
        edgecolor="yellow",
        linewidth=2
    )
    axes[1].add_patch(roi_circle_b)

    # Example kernel centres shown for illustration.
    # These positions are safely inside the circular ROI.
    kernel_centres = [
        (roi_center_x - 10, roi_center_y - 10),
        (roi_center_x - 5, roi_center_y - 5),
        (roi_center_x, roi_center_y),
        (roi_center_x + 5, roi_center_y + 5),
        (roi_center_x + 10, roi_center_y + 10)
    ]

    for number, (kx, ky) in enumerate(kernel_centres, start=1):
        kernel_box = Rectangle(
            (kx - half_kernel, ky - half_kernel),
            kernel_size,
            kernel_size,
            fill=False,
            edgecolor="cyan",
            linewidth=1.6
        )
        axes[1].add_patch(kernel_box)

        axes[1].text(
            kx,
            ky,
            str(number),
            color="cyan",
            fontsize=8,
            ha="center",
            va="center",
            fontweight="bold"
        )

    # Arrow to indicate kernel movement across the ROI.
    axes[1].annotate(
        "",
        xy=(roi_center_x + 10, roi_center_y + 10),
        xytext=(roi_center_x - 10, roi_center_y - 10),
        arrowprops=dict(
            arrowstyle="->",
            color="cyan",
            linewidth=1.8
        )
    )

    axes[1].set_title("(b) Example moving 5 x 5 kernel")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
# ============================================================
# MAIN ANALYSIS
# ============================================================

raw_rows = []
summary_rows = []

for item in uniformity_series:
    protocol = item["Protocol"]
    series_number = item["Series"]

    print(f"\nProcessing {protocol}, Series {series_number}")

    files = load_series_by_number(parent_folder, series_number)

    if len(files) < volume_used:
        print(f"Skipped {protocol}: fewer than {volume_used} volumes.")
        continue

    ds = pydicom.dcmread(files[volume_used - 1][1], force=True)
    arr = ds.pixel_array.astype(float)

    print("Pixel array shape:", arr.shape)

    if arr.ndim != 3:
        print(f"Skipped {protocol}: pixel array was not 3D.")
        continue

    n_slices = arr.shape[0]
    central_slice = int(np.ceil(n_slices / 2))
    slices_used = list(range(central_slice - 2, central_slice + 3))

    image_shape = arr[0].shape

    uniformity_mask = circular_roi_mask(
        image_shape,
        uniformity_x,
        uniformity_y,
        uniformity_radius
    )

    piu_values = []
    cv_values = []

    for slice_number in slices_used:
        slice_index = slice_number - 1
        img = arr[slice_index]

        local_means = local_mean_values_inside_mask(
            img,
            uniformity_mask,
            local_window_size
        )

        Smax = np.max(local_means)
        Smin = np.min(local_means)

        piu = 100 * (1 - ((Smax - Smin) / (Smax + Smin)))

        mean_signal = np.mean(img[uniformity_mask])
        sd_signal = np.std(img[uniformity_mask], ddof=1)
        cv = (sd_signal / mean_signal) * 100

        piu_values.append(piu)
        cv_values.append(cv)

        raw_rows.append({
            "Protocol": protocol,
            "Series number": series_number,
            "Volume used": volume_used,
            "Slice used": slice_number,
            "Number of slices": n_slices,
            "Uniformity ROI centre x": uniformity_x,
            "Uniformity ROI centre y": uniformity_y,
            "Uniformity ROI radius / pixels": uniformity_radius,
            "Local window size / pixels": local_window_size,
            "Smax local mean": Smax,
            "Smin local mean": Smin,
            "Mean signal in ROI": mean_signal,
            "SD signal in ROI": sd_signal,
            "PIU (%)": piu,
            "CV (%)": cv,
            "Notes": "PIU calculated from local mean values inside phantom ROI"
        })

    piu_values = np.array(piu_values)
    cv_values = np.array(cv_values)

    n = len(piu_values)

    mean_piu = np.mean(piu_values)
    sd_piu = np.std(piu_values, ddof=1)
    se_piu = sd_piu / np.sqrt(n)
    t_val = t_value_95_two_sided(n - 1)
    ci_half_width_piu = t_val * se_piu

    mean_cv = np.mean(cv_values)
    sd_cv = np.std(cv_values, ddof=1)
    se_cv = sd_cv / np.sqrt(n)
    ci_half_width_cv = t_val * se_cv

    summary_rows.append({
        "Protocol": protocol,
        "Series number": series_number,
        "Number of slices analysed": n,
        "Mean PIU (%)": mean_piu,
        "PIU standard deviation": sd_piu,
        "PIU standard error": se_piu,
        "PIU 95% CI half-width": ci_half_width_piu,
        "PIU lower 95% CI": mean_piu - ci_half_width_piu,
        "PIU upper 95% CI": mean_piu + ci_half_width_piu,
        "Reported PIU (%)": f"{mean_piu:.1f} ± {ci_half_width_piu:.1f}",
        "Mean CV (%)": mean_cv,
        "CV standard deviation": sd_cv,
        "CV standard error": se_cv,
        "CV 95% CI half-width": ci_half_width_cv,
        "Reported CV (%)": f"{mean_cv:.2f} ± {ci_half_width_cv:.2f}",
        "Notes": "Mean PIU and CV across five central slices"
    })

    print(f"Mean PIU = {mean_piu:.1f} ± {ci_half_width_piu:.1f} %")
    print(f"Mean CV = {mean_cv:.2f} ± {ci_half_width_cv:.2f} %")


df_raw = pd.DataFrame(raw_rows)
df_summary = pd.DataFrame(summary_rows)

with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    df_raw.to_excel(writer, sheet_name="Uniformity_raw_measurements", index=False)
    df_summary.to_excel(writer, sheet_name="Uniformity_summary", index=False)

print("\nSaved uniformity workbook to:")
print(output_excel)

print("\nUniformity summary:")
print(df_summary)


# ============================================================
# SAVE PIU BAR PLOT
# ============================================================

if not df_summary.empty:
    plt.figure(figsize=(9, 5))
    plt.bar(
        df_summary["Protocol"],
        df_summary["Mean PIU (%)"],
        yerr=df_summary["PIU 95% CI half-width"],
        capsize=5
    )
    plt.ylabel("Percentage integral uniformity (%)")
    plt.xlabel("Protocol")
    plt.title("Signal uniformity across analysed EPI protocols")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    plt.show()

    print("\nSaved uniformity plot to:")
    print(output_plot)

# ============================================================
# SAVE ROI AND MOVING-KERNEL FIGURE USING BASELINE
# ============================================================

try:
    baseline_files = load_series_by_number(parent_folder, 6)
    baseline_ds = pydicom.dcmread(baseline_files[volume_used - 1][1], force=True)
    baseline_arr = baseline_ds.pixel_array.astype(float)

    baseline_slice_index = 13 - 1
    img = baseline_arr[baseline_slice_index]

    save_uniformity_roi_kernel_figure(
        image=img,
        output_path=output_roi_plot,
        roi_center_x=uniformity_x,
        roi_center_y=uniformity_y,
        roi_radius=uniformity_radius,
        kernel_size=local_window_size
    )

    print("\nSaved uniformity ROI and moving-kernel figure to:")
    print(output_roi_plot)

except Exception as e:
    print("\nCould not save uniformity ROI and moving-kernel figure:")
    print(e) 