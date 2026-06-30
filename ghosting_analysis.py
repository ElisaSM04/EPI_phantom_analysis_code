# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 01:22:53 2026

@author: one culture
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 23:42:25 2026 

@author: one culture
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

ghosting_series = [
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

# ROI settings for 80 x 80 EPI images.
# These are circular ROIs.
phantom_x = 40
phantom_y = 40
phantom_radius = 12

ghost_radius = 5
background_radius = 5

# For AP/PA phase encoding, ghost ROIs are placed above and below the phantom.
ghost1_x = 40
ghost1_y = 10

ghost2_x = 40
ghost2_y = 70

# Background correction ROIs are placed left and right of the phantom.
background1_x = 10
background1_y = 40

background2_x = 70
background2_y = 40

output_excel = os.path.join(parent_folder, "Ghosting_all_protocols_results.xlsx")
output_plot = os.path.join(parent_folder, "Ghosting_all_protocols_plot.png")
output_roi_plot = os.path.join(parent_folder, "Ghosting_ROI_placement_baseline.png")


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
# MAIN ANALYSIS
# ============================================================

raw_rows = []
summary_rows = []

for item in ghosting_series:
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

    phantom_mask = circular_roi_mask(image_shape, phantom_x, phantom_y, phantom_radius)
    ghost1_mask = circular_roi_mask(image_shape, ghost1_x, ghost1_y, ghost_radius)
    ghost2_mask = circular_roi_mask(image_shape, ghost2_x, ghost2_y, ghost_radius)
    background1_mask = circular_roi_mask(image_shape, background1_x, background1_y, background_radius)
    background2_mask = circular_roi_mask(image_shape, background2_x, background2_y, background_radius)

    ghosting_values = []

    for slice_number in slices_used:
        slice_index = slice_number - 1
        img = arr[slice_index]

        S = np.mean(img[phantom_mask])
        G1 = np.mean(img[ghost1_mask])
        G2 = np.mean(img[ghost2_mask])
        B1 = np.mean(img[background1_mask])
        B2 = np.mean(img[background2_mask])

        ghosting_percent = (abs((G1 + G2) - (B1 + B2)) / (2 * S)) * 100
        ghosting_values.append(ghosting_percent)

        raw_rows.append({
            "Protocol": protocol,
            "Series number": series_number,
            "Volume used": volume_used,
            "Slice used": slice_number,
            "Number of slices": n_slices,
            "Phantom ROI mean S": S,
            "Ghost ROI 1 mean G1": G1,
            "Ghost ROI 2 mean G2": G2,
            "Background ROI 1 mean B1": B1,
            "Background ROI 2 mean B2": B2,
            "Ghosting (%)": ghosting_percent,
            "Notes": "Ghost ROIs above/below phantom; background ROIs left/right"
        })

    ghosting_values = np.array(ghosting_values)
    n = len(ghosting_values)
    mean_ghosting = np.mean(ghosting_values)
    sd_ghosting = np.std(ghosting_values, ddof=1)
    se_ghosting = sd_ghosting / np.sqrt(n)
    t_val = t_value_95_two_sided(n - 1)
    ci_half_width = t_val * se_ghosting

    summary_rows.append({
        "Protocol": protocol,
        "Series number": series_number,
        "Number of slices analysed": n,
        "Mean ghosting (%)": mean_ghosting,
        "Standard deviation": sd_ghosting,
        "Standard error": se_ghosting,
        "t-value": t_val,
        "95% CI half-width": ci_half_width,
        "Lower 95% CI": mean_ghosting - ci_half_width,
        "Upper 95% CI": mean_ghosting + ci_half_width,
        "Reported ghosting (%)": f"{mean_ghosting:.2f} ± {ci_half_width:.2f}",
        "Notes": "Mean ghosting across five central slices"
    })

    print(f"Mean ghosting = {mean_ghosting:.2f} ± {ci_half_width:.2f} %")


df_raw = pd.DataFrame(raw_rows)
df_summary = pd.DataFrame(summary_rows)

with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    df_raw.to_excel(writer, sheet_name="Ghosting_raw_measurements", index=False)
    df_summary.to_excel(writer, sheet_name="Ghosting_summary", index=False)

print("\nSaved ghosting workbook to:")
print(output_excel)

print("\nGhosting summary:")
print(df_summary)


# ============================================================
# SAVE GHOSTING BAR PLOT
# ============================================================

if not df_summary.empty:
    plt.figure(figsize=(9, 5))
    plt.bar(
        df_summary["Protocol"],
        df_summary["Mean ghosting (%)"],
        yerr=df_summary["95% CI half-width"],
        capsize=5
    )
    plt.ylabel("Ghosting ratio (%)")
    plt.xlabel("Protocol")
    plt.title("Ghosting ratio across analysed EPI protocols")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    plt.show()

    print("\nSaved ghosting plot to:")
    print(output_plot)


# ============================================================
# SAVE ROI PLACEMENT FIGURE USING BASELINE
# ============================================================

try:
    baseline_files = load_series_by_number(parent_folder, 6)
    baseline_ds = pydicom.dcmread(baseline_files[volume_used - 1][1], force=True)
    baseline_arr = baseline_ds.pixel_array.astype(float)

    baseline_slice_index = 13 - 1
    img = baseline_arr[baseline_slice_index]

    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap="gray")

    rois = [
        (phantom_x, phantom_y, phantom_radius, "S"),
        (ghost1_x, ghost1_y, ghost_radius, "G1"),
        (ghost2_x, ghost2_y, ghost_radius, "G2"),
        (background1_x, background1_y, background_radius, "B1"),
        (background2_x, background2_y, background_radius, "B2"),
    ]

    for x, y, r, label in rois:
        circle = plt.Circle((x, y), r, fill=False, edgecolor='yellow', linewidth=2)
        plt.gca().add_patch(circle)
        plt.text(x + r + 1, y, label, color='yellow', fontsize=12, fontweight='bold')

    plt.title("Ghosting ROI placement: baseline EPI, volume 15, slice 13")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_roi_plot, dpi=300)
    plt.show()

    print("\nSaved ghosting ROI placement plot to:")
    print(output_roi_plot)

except Exception as e:
    print("\nCould not save ROI placement plot:")
    print(e)