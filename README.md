# EPI Phantom Analysis Code

This repository contains the Python scripts used for the phantom-based Echo Planar Imaging image quality analysis reported in the B.Sc. dissertation:

**Optimising Echo Planar Imaging Pulse Sequences for Ultra-Fast Imaging on a 3 T MRI Scanner**

## Contents

- `geometric_distortion_analysis.py`  
  Calculates multi-slice, multi-direction geometric distortion relative to the T1-weighted reference image.

- `SNR_analysis.py`  
  Calculates signal-to-noise ratio using the two-image subtraction method.

- `ghosting_analysis.py`  
  Calculates background-corrected ghosting ratio using region-of-interest measurements.

- `uniformity_analysis.py`  
  Calculates percentage integral uniformity using a moving 5 x 5 pixel kernel within the phantom ROI.

- `temporal_stability_analysis.py`  
  Calculates temporal signal-to-noise ratio, coefficient of variation and signal drift from a 200-volume EPI time series.

- `statistical_tests.py`  
  Performs exploratory Friedman tests and Holm-adjusted paired comparisons with the baseline EPI protocol.

## Data availability

Raw DICOM image data are not included in this repository. Local file paths were removed and replaced with generic placeholders.

## Requirements

The scripts require standard scientific Python packages, including NumPy, pandas, pydicom, matplotlib, scipy and openpyxl.

## Notes

The DICOM SeriesNumber values used in the scripts refer to exported DICOM series identifiers and do not necessarily correspond to the scan-log order numbers reported in the dissertation appendix.
