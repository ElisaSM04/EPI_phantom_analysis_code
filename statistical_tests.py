# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 01:28:09 2026

@author: Elisa Spiteri
"""
"""

Statistical comparisons for EPI phantom image quality metrics.

This script performs:
1. Friedman omnibus tests across protocols.
2. Planned paired comparisons versus baseline EPI.
3. Holm correction for multiple comparisons.

The tests are exploratory within-dataset comparisons. The repeated units are
slice-specific or slice-direction-specific measurements, not independent scan sessions.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, ttest_rel, wilcoxon


# ============================================================
# USER SETTINGS
# ============================================================

parent_folder = r"path_to_dicom_folder"

baseline_protocol = "Baseline EPI"

output_excel = os.path.join(parent_folder, "Statistical_tests_results.xlsx")


input_files = {
    "Geometric distortion": {
        "file": "Geometric_distortion_multislice_results_corrected.xlsx",
        "sheet": "EPI_raw_measurements",
        "protocol_col": "Protocol",
        "value_col": "Distortion_percent",
        "unit_cols": ["EPI_slice", "Direction_deg"]
    },
    "SNR": {
        "file": "SNR_all_protocols_results.xlsx",
        "sheet": "SNR_raw_measurements",
        "protocol_col": "Protocol",
        "value_col": "SNR",
        "unit_cols": ["Slice used"]
    },
    "Ghosting ratio": {
        "file": "Ghosting_all_protocols_results.xlsx",
        "sheet": "Ghosting_raw_measurements",
        "protocol_col": "Protocol",
        "value_col": "Ghosting (%)",
        "unit_cols": ["Slice used"]
    },
    "Signal uniformity": {
        "file": "Uniformity_all_protocols_results.xlsx",
        "sheet": "Uniformity_raw_measurements",
        "protocol_col": "Protocol",
        "value_col": "PIU (%)",
        "unit_cols": ["Slice used"]
    }
}


# ============================================================
# FUNCTIONS
# ============================================================

def holm_adjust(p_values):
    """
    Holm-Bonferroni adjusted p-values.
    Returns adjusted p-values in the original order.
    """
    p_values = np.array(p_values, dtype=float)
    m = len(p_values)

    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)

    running_max = 0

    for rank, idx in enumerate(order):
        multiplier = m - rank
        adjusted_value = p_values[idx] * multiplier
        running_max = max(running_max, adjusted_value)
        adjusted[idx] = min(running_max, 1.0)

    return adjusted


def make_unit_column(df, unit_cols):
    """
    Creates one matched measurement-unit column from slice or slice-direction columns.
    """
    if len(unit_cols) == 1:
        return df[unit_cols[0]].astype(str)

    return df[unit_cols].astype(str).agg("_".join, axis=1)


def prepare_pivot(df, protocol_col, value_col, unit_cols):
    """
    Converts long-format raw measurements into wide format:
    rows = matched measurement units
    columns = protocols
    values = metric values
    """
    df = df.copy()

    df["Measurement_unit"] = make_unit_column(df, unit_cols)

    pivot = df.pivot_table(
        index="Measurement_unit",
        columns=protocol_col,
        values=value_col,
        aggfunc="mean"
    )

    return pivot


def run_friedman_test(pivot, metric_name):
    """
    Runs Friedman test across protocols using only complete matched rows.
    """
    complete = pivot.dropna(axis=0)

    protocols = list(complete.columns)

    if complete.shape[0] < 2 or complete.shape[1] < 3:
        return {
            "Metric": metric_name,
            "Test": "Friedman",
            "Number of matched units": complete.shape[0],
            "Number of protocols": complete.shape[1],
            "Statistic": np.nan,
            "p value": np.nan,
            "Interpretation": "Not enough matched data for Friedman test"
        }

    stat, p = friedmanchisquare(*[complete[col].values for col in protocols])

    return {
        "Metric": metric_name,
        "Test": "Friedman",
        "Number of matched units": complete.shape[0],
        "Number of protocols": complete.shape[1],
        "Statistic": stat,
        "p value": p,
        "Interpretation": "Overall protocol effect tested across matched measurement units"
    }


def run_pairwise_vs_baseline(pivot, metric_name, baseline_protocol):
    """
    Runs paired comparisons of each protocol against baseline.

    Both paired t-test and Wilcoxon signed-rank test are included. The paired t-test
    is consistent with the t-based confidence intervals already used in the thesis,
    while the Wilcoxon test is a non-parametric sensitivity check.
    """
    rows = []

    if baseline_protocol not in pivot.columns:
        return pd.DataFrame([{
            "Metric": metric_name,
            "Comparison": "N/A",
            "n matched units": np.nan,
            "Mean baseline": np.nan,
            "Mean protocol": np.nan,
            "Mean difference": np.nan,
            "Paired t p value": np.nan,
            "Wilcoxon p value": np.nan,
            "Note": "Baseline protocol not found"
        }])

    protocols = [p for p in pivot.columns if p != baseline_protocol]

    for protocol in protocols:
        paired = pivot[[baseline_protocol, protocol]].dropna()

        n = len(paired)

        if n < 2:
            rows.append({
                "Metric": metric_name,
                "Comparison": f"{protocol} vs {baseline_protocol}",
                "n matched units": n,
                "Mean baseline": np.nan,
                "Mean protocol": np.nan,
                "Mean difference": np.nan,
                "Paired t statistic": np.nan,
                "Paired t p value": np.nan,
                "Wilcoxon statistic": np.nan,
                "Wilcoxon p value": np.nan,
                "Note": "Not enough matched units"
            })
            continue

        baseline_values = paired[baseline_protocol].values
        protocol_values = paired[protocol].values
        differences = protocol_values - baseline_values

        t_stat, t_p = ttest_rel(protocol_values, baseline_values)

        try:
            if np.allclose(differences, 0):
                w_stat, w_p = np.nan, 1.0
            else:
                w_stat, w_p = wilcoxon(protocol_values, baseline_values)
        except Exception:
            w_stat, w_p = np.nan, np.nan

        rows.append({
            "Metric": metric_name,
            "Comparison": f"{protocol} vs {baseline_protocol}",
            "n matched units": n,
            "Mean baseline": np.mean(baseline_values),
            "Mean protocol": np.mean(protocol_values),
            "Mean difference": np.mean(differences),
            "Paired t statistic": t_stat,
            "Paired t p value": t_p,
            "Wilcoxon statistic": w_stat,
            "Wilcoxon p value": w_p,
            "Note": "Matched paired comparison versus baseline"
        })

    results = pd.DataFrame(rows)

    if not results.empty:
        results["Paired t Holm-adjusted p value"] = holm_adjust(
            results["Paired t p value"].fillna(1.0).values
        )
        results["Wilcoxon Holm-adjusted p value"] = holm_adjust(
            results["Wilcoxon p value"].fillna(1.0).values
        )

        results["Significant after Holm correction, paired t"] = (
            results["Paired t Holm-adjusted p value"] < 0.05
        )

        results["Significant after Holm correction, Wilcoxon"] = (
            results["Wilcoxon Holm-adjusted p value"] < 0.05
        )

    return results


# ============================================================
# MAIN SCRIPT
# ============================================================

friedman_rows = []
pairwise_tables = {}

for metric_name, settings in input_files.items():

    file_path = os.path.join(parent_folder, settings["file"])

    print(f"\nProcessing {metric_name}")
    print(file_path)

    if not os.path.exists(file_path):
        print(f"Missing file: {file_path}")
        friedman_rows.append({
            "Metric": metric_name,
            "Test": "Friedman",
            "Number of matched units": np.nan,
            "Number of protocols": np.nan,
            "Statistic": np.nan,
            "p value": np.nan,
            "Interpretation": "Input file not found"
        })
        continue

    df = pd.read_excel(file_path, sheet_name=settings["sheet"])

    pivot = prepare_pivot(
        df=df,
        protocol_col=settings["protocol_col"],
        value_col=settings["value_col"],
        unit_cols=settings["unit_cols"]
    )

    friedman_rows.append(run_friedman_test(pivot, metric_name))

    pairwise_tables[metric_name] = run_pairwise_vs_baseline(
        pivot=pivot,
        metric_name=metric_name,
        baseline_protocol=baseline_protocol
    )


df_friedman = pd.DataFrame(friedman_rows)

with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    df_friedman.to_excel(writer, sheet_name="Friedman_omnibus_tests", index=False)

    for metric_name, table in pairwise_tables.items():
        safe_sheet_name = metric_name.replace(" ", "_")[:25]
        table.to_excel(writer, sheet_name=f"{safe_sheet_name}_vs_baseline", index=False)

print("\nSaved statistical test results to:")
print(output_excel)

print("\nFriedman omnibus tests:")
print(df_friedman)