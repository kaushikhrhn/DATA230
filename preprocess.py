"""
Preprocess hotel booking demand data for team analytics and modeling.

Expected inputs (under pre-data/ relative to this script):
- h1.csv (or H1.csv): resort hotel bookings
- h2.csv (or H2.csv): city hotel bookings

Generated outputs (under processed/):
- hotel_bookings_cleaned.csv: merged, human-readable cleaned dataset for EDA/team use
- hotel_bookings_model_base.csv: model-base dataset (no leakage-risk columns, no one-hot encoding)
- preprocess_summary.md: detailed markdown report of preprocessing actions and audits
- duplicate_audit.csv: exact duplicate groups for manual inspection (if any)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


INTEGER_LIKE_COLUMNS = [
    "IsCanceled",
    "LeadTime",
    "ArrivalDateYear",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "BookingChanges",
    "DaysInWaitingList",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests",
]

FLOAT_LIKE_COLUMNS = ["ADR"]
DATE_COLUMNS = ["ReservationStatusDate"]

SUSPICIOUS_COLUMNS = [
    "ADR",
    "LeadTime",
    "Adults",
    "Children",
    "Babies",
    "BookingChanges",
    "DaysInWaitingList",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests",
]

NULL_LIKE_VALUES = {"", "null", "na", "n/a", "nan", "none", "<na>"}

MONTH_MAP = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Sept": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def find_input_file(input_dir: Path, expected_name: str) -> Path:
    """Find input CSV with case-insensitive filename matching."""
    expected_lower = expected_name.lower()
    matches = [p for p in input_dir.iterdir() if p.is_file() and p.name.lower() == expected_lower]
    if not matches:
        raise FileNotFoundError(f"Missing required input file: {input_dir / expected_name}")
    return matches[0]


def load_csvs(base_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Load H1/H2 CSVs and return raw dataframes with basic metadata."""
    input_dir = base_dir / "pre-data"
    if not input_dir.exists():
        raise FileNotFoundError(f"Missing input folder: {input_dir}")

    h1_path = find_input_file(input_dir, "h1.csv")
    h2_path = find_input_file(input_dir, "h2.csv")

    h1_df = pd.read_csv(h1_path, dtype="string", keep_default_na=False, na_filter=False)
    h2_df = pd.read_csv(h2_path, dtype="string", keep_default_na=False, na_filter=False)

    metadata = {
        "h1_path": str(h1_path),
        "h2_path": str(h2_path),
        "h1_shape": h1_df.shape,
        "h2_shape": h2_df.shape,
        "h1_columns": h1_df.columns.tolist(),
        "h2_columns": h2_df.columns.tolist(),
    }
    return h1_df, h2_df, metadata


def validate_schema(h1_df: pd.DataFrame, h2_df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate exact schema match by column order and names."""
    h1_cols = h1_df.columns.tolist()
    h2_cols = h2_df.columns.tolist()
    schemas_match = h1_cols == h2_cols
    differences: List[str] = []
    if not schemas_match:
        max_len = max(len(h1_cols), len(h2_cols))
        for idx in range(max_len):
            h1_col = h1_cols[idx] if idx < len(h1_cols) else "<MISSING>"
            h2_col = h2_cols[idx] if idx < len(h2_cols) else "<MISSING>"
            if h1_col != h2_col:
                differences.append(f"Position {idx}: H1='{h1_col}' vs H2='{h2_col}'")
    return schemas_match, differences


def standardize_column_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Strip whitespace around column names, error on collisions."""
    original_columns = df.columns.tolist()
    new_columns = [col.strip() for col in original_columns]
    if len(new_columns) != len(set(new_columns)):
        raise ValueError("Column name collision detected after stripping whitespace.")
    df = df.copy()
    df.columns = new_columns
    return df, {"original_columns": original_columns, "final_columns": new_columns}


def clean_strings(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Trim, normalize whitespace, and set NULL-like tokens to missing for string columns."""
    df = df.copy()
    object_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()
    null_normalization_counts: Dict[str, int] = {}
    blank_to_na_counts: Dict[str, int] = {}

    for col in object_columns:
        series = df[col].astype("string")
        series = series.str.replace(r"\s+", " ", regex=True).str.strip()

        blank_mask = series.eq("")
        blank_to_na_counts[col] = int(blank_mask.sum())
        series = series.mask(blank_mask, pd.NA)

        lowered = series.str.lower()
        null_like_mask = lowered.isin(NULL_LIKE_VALUES) | lowered.eq("null")
        null_normalization_counts[col] = int(null_like_mask.fillna(False).sum())
        series = series.mask(null_like_mask, pd.NA)

        df[col] = series

    if "Agent" not in df.columns or "Company" not in df.columns:
        missing = [c for c in ["Agent", "Company"] if c not in df.columns]
        raise KeyError(f"Missing expected columns required for indicators: {missing}")

    df["has_agent"] = df["Agent"].notna().astype("Int8")
    df["has_company"] = df["Company"].notna().astype("Int8")

    report = {
        "string_columns_cleaned": object_columns,
        "null_normalization_counts": null_normalization_counts,
        "blank_to_na_counts": blank_to_na_counts,
    }
    return df, report


def convert_types(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Convert selected columns to numeric/date types with coercion and reporting."""
    df = df.copy()
    report: Dict[str, Dict[str, int]] = {"numeric_failures": {}, "date_failures": {}}

    for col in INTEGER_LIKE_COLUMNS:
        if col not in df.columns:
            continue
        before_non_missing = int(df[col].notna().sum())
        converted = pd.to_numeric(df[col], errors="coerce")
        after_non_missing = int(converted.notna().sum())
        failures = before_non_missing - after_non_missing
        report["numeric_failures"][col] = failures
        df[col] = converted.astype("Int64")

    for col in FLOAT_LIKE_COLUMNS:
        if col not in df.columns:
            continue
        before_non_missing = int(df[col].notna().sum())
        converted = pd.to_numeric(df[col], errors="coerce")
        after_non_missing = int(converted.notna().sum())
        failures = before_non_missing - after_non_missing
        report["numeric_failures"][col] = failures
        df[col] = converted.astype("Float64")

    for col in DATE_COLUMNS:
        if col not in df.columns:
            continue
        before_non_missing = int(df[col].notna().sum())
        parsed = pd.to_datetime(df[col], errors="coerce")
        after_non_missing = int(parsed.notna().sum())
        failures = before_non_missing - after_non_missing
        report["date_failures"][col] = failures
        df[col] = parsed

    return df, report


def audit_duplicates(df: pd.DataFrame) -> Dict:
    """Count exact duplicate rows without removing them."""
    duplicate_total_rows = int(df.duplicated(keep=False).sum())
    duplicate_extra_rows = int(df.duplicated(keep="first").sum())
    report = {
        "duplicates_found_total_rows": duplicate_total_rows,
        "duplicates_found_extra_rows": duplicate_extra_rows,
        "duplicates_removed": 0,
    }
    return report


def write_duplicate_audit(df: pd.DataFrame, output_path: Path) -> str | None:
    """Write exact duplicate rows to a CSV for manual inspection."""
    duplicate_rows = df[df.duplicated(keep=False)].copy()
    if duplicate_rows.empty:
        return None
    duplicate_rows.to_csv(output_path, index=False)
    return str(output_path)


def audit_ranges(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Audit suspicious ranges, create warning summaries, and add outlier flags."""
    df = df.copy()
    warnings: Dict[str, int | float] = {}

    for col in SUSPICIOUS_COLUMNS:
        if col in df.columns:
            negatives = int((df[col] < 0).fillna(False).sum())
            if negatives > 0:
                warnings[f"{col}_negative_count"] = negatives

    if all(col in df.columns for col in ["Adults", "Children", "Babies"]):
        total_guests = (
            df["Adults"].fillna(0).astype("Float64")
            + df["Children"].fillna(0).astype("Float64")
            + df["Babies"].fillna(0).astype("Float64")
        )
        warnings["zero_total_guests_count"] = int((total_guests == 0).sum())
        warnings["very_large_guest_count_gt_10"] = int((total_guests > 10).sum())

    if "ADR" in df.columns:
        adr_numeric = df["ADR"].astype("Float64")
        q1 = adr_numeric.quantile(0.25)
        q3 = adr_numeric.quantile(0.75)
        iqr = q3 - q1
        adr_upper = q3 + (1.5 * iqr) if pd.notna(iqr) else np.nan
        if pd.notna(adr_upper):
            df["extreme_adr_flag"] = (adr_numeric > adr_upper).astype("Int8")
            warnings["extreme_adr_threshold_iqr_upper"] = float(adr_upper)
            warnings["extreme_adr_count"] = int((adr_numeric > adr_upper).sum())
        else:
            df["extreme_adr_flag"] = pd.Series(pd.array([0] * len(df), dtype="Int8"), index=df.index)
            warnings["extreme_adr_threshold_iqr_upper"] = np.nan
            warnings["extreme_adr_count"] = 0

    if "LeadTime" in df.columns:
        lead_numeric = df["LeadTime"].astype("Float64")
        lead_p99 = lead_numeric.quantile(0.99)
        if pd.notna(lead_p99):
            df["extreme_lead_time_flag"] = (lead_numeric > lead_p99).astype("Int8")
            warnings["extreme_lead_time_threshold_p99"] = float(lead_p99)
            warnings["extreme_lead_time_count"] = int((lead_numeric > lead_p99).sum())
        else:
            df["extreme_lead_time_flag"] = pd.Series(pd.array([0] * len(df), dtype="Int8"), index=df.index)
            warnings["extreme_lead_time_threshold_p99"] = np.nan
            warnings["extreme_lead_time_count"] = 0

    return df, warnings


def assign_season(month_num: pd.Series) -> pd.Series:
    """Map month number to season labels."""
    season = pd.Series(pd.NA, index=month_num.index, dtype="string")
    season = season.mask(month_num.isin([12, 1, 2]), "Winter")
    season = season.mask(month_num.isin([3, 4, 5]), "Spring")
    season = season.mask(month_num.isin([6, 7, 8]), "Summer")
    season = season.mask(month_num.isin([9, 10, 11]), "Fall")
    return season


def create_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Create derived features for EDA and baseline ML."""
    df = df.copy()
    feature_report: Dict[str, object] = {}

    required_columns = [
        "StaysInWeekendNights",
        "StaysInWeekNights",
        "Adults",
        "Children",
        "Babies",
        "TotalOfSpecialRequests",
        "RequiredCarParkingSpaces",
        "BookingChanges",
        "PreviousCancellations",
        "PreviousBookingsNotCanceled",
        "ArrivalDateMonth",
    ]
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        raise KeyError(f"Cannot create features. Missing required columns: {missing_required}")

    stays_weekend = df["StaysInWeekendNights"].fillna(0).astype("Float64")
    stays_week = df["StaysInWeekNights"].fillna(0).astype("Float64")
    adults = df["Adults"].fillna(0).astype("Float64")
    children = df["Children"].fillna(0).astype("Float64")
    babies = df["Babies"].fillna(0).astype("Float64")
    special_requests = df["TotalOfSpecialRequests"].fillna(0).astype("Float64")
    parking = df["RequiredCarParkingSpaces"].fillna(0).astype("Float64")
    booking_changes = df["BookingChanges"].fillna(0).astype("Float64")
    prev_cancel = df["PreviousCancellations"].fillna(0).astype("Float64")
    prev_not_cancel = df["PreviousBookingsNotCanceled"].fillna(0).astype("Float64")

    df["total_nights"] = stays_weekend + stays_week
    df["total_guests"] = adults + children + babies
    kids_total = children + babies
    df["has_children"] = (kids_total > 0).astype("Int8")
    df["is_family_booking"] = (kids_total > 0).astype("Int8")
    df["has_special_requests"] = (special_requests > 0).astype("Int8")
    df["has_parking_request"] = (parking > 0).astype("Int8")
    df["has_booking_changes"] = (booking_changes > 0).astype("Int8")

    prev_total = prev_cancel + prev_not_cancel
    df["had_previous_history"] = (prev_total > 0).astype("Int8")
    df["previous_cancel_rate_proxy"] = np.where(prev_total > 0, prev_cancel / prev_total, np.nan)
    df["previous_cancel_rate_proxy"] = pd.Series(df["previous_cancel_rate_proxy"], index=df.index).astype("Float64")

    raw_month = df["ArrivalDateMonth"].astype("string")
    standardized_month = raw_month.str.replace(r"\s+", " ", regex=True).str.strip().str.title()
    df["ArrivalDateMonth"] = standardized_month

    month_num = standardized_month.map(MONTH_MAP)
    df["arrival_month_num"] = pd.to_numeric(month_num, errors="coerce").astype("Int64")
    df["season"] = assign_season(df["arrival_month_num"])

    quarter = np.where(
        df["arrival_month_num"].between(1, 3, inclusive="both"),
        "Q1",
        np.where(
            df["arrival_month_num"].between(4, 6, inclusive="both"),
            "Q2",
            np.where(
                df["arrival_month_num"].between(7, 9, inclusive="both"),
                "Q3",
                np.where(df["arrival_month_num"].between(10, 12, inclusive="both"), "Q4", pd.NA),
            ),
        ),
    )
    df["arrival_quarter"] = pd.Series(quarter, index=df.index, dtype="string")

    unmapped_month_values = sorted(standardized_month[df["arrival_month_num"].isna()].dropna().unique().tolist())
    feature_report["unmapped_month_values"] = unmapped_month_values
    feature_report["derived_columns"] = [
        "total_nights",
        "total_guests",
        "has_children",
        "is_family_booking",
        "has_special_requests",
        "has_parking_request",
        "has_booking_changes",
        "had_previous_history",
        "previous_cancel_rate_proxy",
        "arrival_month_num",
        "season",
        "arrival_quarter",
        "extreme_adr_flag",
        "extreme_lead_time_flag",
        "has_agent",
        "has_company",
    ]
    return df, feature_report


def leakage_audit(df: pd.DataFrame) -> Dict:
    """Classify columns by modeling safety and return model-base column selection."""
    columns = df.columns.tolist()
    target = "IsCanceled"
    leakage_risk = [col for col in ["ReservationStatus", "ReservationStatusDate"] if col in columns]
    eda_only = [col for col in ["AssignedRoomType"] if col in columns]
    excluded_for_model = leakage_risk + eda_only
    model_base_columns = [col for col in columns if col not in excluded_for_model]
    safe_for_model = model_base_columns.copy()

    return {
        "target": target,
        "leakage_risk": leakage_risk,
        "eda_only": eda_only,
        "excluded_for_model": excluded_for_model,
        "safe_for_model": safe_for_model,
    }


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build missingness summary table."""
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df) * 100).round(2) if len(df) else pd.Series(0, index=df.columns)
    table = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": missing_count.values,
            "missing_percent": missing_pct.values,
        }
    ).sort_values(["missing_count", "column"], ascending=[False, True])
    return table


def class_balance_summary(df: pd.DataFrame) -> Dict:
    """Compute cancellation rate overall and by hotel_type."""
    summary: Dict[str, object] = {}
    if "IsCanceled" not in df.columns:
        summary["overall_cancellation_rate"] = np.nan
        summary["by_hotel_type"] = {}
        return summary

    is_canceled_numeric = pd.to_numeric(df["IsCanceled"], errors="coerce")
    summary["overall_cancellation_rate"] = float(is_canceled_numeric.mean(skipna=True))

    if "hotel_type" in df.columns:
        rates = (
            pd.DataFrame({"hotel_type": df["hotel_type"], "IsCanceled": is_canceled_numeric})
            .groupby("hotel_type", dropna=False)["IsCanceled"]
            .mean()
            .to_dict()
        )
    else:
        rates = {}
    summary["by_hotel_type"] = {str(k): float(v) for k, v in rates.items() if pd.notna(v)}
    return summary


def write_markdown_report(
    output_path: Path,
    metadata: Dict,
    merge_report: Dict,
    cleaning_report: Dict,
    type_report: Dict,
    duplicate_report: Dict,
    suspicious_report: Dict,
    feature_report: Dict,
    leak_report: Dict,
    missing_table: pd.DataFrame,
    class_balance: Dict,
    outputs: Dict,
    assumptions: List[str],
) -> None:
    """Write project-friendly markdown summary."""
    top_missing = missing_table.head(15)
    null_norm = cleaning_report["null_normalization_counts"]
    heavily_affected = sorted(
        [(col, cnt) for col, cnt in null_norm.items() if cnt > 0],
        key=lambda x: x[1],
        reverse=True,
    )
    top_null_affected = heavily_affected[:10]

    md_lines: List[str] = []
    md_lines.append("# Preprocessing Summary")
    md_lines.append("")
    md_lines.append("## Input files")
    md_lines.append(f"- H1 file: `{metadata['h1_path']}` with shape `{metadata['h1_shape']}`")
    md_lines.append(f"- H2 file: `{metadata['h2_path']}` with shape `{metadata['h2_shape']}`")
    md_lines.append(f"- Schema match: `{merge_report['schema_match']}`")
    if not merge_report["schema_match"]:
        md_lines.append("- Schema differences detected:")
        for diff in merge_report.get("schema_differences", []):
            md_lines.append(f"  - {diff}")
    md_lines.append("")
    md_lines.append("## Merge result")
    md_lines.append(f"- Rows before merge: H1={merge_report['h1_rows']}, H2={merge_report['h2_rows']}")
    md_lines.append(f"- Rows after merge: {merge_report['merged_rows']}")
    md_lines.append("- Added `hotel_type` column with values `Resort` (H1) and `City` (H2).")
    md_lines.append("")
    md_lines.append("## Cleaning actions performed")
    md_lines.append("- Stripped whitespace from column names and values.")
    md_lines.append("- Collapsed repeated internal whitespace in string columns.")
    md_lines.append("- Converted blank and NULL-like string tokens to missing (`pd.NA`).")
    md_lines.append("- Converted numeric/date columns with coercion and logged parse failures.")
    md_lines.append("- Audited exact duplicate rows but retained them in main outputs because identical rows may represent distinct bookings.")
    md_lines.append("- Created derived features for EDA and baseline ML.")
    md_lines.append("")
    md_lines.append("### Type conversion/coercion summary")
    numeric_failures = {
        col: cnt for col, cnt in type_report.get("numeric_failures", {}).items() if cnt > 0
    }
    date_failures = {
        col: cnt for col, cnt in type_report.get("date_failures", {}).items() if cnt > 0
    }
    if numeric_failures:
        for col, cnt in numeric_failures.items():
            md_lines.append(f"- Numeric parse failures in `{col}` (coerced to missing): {cnt}")
    else:
        md_lines.append("- Numeric parse failures: none")
    if date_failures:
        for col, cnt in date_failures.items():
            md_lines.append(f"- Date parse failures in `{col}` (coerced to missing): {cnt}")
    else:
        md_lines.append("- Date parse failures: none")
    md_lines.append("")
    md_lines.append("## Missingness summary")
    md_lines.append("")
    md_lines.append("| column | missing_count | missing_percent |")
    md_lines.append("|---|---:|---:|")
    for _, row in top_missing.iterrows():
        md_lines.append(f"| {row['column']} | {int(row['missing_count'])} | {float(row['missing_percent']):.2f}% |")
    md_lines.append("")
    if top_null_affected:
        md_lines.append("- Columns most affected by NULL-like normalization:")
        for col, cnt in top_null_affected:
            md_lines.append(f"  - `{col}`: {cnt} normalized values")
    else:
        md_lines.append("- No NULL-like token normalization was needed.")
    md_lines.append("")
    md_lines.append("## Duplicate summary")
    md_lines.append(f"- Rows participating in exact duplicate groups: {duplicate_report['duplicates_found_total_rows']}")
    md_lines.append(f"- Extra duplicate rows beyond first occurrences: {duplicate_report['duplicates_found_extra_rows']}")
    md_lines.append(f"- Exact duplicates removed from main outputs: {duplicate_report['duplicates_removed']}")
    if outputs.get("duplicate_audit_csv"):
        md_lines.append(f"- Duplicate audit file written: `{outputs['duplicate_audit_csv']}`")
    else:
        md_lines.append("- No duplicate audit file written because no exact duplicate groups were found.")
    md_lines.append("")
    md_lines.append("## Suspicious values / outlier audit")
    for key, value in suspicious_report.items():
        if isinstance(value, float):
            md_lines.append(f"- `{key}`: {value:.4f}")
        else:
            md_lines.append(f"- `{key}`: {value}")
    md_lines.append("")
    md_lines.append("## Leakage audit")
    md_lines.append(f"- Target column: `{leak_report['target']}`")
    md_lines.append(f"- Leakage-risk columns excluded from model-ready output: {leak_report['leakage_risk']}")
    md_lines.append(f"- EDA-only columns excluded from model-ready output: {leak_report['eda_only']}")
    md_lines.append(f"- Total columns safe for model base: {len(leak_report['safe_for_model'])}")
    md_lines.append("")
    md_lines.append("## Derived features")
    for col in feature_report["derived_columns"]:
        md_lines.append(f"- `{col}`")
    if feature_report.get("unmapped_month_values"):
        md_lines.append(f"- Unmapped month values requiring review: {feature_report['unmapped_month_values']}")
    else:
        md_lines.append("- All `ArrivalDateMonth` values mapped successfully.")
    md_lines.append("")
    md_lines.append("## Final outputs")
    md_lines.append(f"- `{outputs['cleaned_csv']}` shape: {outputs['cleaned_shape']}")
    md_lines.append(f"- `{outputs['model_base_csv']}` shape: {outputs['model_base_shape']}")
    md_lines.append(f"- `{outputs['report_md']}` written.")
    if outputs.get("cleaned_parquet"):
        md_lines.append(f"- Optional parquet written: `{outputs['cleaned_parquet']}`")
    if outputs.get("model_base_parquet"):
        md_lines.append(f"- Optional parquet written: `{outputs['model_base_parquet']}`")
    if outputs.get("parquet_error"):
        md_lines.append(f"- Parquet export note: {outputs['parquet_error']}")
    md_lines.append("")
    md_lines.append("## Class balance summary")
    overall = class_balance.get("overall_cancellation_rate", np.nan)
    if pd.notna(overall):
        md_lines.append(f"- Overall cancellation rate: {overall:.4f}")
    else:
        md_lines.append("- Overall cancellation rate: unavailable")
    by_hotel = class_balance.get("by_hotel_type", {})
    if by_hotel:
        for hotel, rate in by_hotel.items():
            md_lines.append(f"- Cancellation rate for `{hotel}`: {rate:.4f}")
    else:
        md_lines.append("- Cancellation rate by hotel_type: unavailable")
    md_lines.append("")
    md_lines.append("## Notes / caveats")
    for note in assumptions:
        md_lines.append(f"- {note}")

    output_path.write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting preprocessing pipeline...")

    h1_df, h2_df, metadata = load_csvs(script_dir)
    print(f"Loaded files: H1{metadata['h1_shape']} and H2{metadata['h2_shape']}")
    print(f"H1 columns: {metadata['h1_columns']}")
    print(f"H2 columns: {metadata['h2_columns']}")

    schemas_match, schema_differences = validate_schema(h1_df, h2_df)
    print(f"Schemas match exactly: {schemas_match}")
    if not schemas_match:
        diff_text = "\n".join(schema_differences)
        raise ValueError(f"Schema mismatch detected between H1 and H2.\n{diff_text}")

    h1_df = h1_df.copy()
    h2_df = h2_df.copy()
    h1_df["hotel_type"] = "Resort"
    h2_df["hotel_type"] = "City"

    merged_df = pd.concat([h1_df, h2_df], axis=0, ignore_index=True)
    merge_report = {
        "schema_match": schemas_match,
        "schema_differences": schema_differences,
        "h1_rows": h1_df.shape[0],
        "h2_rows": h2_df.shape[0],
        "merged_rows": merged_df.shape[0],
    }
    print(f"Merged rows: {merged_df.shape[0]}")

    merged_df, col_report = standardize_column_names(merged_df)
    print(f"Final columns after standardization: {col_report['final_columns']}")

    cleaned_df, cleaning_report = clean_strings(merged_df)
    typed_df, type_report = convert_types(cleaned_df)
    numeric_fail_total = sum(type_report["numeric_failures"].values())
    date_fail_total = sum(type_report["date_failures"].values())
    print(f"Type conversion parse failures: numeric={numeric_fail_total}, date={date_fail_total}")

    duplicate_report = audit_duplicates(typed_df)
    print(
        "Exact duplicate rows detected "
        f"(all rows in duplicate groups): {duplicate_report['duplicates_found_total_rows']}"
    )
    print(
        "Extra duplicate rows beyond first occurrences: "
        f"{duplicate_report['duplicates_found_extra_rows']}"
    )

    duplicate_audit_csv = output_dir / "duplicate_audit.csv"
    duplicate_audit_path = write_duplicate_audit(typed_df, duplicate_audit_csv)

    audited_df, suspicious_report = audit_ranges(typed_df)
    featured_df, feature_report = create_features(audited_df)
    leak_report = leakage_audit(featured_df)

    missing_table = missingness_table(featured_df)
    class_balance = class_balance_summary(featured_df)

    cleaned_csv = output_dir / "hotel_bookings_cleaned.csv"
    model_base_csv = output_dir / "hotel_bookings_model_base.csv"
    report_md = output_dir / "preprocess_summary.md"

    model_base_df = featured_df[leak_report["safe_for_model"]].copy()

    export_cleaned_df = featured_df.copy()
    export_model_base_df = model_base_df.copy()

    if "ReservationStatusDate" in export_cleaned_df.columns:
        export_cleaned_df["ReservationStatusDate"] = export_cleaned_df["ReservationStatusDate"].dt.strftime("%Y-%m-%d")
    if "ReservationStatusDate" in export_model_base_df.columns:
        export_model_base_df["ReservationStatusDate"] = export_model_base_df["ReservationStatusDate"].dt.strftime("%Y-%m-%d")

    export_cleaned_df.to_csv(cleaned_csv, index=False)
    export_model_base_df.to_csv(model_base_csv, index=False)

    outputs = {
        "cleaned_csv": str(cleaned_csv),
        "model_base_csv": str(model_base_csv),
        "report_md": str(report_md),
        "duplicate_audit_csv": duplicate_audit_path,
        "cleaned_shape": export_cleaned_df.shape,
        "model_base_shape": export_model_base_df.shape,
        "cleaned_parquet": None,
        "model_base_parquet": None,
        "parquet_error": None,
    }

    try:
        cleaned_parquet = output_dir / "hotel_bookings_cleaned.parquet"
        model_base_parquet = output_dir / "hotel_bookings_model_base.parquet"
        featured_df.to_parquet(cleaned_parquet, index=False)
        model_base_df.to_parquet(model_base_parquet, index=False)
        outputs["cleaned_parquet"] = str(cleaned_parquet)
        outputs["model_base_parquet"] = str(model_base_parquet)
    except Exception as parquet_exc:
        outputs["parquet_error"] = (
            "Parquet export skipped/failed (likely missing optional parquet engine): "
            f"{parquet_exc}"
        )

    assumptions = [
        "Input filenames are matched case-insensitively (`h1.csv`/`H1.csv`, `h2.csv`/`H2.csv`).",
        "Exact duplicate rows are audited but retained in the main outputs because the released dataset has no booking identifier, so identical rows may still represent distinct bookings.",
        "NULL-like tokens are normalized to missing values across string columns, including Agent/Company.",
        "AssignedRoomType is treated as EDA-only and excluded from the model-base output by design.",
        "Outliers are flagged (not removed) using IQR for ADR and 99th percentile for LeadTime.",
    ]

    write_markdown_report(
        output_path=report_md,
        metadata=metadata,
        merge_report=merge_report,
        cleaning_report=cleaning_report,
        type_report=type_report,
        duplicate_report=duplicate_report,
        suspicious_report=suspicious_report,
        feature_report=feature_report,
        leak_report=leak_report,
        missing_table=missing_table,
        class_balance=class_balance,
        outputs=outputs,
        assumptions=assumptions,
    )

    print(f"Outputs written: {cleaned_csv.name}, {model_base_csv.name}, {report_md.name}")
    if duplicate_audit_path:
        print(f"Duplicate audit written: {Path(duplicate_audit_path).name}")
    print("Preprocessing complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Preprocessing failed: {exc}", file=sys.stderr)
        raise