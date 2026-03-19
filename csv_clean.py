from __future__ import annotations

import pandas as pd


INPUT_FILE = "merged.csv"
OUTPUT_ANALYSIS_FILE = "flights_clean.csv"
OUTPUT_CANCELLED_FILE = "cancelled_flights.csv"
OUTPUT_DIVERTED_FILE = "diverted_flights.csv"
CHUNK_SIZE = 100_000


DROP_COLUMNS = [
    "Unnamed: 109",
    "DOT_ID_Reporting_Airline",
    "IATA_CODE_Reporting_Airline",
    "OriginAirportID",
    "OriginAirportSeqID",
    "OriginCityMarketID",
    "OriginStateFips",
    "OriginWac",
    "DestAirportID",
    "DestAirportSeqID",
    "DestCityMarketID",
    "DestStateFips",
    "DestWac",
    "DivAirportLandings",
    "DivReachedDest",
    "DivActualElapsedTime",
    "DivArrDelay",
    "DivDistance",
    "FirstDepTime",
    "TotalAddGTime",
    "LongestAddGTime",
]

CAUSE_COLUMNS = [
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
]

NUMERIC_COLUMNS = [
    "Year",
    "Quarter",
    "Month",
    "DayofMonth",
    "DayOfWeek",
    "Flight_Number_Reporting_Airline",
    "CRSDepTime",
    "DepTime",
    "DepDelay",
    "DepDelayMinutes",
    "DepDel15",
    "DepartureDelayGroups",
    "TaxiOut",
    "WheelsOff",
    "WheelsOn",
    "TaxiIn",
    "CRSArrTime",
    "ArrTime",
    "ArrDelay",
    "ArrDelayMinutes",
    "ArrDel15",
    "ArrivalDelayGroups",
    "Cancelled",
    "Diverted",
    "CRSElapsedTime",
    "ActualElapsedTime",
    "AirTime",
    "Flights",
    "Distance",
    "DistanceGroup",
] + CAUSE_COLUMNS


def build_diversion_columns() -> list[str]:
    columns: list[str] = []
    for i in range(1, 6):
        columns.extend(
            [
                f"Div{i}Airport",
                f"Div{i}AirportID",
                f"Div{i}AirportSeqID",
                f"Div{i}WheelsOn",
                f"Div{i}TotalGTime",
                f"Div{i}LongestGTime",
                f"Div{i}WheelsOff",
                f"Div{i}TailNum",
            ]
        )
    return columns


def to_hour_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return (numeric // 100).astype("Int64")


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")
    df["ScheduledDepHour"] = to_hour_series(df["CRSDepTime"])
    df["ScheduledArrHour"] = to_hour_series(df["CRSArrTime"])
    df["IsWeekend"] = df["DayOfWeek"].isin([6, 7]).astype("Int64")
    df["Route"] = df["Origin"].astype("string") + "-" + df["Dest"].astype("string")
    df["CarrierControlledDelay"] = df["CarrierDelay"] + df["LateAircraftDelay"]
    df["ExternalDelay"] = df["WeatherDelay"] + df["NASDelay"] + df["SecurityDelay"]
    df["DelayCauseTotal"] = df[CAUSE_COLUMNS].sum(axis=1)
    return df


def clean_chunk(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.drop(columns=DROP_COLUMNS + build_diversion_columns(), errors="ignore").copy()

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Reporting_Airline"] = df["Reporting_Airline"].astype("string")
    df["Origin"] = df["Origin"].astype("string")
    df["Dest"] = df["Dest"].astype("string")
    df["CancellationCode"] = df["CancellationCode"].astype("string")

    cancelled_mask = df["Cancelled"].fillna(0).eq(1)
    diverted_mask = df["Diverted"].fillna(0).eq(1)
    cancelled_df = df.loc[cancelled_mask].copy()
    diverted_df = df.loc[~cancelled_mask & diverted_mask].copy()

    analysis_df = df.loc[~cancelled_mask & ~diverted_mask].copy()

    not_delayed_mask = analysis_df["ArrDel15"].fillna(0).eq(0)
    analysis_df.loc[not_delayed_mask, CAUSE_COLUMNS] = analysis_df.loc[
        not_delayed_mask, CAUSE_COLUMNS
    ].fillna(0)

    analysis_df = add_derived_columns(analysis_df)
    cancelled_df = add_derived_columns(cancelled_df)
    diverted_df = add_derived_columns(diverted_df)

    dedupe_keys = [
        "FlightDate",
        "Reporting_Airline",
        "Flight_Number_Reporting_Airline",
        "Origin",
        "Dest",
        "CRSDepTime",
    ]
    analysis_df = analysis_df.drop_duplicates(subset=dedupe_keys)
    cancelled_df = cancelled_df.drop_duplicates(subset=dedupe_keys)
    diverted_df = diverted_df.drop_duplicates(subset=dedupe_keys)

    return analysis_df, cancelled_df, diverted_df


def write_chunk(df: pd.DataFrame, path: str, first_chunk: bool) -> None:
    df.to_csv(path, mode="w" if first_chunk else "a", index=False, header=first_chunk)


def main() -> None:
    first_analysis_chunk = True
    first_cancelled_chunk = True
    first_diverted_chunk = True
    processed_rows = 0
    analysis_rows = 0
    cancelled_rows = 0
    diverted_rows = 0

    for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False):
        processed_rows += len(chunk)
        analysis_df, cancelled_df, diverted_df = clean_chunk(chunk)

        analysis_rows += len(analysis_df)
        cancelled_rows += len(cancelled_df)
        diverted_rows += len(diverted_df)

        write_chunk(analysis_df, OUTPUT_ANALYSIS_FILE, first_analysis_chunk)
        write_chunk(cancelled_df, OUTPUT_CANCELLED_FILE, first_cancelled_chunk)
        write_chunk(diverted_df, OUTPUT_DIVERTED_FILE, first_diverted_chunk)

        first_analysis_chunk = False
        first_cancelled_chunk = False
        first_diverted_chunk = False

        print(
            f"Processed {processed_rows:,} rows | "
            f"analysis rows written: {analysis_rows:,} | "
            f"cancelled rows written: {cancelled_rows:,} | "
            f"diverted rows written: {diverted_rows:,}"
        )

    print(
        "Finished. Wrote "
        f"{OUTPUT_ANALYSIS_FILE}, {OUTPUT_CANCELLED_FILE}, and {OUTPUT_DIVERTED_FILE}."
    )


if __name__ == "__main__":
    main()
