from pathlib import Path
from typing import Optional

import duckdb


def process_bts_month(
    csv_path: Path,
    parquet_path: Path,
    expected_year: int,
    expected_month: int,
    reference_parquet_path: Optional[Path] = None,
) -> dict:
    
    """
    Validate one monthly BTS Reporting Carrier CSV
    and convert it to compressed Parquet.
    """

    con = duckdb.connect()

    csv_path = Path(csv_path)
    parquet_path = Path(parquet_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    con.execute(
        f"""
        CREATE OR REPLACE VIEW monthly_flights AS
        SELECT *
        FROM read_csv_auto(
            '{csv_path}',
            sample_size = 100000,
            ignore_errors = true
        )
        """
    )

        # Validate schema against a trusted reference month
    if reference_parquet_path is not None:
        reference_parquet_path = Path(reference_parquet_path)

        if not reference_parquet_path.exists():
            raise FileNotFoundError(
                f"Reference Parquet not found: {reference_parquet_path}"
            )

        current_schema = con.sql("""
            DESCRIBE monthly_flights
        """).df()["column_name"].tolist()

        reference_schema = con.sql(
            f"""
            DESCRIBE
            SELECT *
            FROM read_parquet('{reference_parquet_path}')
            """
        ).df()["column_name"].tolist()

        if current_schema != reference_schema:
            raise ValueError(
                "Schema mismatch between source CSV "
                "and reference dataset."
            )

    validation = con.sql(
        """
        SELECT
            COUNT(*) AS total_rows,
            MIN(FlightDate) AS first_date,
            MAX(FlightDate) AS last_date,
            COUNT(DISTINCT FlightDate) AS distinct_days,
            MIN(Year) AS min_year,
            MAX(Year) AS max_year,
            MIN(Month) AS min_month,
            MAX(Month) AS max_month
        FROM monthly_flights
        """
    ).df().iloc[0]

    if (
        validation["min_year"] != expected_year
        or validation["max_year"] != expected_year
    ):
        raise ValueError("Unexpected year in source data.")

    if (
        validation["min_month"] != expected_month
        or validation["max_month"] != expected_month
    ):
        raise ValueError("Unexpected month in source data.")

    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    con.execute(
        f"""
        COPY (
            SELECT *
            FROM monthly_flights
        )
        TO '{parquet_path}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
        """
    )

    parquet_rows = con.sql(
        f"""
        SELECT COUNT(*) AS total_rows
        FROM read_parquet('{parquet_path}')
        """
    ).df().iloc[0]["total_rows"]

    if parquet_rows != validation["total_rows"]:
        raise ValueError(
            "Parquet row count does not match CSV row count."
        )

    csv_size_mb = csv_path.stat().st_size / 1024**2
    parquet_size_mb = parquet_path.stat().st_size / 1024**2

    return {
        "year": expected_year,
        "month": expected_month,
        "rows": int(validation["total_rows"]),
        "first_date": validation["first_date"],
        "last_date": validation["last_date"],
        "days": int(validation["distinct_days"]),
        "csv_size_mb": round(csv_size_mb, 2),
        "parquet_size_mb": round(parquet_size_mb, 2),
        "storage_reduction_pct": round(
            (1 - parquet_size_mb / csv_size_mb) * 100,
            1,
        ),
    }