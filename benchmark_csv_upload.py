#!/usr/bin/env python3
"""
Benchmark script for CSV upload performance using everyrow SDK.

Usage:
    uv run python benchmark_csv_upload.py --rows 1000
    uv run python benchmark_csv_upload.py --rows 5000
    uv run python benchmark_csv_upload.py --sweep
"""

import argparse
import asyncio
import io
import time

import pandas as pd

from everyrow import create_session
from everyrow.ops import create_table_artifact


def generate_dataframe(num_rows: int, num_columns: int = 10) -> pd.DataFrame:
    data: dict[str, list] = {
        f"col_{i}": [f"value_{i}_{j}" for j in range(num_rows)]
        for i in range(num_columns)
    }
    data["numeric_col"] = list(range(num_rows))
    data["float_col"] = [i * 0.5 for i in range(num_rows)]
    return pd.DataFrame(data)


async def run_benchmark(num_rows: int):
    print(f"\n{'=' * 60}")
    print(f"Benchmarking dataset upload with {num_rows:,} rows")
    print(f"{'=' * 60}")

    print(f"\nGenerating DataFrame with {num_rows:,} rows...")
    gen_start = time.perf_counter()
    df = generate_dataframe(num_rows)
    gen_duration = time.perf_counter() - gen_start
    print(f"DataFrame generation: {gen_duration:.2f}s")

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_size_mb = len(csv_buffer.getvalue().encode("utf-8")) / (1024 * 1024)
    print(f"Estimated CSV size: {csv_size_mb:.2f} MB")

    print("\nUploading via everyrow SDK...")
    upload_start = time.perf_counter()
    try:
        async with create_session(name=f"benchmark-{num_rows}-rows") as session:
            print(f"Session URL: {session.get_url()}")
            artifact_id = await create_table_artifact(df, session)
            upload_duration = time.perf_counter() - upload_start
            print(f"Upload completed: {upload_duration:.2f}s")
            print(f"Artifact ID: {artifact_id}")

        rows_per_second = num_rows / upload_duration
        print(f"\nPerformance: {rows_per_second:.0f} rows/second")
    except Exception as e:
        upload_duration = time.perf_counter() - upload_start
        print(f"Upload FAILED after {upload_duration:.2f}s: {e}")
        raise

    return {
        "rows": num_rows,
        "csv_size_mb": csv_size_mb,
        "gen_duration": gen_duration,
        "upload_duration": upload_duration,
        "rows_per_second": rows_per_second,
    }


async def main():
    parser = argparse.ArgumentParser(description="Benchmark everyrow SDK upload performance")
    parser.add_argument(
        "--rows", type=int, default=1000, help="Number of rows to upload"
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run sweep across multiple row counts: 100, 500, 1000, 2000, 5000, 10000",
    )
    args = parser.parse_args()

    if args.sweep:
        row_counts = [100, 500, 1_000, 2_000, 5_000, 10_000, 50_000]
        results = []
        for rows in row_counts:
            result = await run_benchmark(rows)
            results.append(result)

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"{'Rows':>10} {'Size (MB)':>10} {'Time (s)':>10} {'Rows/sec':>12}")
        print("-" * 45)
        for r in results:
            print(
                f"{r['rows']:>10,} {r['csv_size_mb']:>10.2f} {r['upload_duration']:>10.2f} {r['rows_per_second']:>12,.0f}"
            )
    else:
        await run_benchmark(args.rows)


if __name__ == "__main__":
    asyncio.run(main())
