"""
Script to download OpenML dataset 45034 (delays_zurich_transport) and save it locally.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import urllib.request
import pandas as pd

DEFAULT_DATASET_ID = 45034
OPENML_API_URL = f"https://www.openml.org/api/v1/json/data/{DEFAULT_DATASET_ID}"


def fetch_dataset_metadata(dataset_id: int = DEFAULT_DATASET_ID) -> dict:
    """Fetch metadata for the given OpenML dataset ID."""
    url = f"https://www.openml.org/api/v1/json/data/{dataset_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "openml-downloader/1.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("data_set_description", {})


def download_file_with_progress(url: str, dest_path: Path) -> None:
    """Download a file with a progress bar using tqdm or standard urllib."""
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    req = urllib.request.Request(url, headers={"User-Agent": "openml-downloader/1.0"})
    with urllib.request.urlopen(req) as response:
        total_size = response.headers.get("Content-Length")
        total_bytes = int(total_size) if total_size else None
        chunk_size = 1024 * 1024  # 1MB chunks

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            if has_tqdm and total_bytes:
                with tqdm(total=total_bytes, unit="B", unit_scale=True, desc=dest_path.name) as pbar:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                downloaded = 0
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_bytes:
                        pct = (downloaded / total_bytes) * 100
                        print(f"\rDownloading: {downloaded / 1024 / 1024:.1f}/{total_bytes / 1024 / 1024:.1f} MB ({pct:.1f}%)", end="")
                print()


def download_and_save(
    dataset_id: int = DEFAULT_DATASET_ID,
    output_dir: str = "data",
    file_format: str = "parquet",
    method: str = "direct"
) -> Path:
    """
    Download OpenML dataset and save it in the requested format.

    Args:
        dataset_id: The OpenML dataset ID (default: 45034).
        output_dir: Destination folder.
        file_format: Desired output format ('parquet', 'csv', or 'both').
        method: Download method ('direct' for fast parquet download or 'sklearn').

    Returns:
        Path to the primary saved file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching metadata for OpenML dataset {dataset_id}...")
    metadata = fetch_dataset_metadata(dataset_id)
    dataset_name = metadata.get("name", f"dataset_{dataset_id}")
    parquet_url = metadata.get("parquet_url")
    print(f"Dataset Name: {dataset_name}")
    print(f"Description:  {metadata.get('description', '').splitlines()[0] if metadata.get('description') else 'N/A'}")

    start_time = time.time()
    saved_files = []

    if method == "direct" and parquet_url:
        parquet_file = out_dir / f"{dataset_name}_{dataset_id}.parquet"
        print(f"\nDownloading directly from OpenML Parquet store...")
        download_file_with_progress(parquet_url, parquet_file)
        saved_files.append(parquet_file)

        # If CSV or both requested, convert via pandas
        if file_format in ("csv", "both"):
            print("Converting to CSV format...")
            df = pd.read_parquet(parquet_file)
            csv_file = out_dir / f"{dataset_name}_{dataset_id}.csv"
            df.to_csv(csv_file, index=False)
            saved_files.append(csv_file)
            if file_format == "csv":
                # Remove intermediate parquet if only csv requested
                parquet_file.unlink()
                saved_files.remove(parquet_file)
    else:
        # Fallback or explicit sklearn fetch
        print(f"\nFetching via sklearn.datasets.fetch_openml(data_id={dataset_id})...")
        from sklearn.datasets import fetch_openml
        bunch = fetch_openml(data_id=dataset_id, as_frame=True)
        df = bunch.frame

        if file_format in ("parquet", "both"):
            parquet_file = out_dir / f"{dataset_name}_{dataset_id}.parquet"
            df.to_parquet(parquet_file, index=False)
            saved_files.append(parquet_file)

        if file_format in ("csv", "both"):
            csv_file = out_dir / f"{dataset_name}_{dataset_id}.csv"
            df.to_csv(csv_file, index=False)
            saved_files.append(csv_file)

    elapsed = time.time() - start_time
    print(f"\nDownload and save completed in {elapsed:.2f} seconds.")

    # Inspect the saved data
    sample_file = saved_files[0]
    if sample_file.suffix == ".parquet":
        df_summary = pd.read_parquet(sample_file)
    else:
        df_summary = pd.read_csv(sample_file, nrows=5)

    print("\nDataset Summary:")
    print(f"  Total Rows:    {len(df_summary):,}")
    print(f"  Total Columns: {len(df_summary.columns)}")
    print(f"  Columns:       {list(df_summary.columns)}")

    print("\nSaved files:")
    for path in saved_files:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  - {path.resolve()} ({size_mb:.2f} MB)")

    return saved_files[0]


def main():
    parser = argparse.ArgumentParser(
        description="Download OpenML dataset 45034 (delays_zurich_transport) and save it locally."
    )
    parser.add_argument(
        "--data-id",
        type=int,
        default=DEFAULT_DATASET_ID,
        help=f"OpenML dataset ID (default: {DEFAULT_DATASET_ID})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save the dataset (default: 'data')",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv", "both"],
        default="parquet",
        help="File format to save: 'parquet' (recommended, ~25MB), 'csv' (~300MB), or 'both' (default: parquet)",
    )
    parser.add_argument(
        "--method",
        choices=["direct", "sklearn"],
        default="direct",
        help="Download method: 'direct' (fast parquet stream) or 'sklearn' (fetch_openml) (default: direct)",
    )

    args = parser.parse_args()
    download_and_save(
        dataset_id=args.data_id,
        output_dir=args.output_dir,
        file_format=args.format,
        method=args.method,
    )


if __name__ == "__main__":
    main()
