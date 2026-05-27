"""
download.py

Downloads MAVEN-ERE dataset files.
Works on Mac, Linux, and Windows.

Usage:
    python data/download.py
"""

import urllib.request
import zipfile
import os
from pathlib import Path


def download_file(url, output_path, description):
    """Download a file from a URL with a progress indicator."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"  Already exists, skipping: {output_path}")
        return

    print(f"Downloading {description}...")
    print(f"  URL: {url}")
    print(f"  Destination: {output_path}")

    def progress_hook(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            print(f"\r  Progress: {percent:.1f}%", end='', flush=True)

    urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
    print()  # newline after progress
    print(f"  Done: {output_path}")


def unzip_file(zip_path, extract_to):
    """Unzip a file into a target directory."""
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)

    if not zip_path.exists():
        print(f"  Zip file not found: {zip_path}")
        return

    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"  Extracted to: {extract_to}")


def main():
    # Paths
    raw_dir = Path("data/raw/")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # ── Download MAVEN-ERE ────────────────────────────────────────────────
    maven_zip = raw_dir / "MAVEN_ERE.zip"
    download_file(
        url="https://cloud.tsinghua.edu.cn/f/a7d1db6c44ea458bb6f0/?dl=1",
        output_path=maven_zip,
        description="MAVEN-ERE dataset"
    )
    unzip_file(maven_zip, raw_dir)

    

    print("\nAll downloads complete.")
    print("Data is in data/raw/")
    


if __name__ == '__main__':
    main()