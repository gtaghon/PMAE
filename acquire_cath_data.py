#!/usr/bin/env python3
"""
acquire_cath_data.py - Download mdCATH dataset files from HuggingFace

This script downloads a specified number of mdCATH dataset files from the
HuggingFace repository and saves them to a local directory. It supports
parallel downloading for efficiency.

Usage:
    python acquire_cath_data.py --output_dir ./mdcath_data --num_samples 100 --num_workers 8
"""

import os
import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from huggingface_hub import HfApi, hf_hub_download


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Download mdCATH dataset files from HuggingFace')
    parser.add_argument('--output_dir', type=str, default='./mdcath_data',
                        help='Directory to save downloaded files')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of dataset files to download')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel download threads')
    parser.add_argument('--repo_id', type=str, default='compsciencelab/mdCATH',
                        help='HuggingFace repository ID')
    parser.add_argument('--subfolder', type=str, default='data',
                        help='Subfolder in the repository containing the dataset files')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()


def list_available_files(repo_id, subfolder='data'):
    """List all available files in the HuggingFace repository."""
    api = HfApi()
    try:
        file_list = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        dataset_files = [f for f in file_list if f.startswith(subfolder) and f.endswith('.h5')]
        return dataset_files
    except Exception as e:
        print(f"Error listing files from repository: {e}")
        return []


def download_file(file_info):
    """Download a single file from HuggingFace."""
    repo_id, file_path, output_dir = file_info
    
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        return local_path, True
    except Exception as e:
        print(f"Error downloading {file_path}: {e}")
        return file_path, False


def download_dataset(repo_id, file_list, output_dir, num_workers=4):
    """Download files in parallel using ThreadPoolExecutor."""
    os.makedirs(output_dir, exist_ok=True)
    
    download_tasks = [(repo_id, file_path, output_dir) for file_path in file_list]
    
    successful_downloads = 0
    failed_downloads = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(download_file, task): task[1] for task in download_tasks}
        
        with tqdm(total=len(download_tasks), desc="Downloading files") as pbar:
            for future in as_completed(futures):
                file_path, success = future.result()
                if success:
                    successful_downloads += 1
                else:
                    failed_downloads += 1
                pbar.update(1)
    
    return successful_downloads, failed_downloads


def main():
    """Main function to download mdCATH dataset files."""
    args = parse_arguments()
    random.seed(args.seed)
    
    print(f"Listing available files from {args.repo_id}...")
    available_files = list_available_files(args.repo_id, args.subfolder)
    
    if not available_files:
        print("No files found in the repository. Please check the repository ID and subfolder.")
        return
    
    print(f"Found {len(available_files)} files in the repository.")
    
    if args.num_samples > len(available_files):
        print(f"Requested {args.num_samples} samples, but only {len(available_files)} files are available.")
        selected_files = available_files
    else:
        selected_files = random.sample(available_files, args.num_samples)
    
    print(f"Selected {len(selected_files)} files for download.")
    
    start_time = time.time()
    successful, failed = download_dataset(
        args.repo_id,
        selected_files,
        args.output_dir,
        args.num_workers
    )
    
    elapsed_time = time.time() - start_time
    
    print("\nDownload Summary:")
    print(f"Successfully downloaded: {successful} files")
    print(f"Failed downloads: {failed} files")
    print(f"Total time: {elapsed_time:.2f} seconds")
    
    # Create a file list for easy reference
    file_list_path = os.path.join(args.output_dir, "file_list.txt")
    with open(file_list_path, 'w') as f:
        for file_path in selected_files:
            local_filename = os.path.basename(file_path)
            f.write(f"{local_filename}\n")
    
    print(f"File list saved to {file_list_path}")


if __name__ == "__main__":
    main()