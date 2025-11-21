import logging
import os
from os import path
from typing import Callable, Optional
import kagglehub as kh


def download_data(
    raw_data_path: str,
    download_func: Optional[Callable[[str], str]],
) -> str:
    '''
    Download data, using the provided download function.

    Args:
        raw_data_path: Destination directory for the downloaded data
        download_func: Function to perform the download.

    Returns:
        Path to the downloaded dataset

    Raises:
        ValueError: If raw_data_path is empty or download_func is empty
        FileNotFoundError: If the download fails or source file doesn't exist
    '''
    if not raw_data_path:
        raise ValueError('raw_data_path cannot be empty')
    if not download_func:
        raise ValueError('download_func cannot be empty')

    logging.info(f'>>> Downloading dataset ...')
    file_path = download_func()

    # Ensure destination directory exists
    os.makedirs(raw_data_path, exist_ok=True)

    # Move downloaded file to target directory
    new_file_path = path.join(raw_data_path, os.path.basename(file_path))
    os.rename(file_path, new_file_path)

    logging.info(f'>>> Data saved to: {new_file_path}')
    return new_file_path


def main() -> None:
    '''Main entry point for downloading data.'''
    def download_kaggle_data() -> str:
        dataset_id = 'ahmedlahlou/accidents-in-france-from-2005-to-2016'
        return kh.datasets.download_dataset(dataset_id)

    root_dir = path.dirname(path.dirname(path.abspath(__file__)))
    raw_data_path = path.join(root_dir, os.getenv('RAW_DATA_PATH', 'data/raw/'))
    # Create the destination directory if it doesn't exist
    os.makedirs(raw_data_path, exist_ok=True)
    download_data(raw_data_path, download_kaggle_data)

if __name__ == "__main__":
    main()