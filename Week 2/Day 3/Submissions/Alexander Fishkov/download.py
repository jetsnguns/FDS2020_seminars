import argparse
import os
import numpy as np
import pandas as pd
import tarfile
import urllib.request
import zipfile
from glob import glob


def download_flights_data(url, download_dir, data_dir):
    """
    Download New York flights dataset
    :param url: download link to the archive file
    :param data_dir: path relative to the current working directory to download and extract dataset to
    :return: 
    """
    download_path = os.path.join(download_dir, 'nycflights.tar.gz')

    print("** Start downloading **")

    if not os.path.exists(download_dir):
        os.mkdir(download_dir)

    if not os.path.exists(download_path):
        print("- Downloading NYC Flights dataset... ", end='', flush=True)
        urllib.request.urlretrieve(url, download_path)
        print("done", flush=True)
    else:
        print("- File {} already exists, skipping download".format(download_path))

    if not os.path.exists(os.path.join(data_dir, 'nycflights')):
        print("- Extracting flight data... ", end='', flush=True)
        with tarfile.open(download_path, mode='r:gz') as flights:
            flights.extractall(data_dir)
        print("done", flush=True)
    else:
        print("- Data directory {} already exists, skipping extraction".format(data_dir))

    print("** Finished downloading **")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download raw dataset and store it in a dedicated directory.')
    parser.add_argument('-u', '--url', action="store", dest='url', type=str, required=True)
    parser.add_argument('-d', '--download_dir', action="store", dest='download_dir', type=str, required=True)
    parser.add_argument('-o', '--output_dir', action="store", dest='data_dir', type=str, required=True)

    args = parser.parse_args()

    download_flights_data(args.url, args.download_dir, args.data_dir)
