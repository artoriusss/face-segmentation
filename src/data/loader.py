import requests
import os 
from .path import RAW_DATA
import tarfile
import shutil


class Dataset:
    def load(self, url):
        """
            Downloads and extracts the dataset from the URL
            Args:
                url (str): URL of the dataset
            Returns:
        """
        os.makedirs(RAW_DATA,exist_ok=True)
        tar_path = RAW_DATA / 'raw.tar'
        print('Loading dataset...')

        response = requests.get(url)
        open(tar_path, "wb").write(response.content)

        print('Extracting...')

        with tarfile.open(tar_path, 'r') as tar_ref:
            tar_ref.extractall(RAW_DATA)
        os.remove(tar_path)

        generic_path = RAW_DATA / 'wiki_crop'
        for item in generic_path.iterdir():
            shutil.move(item, RAW_DATA)
        os.rmdir(generic_path)
        print('Data loaded successfully.')
