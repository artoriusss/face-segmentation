import requests
import os 
from .path import RAW_DATA
from pathlib import Path
import tarfile
import shutil

URL = "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar"

class Dataset:
    def load(self):
        tar_path = RAW_DATA / 'raw.tar'
        print('Loading dataset...')

        response = requests.get(URL)
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
