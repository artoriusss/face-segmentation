from src.data.loader import Dataset
from src.data.preprocess import Preprocesser

URL = "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar"

def main():
    #Dataset().load(URL)
    Preprocesser().prt()

if __name__ == '__main__':
    main()