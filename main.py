from src.data.loader import Dataset
from src.data.preprocess import Preprocesser
from src.clustering.deep_face_embedder import DeepFaceEmbedder

URL = "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar"

def main():
    #Dataset().load(URL)
    # preprocesser = Preprocesser()
    # preprocesser.preprocess()
    # preprocesser.preprocess(gray_scaled=False)

    # FACES --> PCA --> EMBEDDINGS --> K-MEANS

    emb = DeepFaceEmbedder()
    embedings = emb.get_embeddings()
    embedings = emb.unpack_embeddings(embedings)
    clusters = emb.kmeans(embedings)
    emb.visualise_clusters(clusters)


if __name__ == '__main__':
    main()