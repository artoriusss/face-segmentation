from pathlib import Path
import cv2
import numpy as np
from src.clustering.path import DATA_ALIGNED, DATA_PREPROCESSED
from src.data.data_transformation import DataTransformation
from src.clustering.k_means import KMeansWrapper
from src.data.loader import Dataset
from src.data.preprocess import Preprocesser
from src.clustering.deep_face_embedder import DeepFaceEmbedder
from src.clustering.visualize import Visualize

URL = "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar"

def main():
    # Dataset().load(URL)
    # preprocesser = Preprocesser()
    # preprocesser.preprocess()
    # preprocesser.preprocess(gray_scaled=False)

    # ----
    # FACES --> PCA --> K-MEANS

    # data_tranformation = DataTransformation()
    # data = np.array([(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)).flatten() for path in Path(DATA_ALIGNED).rglob('**/*.jpg')])

    # scaled_data = data_tranformation.scale_data(data)
    # pca_transformed_data, pca = data_tranformation.perform_pca(scaled_data)
    # k_means = KMeansWrapper()
    # labels = k_means.run_selected(pca_transformed_data, 10)
    # visualizer = Visualize()
    # visualizer.visualize(data, labels)
    # visualizer.visualize_average(data, labels)

    # -----
    # FACES --> PCA --> EMBEDDINGS --> K-MEANS

    emb = DeepFaceEmbedder()
    embedings = emb.get_embeddings()
    embedings = emb.unpack_embeddings(embedings)
    clusters = emb.kmeans(embedings, k=7)
    # K_options, sklearn_silhouette_score = emb.run_kmeans_silouhette_score(embedings)
    # emb.plot_silouhette_score(K_options, sklearn_silhouette_score)
    emb.visualise_clusters(clusters)
    emb.visualise_average(clusters)


if __name__ == '__main__':
    main()