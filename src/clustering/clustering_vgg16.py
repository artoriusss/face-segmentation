import numpy as np
from PIL import Image

import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tensorflow.keras.applications import vgg16
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.preprocessing import image

from .path import * 
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ClusterringVGG16():
    def __init__(self) -> None:
        self.weights = 'imagenet'
        self.include_top = False
        self.model = vgg16.VGG16(weights=self.weights, include_top=self.include_top)

    def load_image(self, image_path, target_size=(224, 224)):
        input_image = Image.open(image_path)
        resized_image = input_image.resize(target_size)
        return resized_image

    def process_images_in_parallel(self, load_image, image_paths):
        with ThreadPoolExecutor(max_workers=12) as executor:
            images = list(executor.map(load_image, image_paths))
        embeddings = self.get_image_embeddings(images)
        return embeddings

    def get_image_embeddings(self, images):
        # Convert list of PIL Images into a single numpy array with correct shape
        # Preprocess each image individually and stack
        preprocessed_images = np.vstack(
            [np.expand_dims(vgg16.preprocess_input(image.img_to_array(img)), axis=0) for img in images])

        # Verify the shape explicitly before model prediction
        # Expected shape: (batch_size, 224, 224, 3)
        print("Batch shape is:", preprocessed_images.shape)
        if preprocessed_images.shape[1:] != (224, 224, 3):
            raise ValueError("Incorrect image shape for VGG16 prediction")

        # Model prediction
        image_embeddings = self.model.predict(preprocessed_images)
        return image_embeddings

    def process_images_batch_pipeline(self, image_paths, batch_size=32):

        total_images = len(image_paths)
        all_embeddings = []

        for start_idx in range(0, total_images, batch_size):
            end_idx = start_idx + batch_size
            batch_paths = image_paths[start_idx:end_idx]
            batch_embeddings = self.process_images_in_parallel(batch_paths)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def get_embeddings(self, image_paths, batch_size=32):

        batch_size = 32  # Adjust based on your system's capabilities
        embeddings = self.process_images_batch_pipeline(image_paths, batch_size=batch_size)

    def save_embeddings(self, embeddings, output_path):
        np.save(output_path, embeddings)

    def kmeans(self, n_clusters, embeddings, random_state=42, init='k-means++', tol=1e-4, max_iter=300,
               n_init=10, n_jobs=-1):
        model = KMeans(n_clusters=n_clusters, random_state=random_state,
                       init=init, tol=tol, max_iter=max_iter, n_init=n_init, n_jobs=n_jobs)
        model = model.fit_predict(embeddings)

        return model

    def pca(self, embeddings, n_components=2):
        pca = PCA(n_components=n_components)
        pca_embeddings = pca.fit_transform(embeddings)
        return pca_embeddings
