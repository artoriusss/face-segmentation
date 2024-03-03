import os
import numpy as np
import cv2
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from deepface import DeepFace
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import random

from .path import * 

class DeepFaceEmbedder():
    def get_embeddings(self):

        img_folder = str(DATA_ALIGNED)
        img_path = os.listdir(img_folder)
        img_path = random.sample(img_path, 1000)

        output_folder = str(INVERSE_PATH)
        os.makedirs(output_folder, exist_ok=True)

        flattened_images = []

        for img_name in tqdm(img_path, desc="Flattening images"):
            img_full_path = os.path.join(img_folder, img_name)
            img = cv2.imread(img_full_path, cv2.IMREAD_GRAYSCALE)  
            if img is not None:
                flattened_images.append(img.flatten())
            else:
                print(f"Image {img_name} could not be read and will be skipped.")

        flattened_images = np.array(flattened_images)

        pca = PCA(n_components=0.80, whiten=True)
        print("Applying PCA to images...")
        transformed_images = pca.fit_transform(flattened_images)

        inverse_transformed_image_paths = []

        print("Saving inverse-transformed images...")
        for i, transformed_img in enumerate(tqdm(transformed_images, desc="Inverse transforming images")):
            img_name = f"inverse_transformed_{i}.png"
            img_full_path = os.path.join(output_folder, img_name)
            approx_image = pca.inverse_transform(transformed_img).reshape(256, 256)
            approx_image = approx_image.astype(np.uint8)
            cv2.imwrite(img_full_path, approx_image)
            inverse_transformed_image_paths.append(img_full_path)

        models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "Dlib", "ArcFace"]
        embeddings = []

        print("Computing embeddings from saved images...")
        for img_path in tqdm(inverse_transformed_image_paths, desc="Computing embeddings"):
            embedding = DeepFace.represent(img_path, model_name=models[0], enforce_detection=False)
            if embedding is not None:
                embeddings.append(embedding)

        return embeddings

    def unpack_embeddings(self, embeddings):
        embs = []
        for emb in embeddings:
            embs.append(emb[0]['embedding'])
        return embs
    
    def run_kmeans_silouhette_score(self,embeddings):
        K_options = range(2, 50)
        silouhette_scores = []

        for num_clusters in K_options:
            print(f"Number of clusters: {num_clusters}")
            clustering = KMeans(n_clusters=num_clusters,).fit(embeddings)

            sklearn_silhouette_score = silhouette_score(embeddings,clustering.labels_)
            silouhette_scores.append(sklearn_silhouette_score)
            print(f"scikit-learn silhouette score: {sklearn_silhouette_score}")
        return K_options, silouhette_scores
    
    def plot_silouhette_score(self, K_options, sklearn_silhouette_score):
        plt.plot(K_options, sklearn_silhouette_score,marker='o')
        plt.ylabel('Custom silhouette score')
        plt.xlabel('Number of clusters')
        plt.xticks(K_options)
        plt.show()

    def kmeans(self, embeddings,k):
        image_paths = os.listdir(str(DATA_ALIGNED))
        kmeans = KMeans(n_clusters=k, random_state=42).fit(embeddings)
        labels = kmeans.labels_

        image_clusters = {label: [] for label in set(labels)}
        for path, cluster_label in zip(image_paths, labels):
            image_clusters[cluster_label].append(path)
        return image_clusters

    def visualise_clusters(self, image_clusters):
        number_to_display_per_cluster = 5  # Number of images to display per cluster
        ncols = 7  # Number of columns in the grid

        for cluster_label, images_in_cluster in image_clusters.items():
            sample_size = min(len(images_in_cluster), number_to_display_per_cluster)
            images_to_display = random.sample(images_in_cluster, sample_size) if len(images_in_cluster) > number_to_display_per_cluster else images_in_cluster
            nrows_per_cluster = -(-len(images_to_display) // ncols)  # Ceiling division to get the number of rows
            
            fig, axes = plt.subplots(nrows=nrows_per_cluster, ncols=ncols, figsize=(20, 5 * nrows_per_cluster))
            
            if nrows_per_cluster * ncols > 1:
                axes = axes.flatten()
            else:
                axes = [axes]

            for ax, img_path in zip(axes, images_to_display):
                img_path = f'{str(DATA_ALIGNED)}/{img_path}'
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
                    ax.imshow(img)
                    ax.set_title(f"Cluster {cluster_label}")
                else:
                    print(f"Warning: could not read image {img_path}. Skipping.")
                ax.axis('off')  # Turn off axes
            
            # Hide any unused subplots in case the last row is not complete
            for ax in axes[len(images_to_display):]:
                ax.axis('off')

            plt.tight_layout()
            plt.show()

    def visualise_average(self, image_clusters):
        for cluster_label, image_paths_in_cluster in image_clusters.items():
            imgs = np.array([(cv2.imread(os.path.join(str(DATA_ALIGNED), path), cv2.IMREAD_GRAYSCALE)).flatten() for path in image_paths_in_cluster])
            avg = np.mean(imgs, axis=0)
            print(avg.shape)
            plt.imshow(avg.reshape((256,256)), cmap='gray')
            plt.title(f"Cluster {cluster_label}")
            plt.axis('off')
            plt.show()