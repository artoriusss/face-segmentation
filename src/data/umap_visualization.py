from matplotlib import pyplot as plt
from umap import UMAP
from sklearn.preprocessing import StandardScaler


class UmapVisualization:
    def __init__(self, n_components=2, metric='cosine'):
        self.umap = UMAP(n_components=n_components, metric=metric)
        self.scaler = StandardScaler()
    
    def visualize(self, data):
        print('Umap visualization started...')
        transformed = self._transform(data)
        plt.figure(figsize=(8, 6))
        plt.scatter(transformed[:, 0], transformed[:, 1], c='b', marker='o')
        plt.title('UMAP Visualization of Image Embeddings')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.grid(True)
        plt.show()

    def _transform(self, data):
        print('Data is transforming...')
        transformed_data = self.umap.fit_transform(data)
        return transformed_data