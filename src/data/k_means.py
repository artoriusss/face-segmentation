from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class KMeansWrapper:
    def __init__(self):
        self.silouhette_score = []
        self.K_options=None

    def run(self,data):
        self.K_options = range(2, 50)

        for num_clusters in self.K_options:
            print(f"Number of clusters: {num_clusters}")
            clustering = KMeans(n_clusters=num_clusters,).fit(data)

            sklearn_silhouette_score = silhouette_score(data,clustering.labels_)
            self.silouhette_score.append(sklearn_silhouette_score)
            print(f"scikit-learn silhouette score: {sklearn_silhouette_score}")

    def run_selected(self, data, k):
        clustering = KMeans(n_clusters=k).fit(data)
        sklearn_silhouette_score = silhouette_score(data,clustering.labels_)
        self.silouhette_score.append(sklearn_silhouette_score)
        print(f"scikit-learn silhouette score: {sklearn_silhouette_score}")
        return clustering.labels_
    
    def plot(self):
        plt.plot(self.K_options, self.silouhette_score,marker='o')
        plt.ylabel('Custom silhouette score')
        plt.xlabel('Number of clusters')
        plt.xticks(self.K_options)
        plt.show()
