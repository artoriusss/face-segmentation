from matplotlib import pyplot as plt
import numpy as np

number_to_display_per_cluster = 5  # Display 5 images per cluster, for example
ncols = 5  # Number of columns in the grid

class Visualize():
    def visualize(self, data, labels):
        for cluster_label in np.unique(labels):
            images_in_cluster = data[labels == cluster_label]
            sample_size = min(len(images_in_cluster), number_to_display_per_cluster)
            images_to_display = images_in_cluster[np.random.choice(images_in_cluster.shape[0], sample_size, replace=False)] if len(images_in_cluster) > number_to_display_per_cluster else images_in_cluster
            nrows_per_cluster = (len(images_to_display) + ncols - 1) // ncols

            fig, axes = plt.subplots(nrows=nrows_per_cluster, ncols=ncols, figsize=(20, 5 * nrows_per_cluster))
            if nrows_per_cluster == 1:
                axes = [axes]

            axes = [ax for sublist in axes for ax in (sublist if isinstance(sublist, np.ndarray) else [sublist])]
            for ax, img_data in zip(axes, images_to_display):
                ax.imshow(img_data.reshape((256,256)),cmap='gray')
                ax.set_title(f"Cluster {cluster_label}")
                ax.axis('off')  # Hide axes ticks

            for ax in axes[len(images_to_display):]:
                ax.axis('off')

        plt.tight_layout()
        plt.show()

    def visualize_average(self, data, labels):
        for cluster_label in np.unique(labels):
            images_in_cluster = data[labels == cluster_label]
            avg = np.mean(images_in_cluster, axis=0)
            print(avg.shape)
            plt.imshow(avg.reshape((256,256)), cmap='gray')
            plt.title(f"Cluster {cluster_label}")
            plt.axis('off')
            plt.show()
        