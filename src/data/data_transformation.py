
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class DataTransformation:
  def __init__(self):
     self.scaler = StandardScaler()
  
  def scale_data(self,data):
      print('Scaling data...')
      normalized = data/255
      return self.scaler.fit_transform(normalized)
  
  def perform_pca(self, data):
     print('Performing PCA...')
     pca = PCA(n_components=50)
     pca.fit(data)
     tranformed_data = pca.transform(data)
     print(tranformed_data.shape)
     return (tranformed_data,pca)
  
  def plot_pca_components(self, pca):
   plt.figure(figsize=(12, 12))
   for i in range(min(pca.components_.shape[0], 25)):
      plt.subplot(5, 5, i + 1)
      plt.imshow(pca.components_[i].reshape((64,64)), cmap='gray')
      plt.title(f'Component {i + 1}')
      plt.axis('off')
   plt.show()
