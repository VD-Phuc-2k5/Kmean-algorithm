import cv2
import os
import numpy as np
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from yellowbrick.cluster import SilhouetteVisualizer

data_path = r'D:\KMean\test\flowers'

def get_feature(img):
    if img is None:
        return None
    intensity = img.sum(axis=1)
    intensity = intensity.sum(axis=0) / (255 * img.shape[0] * img.shape[1])
    return intensity

def create_pickle(data_path=data_path):
    X = []
    L = []
    for file in os.listdir(data_path):
        img_path = os.path.join(data_path, file)
        
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(img_path)
            c_x = get_feature(img)
            if c_x is not None:  
                X.append(c_x)
                L.append(file)

    X = np.array(X)
    L = np.array(L)

    with open('data.pickle', 'wb') as handle:
        pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('label.pickle', 'wb') as handle:
        pickle.dump(L, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Pickle files created successfully!")
    
    return X, L

def load_data():
    try:
        with open('data.pickle', 'rb') as handle:
            X = pickle.load(handle)
        with open('label.pickle', 'rb') as handle:
            L = pickle.load(handle)
        return X, L
    except (FileNotFoundError, EOFError):
        print("Pickle files not found or corrupted. Creating new data.")
        return create_pickle()

def main():
    X, L = load_data()
    print(f"Feature shape: {X.shape}")

    # cluster_range = list(range(2, 31))  

    # for k in cluster_range:
    #     km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=100, random_state=42)
        
    #     plt.figure(figsize=(10, 6))
    #     visualizer = SilhouetteVisualizer(km, colors='yellowbrick')
    #     visualizer.fit(X)

    #     plt.title(f'Silhouette Visualization for {k} Clusters')
    #     plt.tight_layout()
    #     plt.show()

    n_clusters = 21
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)

    print("Cluster centers:", kmeans.cluster_centers_)

    n_row, n_col = 6, 6
    for i in range(n_clusters):
        _, axs = plt.subplots(n_row, n_col, figsize=(7, 7))
        axs = axs.flatten()
        images = L[kmeans.labels_ == i][:36]
        for img, ax in zip(images, axs):
            img_path = os.path.join(data_path, img)
            img_data = mpimg.imread(img_path)
            ax.imshow(img_data)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()