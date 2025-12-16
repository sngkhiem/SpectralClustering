import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans

class SpectralClustering:
    def __init__(self, clusters=2, sigma=1.0, neighbors=10, randomState=143):
        self.clusters = clusters
        self.sigma = sigma
        self.neighbors = neighbors
        self.randomState = randomState
        self.labels_ = None
        self.eigenvectors_ = None
        self.eigenvalues_ = None
        self.W_ = None

    def generateMultiMoons(self, num=200, clusters=4, noise=0.05):
        n_pairs = int(np.ceil(clusters / 2))
        X_list = []
        y_list = []
        samples_per_pair = num // n_pairs
        
        for i in range(n_pairs):
            X_pair, y_pair = make_moons(n_samples=samples_per_pair, noise=noise, random_state=self.randomState + i)
            X_pair[:, 0] += (i * 3.0)
            y_pair += (i * 2)
            X_list.append(X_pair)
            y_list.append(y_pair)
            
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        
        if clusters % 2 != 0:
            mask = y < clusters
            X = X[mask]
            y = y[mask]
            
        return X, y

    def fit_predict(self, X):
        # Similarity Graph
        euDist = squareform(pdist(X, 'sqeuclidean'))
        S = np.exp(-euDist / (2 * self.sigma * self.sigma))

        # Weight Matrix
        knn = kneighbors_graph(X, n_neighbors=self.neighbors, mode='connectivity', include_self=False)
        W = knn.multiply(S).toarray()
        W = (W + W.T) / 2
        self.W_ = W

        # Degree Matrix
        D = np.diag(np.sum(W, axis=1))

        # Unnormalized Laplacian Matrix
        L = D - W

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = eigenvectors

        # Spectral Embedding
        U = eigenvectors[:, :self.clusters]

        # Perform KMeans on Spectral Embedding
        kmeans = KMeans(n_clusters=self.clusters, random_state=self.randomState)
        self.labels_ = kmeans.fit_predict(U)

        return self.labels_
    
    def compareToKMeans(self, X):
        kmeans = KMeans(n_clusters=self.clusters, random_state=self.randomState)
        ansKmeans = kmeans.fit_predict(X)

        if self.labels_ is None:
            self.fit_predict(X)

        plt.figure(figsize=(12, 5))
        # KMeans
        plt.subplot(1, 2, 1)
        plt.scatter(X[:, 0], X[:, 1], c=ansKmeans, s=30, cmap='viridis', edgecolor='k')
        plt.title("Standard K-Means")
        # Plot 2: Spectral Clustering
        plt.subplot(1, 2, 2)
        plt.scatter(X[:, 0], X[:, 1], c=self.labels_, s=30, cmap='viridis', edgecolor='k')
        plt.title("Spectral Clustering")
        plt.tight_layout()
        plt.show()

    def plotEigenvalues(self, num=5):
        if self.eigenvalues_ is None:
            print("Error: Run fit_predict() first.")
            return
        
        n = min(num, len(self.eigenvalues_))
        plt.figure(figsize=(10, 5))
        plt.scatter(range(1, n + 1), self.eigenvalues_[:n], marker='o', s=80, c='red', zorder=2)
        plt.plot(range(1, n + 1), self.eigenvalues_[:n], linestyle='--', color='gray', zorder=1)
        plt.title('Eigenvalues', fontsize=14)
        plt.xlabel('Index $\lambda_i$', fontsize=12)
        plt.ylabel('Eigenvalue Value', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(range(1, n + 1))
        plt.legend()
        plt.show()

    def plotEigenvectors(self, num=5, X=None):
        if self.eigenvectors_ is None:
            print("Error: Run fit_predict() first.")
            return
            
        n = min(num, self.eigenvectors_.shape[1])
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1: axes = [axes] 
        fig.suptitle(f'First {n} Eigenvectors', fontsize=16)
        for i in range(n):
            vec = self.eigenvectors_[:, i]
            if X is not None:
                sc = axes[i].scatter(X[:, 0], X[:, 1], c=vec, cmap='viridis', s=20, edgecolor='k', linewidth=0.5)
                axes[i].set_xticks([])
                axes[i].set_yticks([])
                plt.colorbar(sc, ax=axes[i], fraction=0.046, pad=0.04)
            else:
                axes[i].plot(vec, color='blue')
                axes[i].set_xlabel('Point Index')
                axes[i].grid(True, alpha=0.3)
            axes[i].set_title(f'Eigenvector $u_{i+1}$ ($\lambda_{i+1}$={self.eigenvalues_[i]:.4f})')

        plt.tight_layout()
        plt.show()

    def plotConnectivity(self, X):
        if self.W_ is None:
            print("Error: Run fit_predict() first.")
            return
        
        plt.figure(figsize=(10, 8))
        c = self.labels_ if self.labels_ is not None else 'blue'
        plt.scatter(X[:, 0], X[:, 1], c=c, cmap='viridis', s=50, zorder=2, edgecolor='k')
        rows, cols = np.where(np.triu(self.W_, k=1) > 0)
        from matplotlib.collections import LineCollection
        segments = [[X[r], X[c]] for r, c in zip(rows, cols)]
        lc = LineCollection(segments, colors='gray', linewidths=0.5, alpha=0.5, zorder=1)
        plt.gca().add_collection(lc)
        plt.title(f"Connectivity of data")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.show()