from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os


K = 16 # Number of Clusters


def kmeans(x, k=K, min_iters=30):
    """
    Carry out k-means algorithm until converge.

    Args:
        k: Number of clusters, int.
        x: Data matrix, of shape (m, n).
        iters: Minimum iterations that it should take, int.

    Returns:
        An array of cluster labels, shape of (m, n).
    """
    eps = 1e-3 # Convergence threshold
    dis = prev_dis = None
    it = 0
    # Initialize centroids mu and labels c
    m, n = x.shape
    idx = np.random.choice(m, size=k, replace=False)
    mu = x[idx]
    c_labels = np.zeros(m) # The indices of centroids

    while it < min_iters or (prev_dis is None or np.abs(dis - prev_dis) > eps):
        # Update cluster lables
        dist_matrix = np.linalg.norm(x[:, None, :] - mu[None, :, :], ord=2, axis=2)
        c_labels = np.argmin(dist_matrix, axis=1)
        # Update centroids
        sum = np.zeros((k, n))
        counts = np.zeros((k, 1))
        # The same as mu[c[i]] += x[i]
        np.add.at(sum, c_labels, x)
        np.add.at(counts, c_labels, 1)
        mu = sum / counts
        # Update the loop condition
        it += 1
        prev_dis = dis
        dis = np.sum(np.linalg.norm(x - mu[c_labels], ord=2, axis=1), axis=0)

    return mu, c_labels


if __name__ == '__main__':
    # Load the data
    A_small = imread("./../data/peppers-small.tiff")
    plt.imshow(A_small)
    save_path1 = os.path.join('output', 'peppers-small.png')
    plt.savefig(save_path1)
    
    A_large = imread("./../data/peppers-large.tiff")
    plt.imshow(A_large)
    save_path2 = os.path.join('output', 'peppers-large.png')
    plt.savefig(save_path2)

    h, w, c = A_small.shape
    X_small = A_small.reshape(-1, 3)
    X_large = A_large.reshape(-1, 3)
    # Run K-means algorithm
    mu, labels = kmeans(X_small, k=K)
    
    # Find the closest centroids
    dist_matrix_large = np.linalg.norm(X_large[:, None, :] - mu[None, :, :], axis=2)
    labels_large = np.argmin(dist_matrix_large, axis=1)
    compressed_large = mu[labels_large].reshape(A_large.shape)

    # Export the figure
    plt.imshow(compressed_large.astype(np.uint8))
    save_path = os.path.join('output', 'compressed-large.png')
    plt.savefig(save_path)

    print("All figures are saved in ./output directory.")
