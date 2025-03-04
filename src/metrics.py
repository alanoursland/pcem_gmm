import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, silhouette_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

# ========================
# Clustering Metrics
# ========================

def calculate_ari(true_labels, predicted_labels):
    """
    Calculate Adjusted Rand Index (ARI) to evaluate clustering accuracy.
    :param true_labels: Ground truth labels
    :param predicted_labels: Predicted cluster labels
    :return: ARI score
    """
    return adjusted_rand_score(true_labels, predicted_labels)

def calculate_silhouette_score(data, predicted_labels):
    """
    Calculate Silhouette Score to evaluate the quality of clustering.
    :param data: Dataset used for clustering
    :param predicted_labels: Predicted cluster labels
    :return: Silhouette score
    """
    return silhouette_score(data, predicted_labels)

def compute_confusion_matrix(true_labels, predicted_labels):
    """
    Compute confusion matrix and clustering accuracy by aligning predicted clusters to true labels.
    :param true_labels: Ground truth labels
    :param predicted_labels: Predicted cluster labels
    :return: Confusion matrix and accuracy score
    """
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)  # Solve for best alignment
    reordered_conf_matrix = conf_matrix[row_ind][:, col_ind]  # Apply optimal label assignment
    accuracy = reordered_conf_matrix.trace() / reordered_conf_matrix.sum()
    return reordered_conf_matrix, accuracy

# ========================
# Model Evaluation Metrics
# ========================

def compare_means(true_means, estimated_means):
    """
    Compare the means of the ground truth and the estimated components.
    :param true_means: True means of the Gaussian components (1D array)
    :param estimated_means: Estimated means of the Gaussian components (1D array)
    :return: Euclidean distance between true and estimated means
    """
    return np.linalg.norm(true_means - estimated_means)  # No axis argument needed for 1D vectors

def compare_eigenvalues(true_eigenvalues, estimated_eigenvalues):
    """
    Compare the eigenvalues of the ground truth and the estimated components.
    :param true_eigenvalues: True eigenvalues of the Gaussian components
    :param estimated_eigenvalues: Estimated eigenvalues of the Gaussian components
    :return: Absolute difference between true and estimated eigenvalues
    """
    return np.abs(true_eigenvalues - estimated_eigenvalues)

def compare_eigenvectors(true_eigenvectors, estimated_eigenvectors):
    """
    Compare the eigenvectors of the ground truth and the estimated components.
    :param true_eigenvectors: True eigenvectors of the Gaussian components
    :param estimated_eigenvectors: Estimated eigenvectors of the Gaussian components
    :return: Angular difference between true and estimated eigenvectors
    """
    cosine_similarity = np.dot(true_eigenvectors, estimated_eigenvectors.T)
    return np.arccos(np.clip(cosine_similarity, -1, 1))

def compute_log_likelihood(data, model):
    """
    Compute the log-likelihood of the data under the current model.
    :param data: Data tensor of shape [n_samples, n_dimensions]
    :param model: Fitted GMM model
    :return: Log-likelihood value
    """
    return model.calculate_log_likelihood(data)

# ========================
# Visualization Functions
# ========================

def plot_covariance_ellipses(means, eigenvectors, eigenvalues, nstd=2):
    """
    Plot covariance ellipses to visually compare the true and estimated covariances.
    :param means: Mean of the Gaussian components
    :param eigenvectors: Eigenvectors of the covariance matrices
    :param eigenvalues: Eigenvalues of the covariance matrices
    :param nstd: Number of standard deviations for the ellipse size
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    for mean, eigenvector, eigenvalue in zip(means, eigenvectors, eigenvalues):
        # Convert eigenvalues and eigenvectors to a covariance matrix
        cov_matrix = eigenvector @ np.diag(eigenvalue) @ eigenvector.T
        evals, evecs = np.linalg.eigh(cov_matrix)
        angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))  # Convert to degrees
        width, height = 2 * nstd * np.sqrt(evals)  # Compute ellipse width & height
        ellip = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor='black', facecolor='none')
        ax.add_patch(ellip)
    ax.set_aspect('equal', 'box')
    plt.grid(True)
    plt.show()

# ========================
# Utility Functions
# ========================

def get_cluster_assignments(responsibilities):
    """
    Extract the cluster assignments (most likely component for each sample) from responsibilities.
    :param responsibilities: Tensor of shape (n_samples, n_components) with responsibility values
    :return: Tensor of cluster assignments for each sample
    """
    return torch.argmax(responsibilities, dim=1)
