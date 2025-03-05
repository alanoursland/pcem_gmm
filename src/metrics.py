import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import torch

from matplotlib.patches import Ellipse
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import adjusted_rand_score, silhouette_score, confusion_matrix
from numpy.linalg import pinv

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

def compute_clustering_metrics(true_labels, predicted_labels):
    """
    Compute clustering metrics: confusion matrix and clustering accuracy.
    :param true_labels: Ground truth labels (true cluster assignments)
    :param predicted_labels: Predicted cluster assignments from the model
    :return: confusion_matrix, accuracy
    """
    # Compute confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    
    # Use the Hungarian algorithm (linear_sum_assignment) to optimally align clusters
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)  # Solve for best alignment
    
    # Reorder the confusion matrix for optimal cluster matching
    reordered_conf_matrix = conf_matrix[row_ind][:, col_ind]
    
    # Calculate accuracy (correctly predicted points divided by total points)
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

# ========================
# More Metrics
# ========================

def calculate_log_likelihood(samples, model):
    """
    Calculate the log-likelihood of the data given the model.

    :param samples: (n_samples, n_dimensions) array or tensor of data samples
    :param model: Fitted model (PCEM-GMM, PCA-GMM, GMM)
    :return: Log-likelihood of the data
    """
    if hasattr(model, 'score_samples'):  # For PCA-GMM and GMM, use built-in function
        log_likelihood = np.sum(model.score_samples(samples))
    else:  # For PCEM-GMM, calculate log-likelihood manually
        log_likelihood = _calculate_pcem_log_likelihood(samples, model)  # PCEM-GMM-specific
    return log_likelihood


def _calculate_pcem_log_likelihood(samples, model):
    """
    Calculate log-likelihood for PCEM-GMM. This requires computing Mahalanobis distances
    and using the component-wise mixture model likelihood formula.

    :param samples: Data samples to calculate log-likelihood for
    :param model: Fitted PCEM-GMM model
    :return: Log-likelihood of the data for PCEM-GMM
    """
    n_samples, n_dim = samples.shape
    log_likelihood = 0

    for i in range(len(model.gaussians)):
        # Covariance matrix for component i (constructed from eigenvectors and eigenvalues)
        cov_matrix = _compute_pcem_covariance_matrix(model.gaussians[i])
        
        # Inverse and determinant of covariance matrix
        cov_inv = np.linalg.inv(cov_matrix)
        log_det_cov = np.linalg.slogdet(cov_matrix)[1]

        for x in samples:
            diff = x - model.gaussians[i].mean
            mahalanobis_dist = np.dot(np.dot(diff.T, cov_inv), diff)
            
            # Log-likelihood for this sample under component i
            log_likelihood += np.log(model.mixing_coeffs[i]) - 0.5 * (n_dim * np.log(2 * np.pi) + log_det_cov + mahalanobis_dist)
    
    return log_likelihood

def calculate_AIC_BIC(log_likelihood, n_samples, model):
    """
    Calculate AIC and BIC for the model.

    :param log_likelihood: Log-likelihood of the model
    :param n_samples: Number of samples in the dataset
    :param model: Fitted model (PCEM-GMM, PCA-GMM, GMM)
    :return: AIC and BIC values
    """
    k = _count_model_parameters(model)  # Count the model parameters
    AIC = 2 * k - 2 * log_likelihood
    BIC = np.log(n_samples) * k - 2 * log_likelihood
    return AIC, BIC


def _count_model_parameters(model):
    """
    Count the number of free parameters in the model for AIC/BIC calculation.
    
    :param model: Fitted model (PCEM-GMM, PCA-GMM, GMM)
    :return: Number of parameters in the model
    """
    if hasattr(model, 'covariances_'):  # For PCA-GMM and GMM
        n_params = model.n_components * (model.means_.shape[1] + model.means_.shape[1] * (model.means_.shape[1] + 1) // 2 + 1)
    else:  # For PCEM-GMM
        n_params = model.n_components * (model.means_.shape[1] + model.means_.shape[1] * (model.means_.shape[1] + 1) // 2 + 1) + model.n_components
    return n_params

import numpy as np
from scipy.spatial.distance import mahalanobis

def calculate_mahalanobis_distance(samples, gmm_model, device="cpu"):
    """
    Calculate the Mahalanobis distance for each sample in the dataset to each Gaussian component.

    :param samples: (n_samples, n_dimensions) array or tensor of data samples
    :param gmm_model: Fitted model (PCEM-GMM, PCA-GMM, GMM)
    :return: Mahalanobis distances of samples to each Gaussian component, shape: (n_samples, n_components)
    """
    if isinstance(samples, np.ndarray):
        samples = torch.tensor(samples, dtype=torch.float32).to(device)
    if hasattr(gmm_model, 'covariances_'):  # For PCA-GMM and GMM, use covariance matrices directly
        means = torch.tensor(gmm_model.means_, dtype=torch.float32).to(device)
        covariances = [torch.tensor(cov_matrix, dtype=torch.float32).to(device) for cov_matrix in gmm_model.covariances_]

        n_samples = samples.shape[0]
        n_components = gmm_model.n_components
        n_dims = samples.shape[1]

        # Precompute the inverse covariance matrices for each component
        cov_inv = [torch.pinverse(cov_matrix) for cov_matrix in covariances]  # (n_components, n_dimensions, n_dimensions)

        # Initialize the distances tensor for all samples and components
        distances = torch.zeros((n_samples, n_components), device=device)

        # Center the samples by subtracting the means
        centered_samples = samples.unsqueeze(1) - means  # Shape: (n_samples, n_components, n_dims)

        # Compute the Mahalanobis distances using batch matrix multiplication
        for j in range(n_components):
            cov_inv_matrix = cov_inv[j]  # Inverse covariance matrix for the j-th component

            # For each component, calculate the Mahalanobis distance
            diff = centered_samples[:, j, :]
            mahalanobis_sq = torch.sum(torch.matmul(diff, cov_inv_matrix) * diff, dim=1)

            # Store the squared Mahalanobis distance for each sample and component
            distances[:, j] = torch.sqrt(mahalanobis_sq)

        return distances
    elif hasattr(gmm_model, 'calculate_mahalanobis_distances'):  # For PCEM-GMM, use custom method
        return gmm_model.calculate_mahalanobis_distances(samples)
    else:
        raise NotImplementedError("Mahalanobis distance calculation is not implemented for this model.")
    
def compare_covariance_matrices(true_covariance, model):
    """
    Compare the covariance matrix of the model to the true covariance matrix.

    :param true_covariance: True covariance matrix
    :param model: Fitted model (PCEM-GMM, PCA-GMM, GMM)
    :return: Frobenius norm of the difference between the covariance matrices
    """
    if hasattr(model, 'covariances_'):  # For PCA-GMM and GMM, use covariance matrices directly
        estimated_covariance = model.covariances_
    else:  # For PCEM-GMM, compute covariance from eigenvalues/eigenvectors
        estimated_covariance = _compute_pcem_covariance_matrix(model)  # PCEM-GMM-specific
    
    # Compute Frobenius norm of the difference
    return np.linalg.norm(true_covariance - estimated_covariance)

def _compute_pcem_covariance_matrix(model):
    """
    Compute the covariance matrix for PCEM-GMM from eigenvalues and eigenvectors.

    :param model: Fitted PCEM-GMM model
    :return: Covariance matrix for PCEM-GMM
    """
    # For each component, we need to construct the covariance matrix from the eigenvalues and eigenvectors
    covariance_matrices = []
    for i in range(model.n_components):
        eigenvalues = model.gaussians[i].eigenvalues_
        eigenvectors = model.gaussians[i].eigenvectors_
        
        # Reconstruct covariance matrix from eigenvalues and eigenvectors
        cov_matrix = np.zeros((eigenvectors.shape[0], eigenvectors.shape[0]))
        for j in range(len(eigenvalues)):
            cov_matrix += eigenvalues[j] * np.outer(eigenvectors[:, j], eigenvectors[:, j])
        covariance_matrices.append(cov_matrix)
    
    return np.array(covariance_matrices)

