import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from pcem import ComponentGMM
from metrics import (
    compare_means, compare_eigenvalues, compare_eigenvectors, 
    calculate_log_likelihood, calculate_AIC_BIC, 
    calculate_mahalanobis_distance)
from utils import Timer


def generate_eigenvalues(dim, midpoint=0.5, steepness=10):
    """Generate eigenvalues for each Gaussian using an exponential decay."""
    eigenvalues = 1 / (1 + np.exp(-steepness * (np.linspace(0, 1, dim) - midpoint)))
    eigenvalues = eigenvalues / eigenvalues.sum()  # Normalize to avoid too large eigenvalues
    return torch.tensor(eigenvalues, dtype=torch.float32)


def random_rotation_matrix(dim):
    """Generate a random orthonormal matrix for eigenvectors."""
    mat = np.random.randn(dim, dim)
    q, _ = np.linalg.qr(mat)  # QR decomposition gives orthonormal matrix
    return torch.tensor(q, dtype=torch.float32)


def create_groundtruth_from_params(params, device="cpu"):
    """
    Generate a groundtruth Gaussian model based on provided parameters.
    
    :param params: A dictionary containing parameters like dimensionality, decay_midpoint, decay_steepness, and component_count
    :param device: The device to run on ('cpu' or 'cuda')
    :return: A ComponentGaussian model with parameters as defined in `params`
    """
    # Extract parameters from input
    dimensionality = params["dimensionality"]
    decay_midpoint = params["decay_midpoint"]
    decay_steepness = params["decay_steepness"]
    component_count = params["component_count"]
    
    # Generate the eigenvalues based on decay function
    eigenvalues = generate_eigenvalues(dimensionality, midpoint=decay_midpoint, steepness=decay_steepness)
    
    # Limit eigenvalues to the first `component_count` components
    eigenvalues = eigenvalues[:component_count]
    
    # Generate random eigenvectors (orthogonal matrix)
    eigenvectors = random_rotation_matrix(dimensionality)
    
    # Create the groundtruth Gaussian with the generated eigenvalues and eigenvectors
    mean = torch.zeros(dimensionality, device=device)
    
    # Return the generated groundtruth Gaussian model
    return ComponentGMM(n_gaussians=1, n_dimensions=dimensionality, device=device)


def experiment(model, samples, num_samples, dimensionality, groundtruth, threshold_percentage=0.99, max_iter=100, device="cpu"):
    """
    Run the experiment for different models (PCEM-GMM, PCA-GMM, GMM) with shared data (samples).
    
    :param model: The model to train (e.g., PCEM-GMM, PCA-GMM, GMM)
    :param groundtruth: The groundtruth model (optional). If provided, generate samples from this groundtruth.
    :param samples: Pre-sampled data (optional). If provided, skip groundtruth-based sampling.
    :param num_samples: Number of samples to generate or use.
    :param dimensionality: Dimensionality of the dataset.
    :param threshold_percentage: The variance threshold for principal component extraction.
    :param max_iter: Maximum number of iterations for fitting the model.
    :param device: Device for computation ('cpu' or 'cuda').
    :return: Dictionary of results for comparison.
    """
    timer = Timer()

    # Step 1: Fit the model using the provided samples
    print("Running fit")
    model.fit(samples)

    # Step 2: Evaluate model performance
    elapsed_time = timer.elapsed()

    # Log-likelihood and convergence (if applicable)
    print("calculate_log_likelihood")
    log_likelihood = calculate_log_likelihood(samples, model)
    
    # AIC and BIC calculations
    print("calculate_AIC_BIC")
    AIC, BIC = calculate_AIC_BIC(log_likelihood, num_samples, model)

    # Mahalanobis Distance
    print("calculate_mahalanobis_distance")
    mahalanobis_distances = calculate_mahalanobis_distance(samples, model, device=device)

    # Covariance Matrix Comparison (if groundtruth is provided)
    print("Covariance Matrix Comparison")
    if hasattr(groundtruth, 'covariances_') or hasattr(groundtruth, 'eigenvalues'):
        true_covariance = groundtruth.covariances_ if hasattr(groundtruth, 'covariances_') else _compute_pcem_covariance_matrix(groundtruth)
        cov_matrix_diff = compare_covariance_matrices(true_covariance, model)
    else:
        cov_matrix_diff = None

    # For simplicity, we will compare means, eigenvalues, and eigenvectors if the model supports these parameters
    print("Compare stats")
    if hasattr(model, 'means') and hasattr(model, 'eigenvalues') and hasattr(model, 'eigenvectors'):
        mean_diff = compare_means(groundtruth.mean.cpu().numpy(), model.means.cpu().numpy())
        eigenvalue_diff = compare_eigenvalues(groundtruth.eigenvalues.cpu().numpy(), model.eigenvalues.cpu().numpy())
        eigenvector_diff = compare_eigenvectors(groundtruth.eigenvectors.cpu().numpy(), model.eigenvectors.cpu().numpy())
    else:
        mean_diff = eigenvalue_diff = eigenvector_diff = None

    # Collect results in a dictionary for easy comparison
    results = {
        "dimensionality": dimensionality,
        "num_samples": num_samples,
        "log_likelihood": log_likelihood,
        "AIC": AIC,
        "BIC": BIC,
        "elapsed_time": elapsed_time,
        "mahalanobis_distances": mahalanobis_distances,
        "cov_matrix_diff": cov_matrix_diff,
        "mean_diff": mean_diff,
        "eigenvalue_diff": eigenvalue_diff,
        "eigenvector_diff": eigenvector_diff,
        "threshold_percentage": threshold_percentage
    }

    return results

def print_results(results_name, results):
    """
    Print the results of the experiment in a human-readable format.

    :param results: Dictionary containing the results of the experiment.
    """

    print("======================================================")
    print(results_name)

    # Extracting the individual metrics from results
    log_likelihood = results["log_likelihood"]
    AIC = results["AIC"]
    BIC = results["BIC"]
    mahalanobis_distances = results["mahalanobis_distances"]
    dimensionality = results["dimensionality"]
    num_samples = results["num_samples"]
    elapsed_time = results["elapsed_time"]

    # Display high-level results
    print(f"--- Experiment Results ---")
    print(f"Dimensionality: {dimensionality}")
    print(f"Sample Size: {num_samples}")
    print(f"Log-Likelihood: {log_likelihood}")
    print(f"AIC: {AIC}")
    print(f"BIC: {BIC}")
    print(f"Elapsed Time: {elapsed_time:.4f} seconds")
    print()

    # Mahalanobis Distance Statistics
    print("Mahalanobis Distance Statistics:")
    # Flatten the Mahalanobis distances (since it's num_samples x num_components)
    flattened_distances = mahalanobis_distances.flatten()

    mean_md = torch.mean(flattened_distances)
    std_md = torch.std(flattened_distances)
    min_md = torch.min(flattened_distances)
    max_md = torch.max(flattened_distances)

    print(f"  Mean Mahalanobis Distance: {mean_md:.4f}")
    print(f"  Std of Mahalanobis Distance: {std_md:.4f}")
    print(f"  Min Mahalanobis Distance: {min_md:.4f}")
    print(f"  Max Mahalanobis Distance: {max_md:.4f}")

    print()

    # Mahalanobis distance statistics per component
    print("Mahalanobis Distance Statistics per Component:")
    num_components = mahalanobis_distances.shape[1]  # num_components is the second dimension
    for i in range(num_components):
        component_distances = mahalanobis_distances[:, i]  # Get distances for the i-th component
        mean_comp = torch.mean(component_distances)
        std_comp = torch.std(component_distances)
        min_comp = torch.min(component_distances)
        max_comp = torch.max(component_distances)
        
        print(f"  Component {i+1}:")
        print(f"    Mean: {mean_comp:.4f}")
        print(f"    Std: {std_comp:.4f}")
        print(f"    Min: {min_comp:.4f}")
        print(f"    Max: {max_comp:.4f}")
    
    print("\n")

# Groundtruth parameter configurations for experimentation
gt_params = [
    {"dimensionality": 10, "decay_midpoint": 0.5, "decay_steepness": 10, "component_count": 8},
    {"dimensionality": 100, "decay_midpoint": 0.5, "decay_steepness": 100, "component_count": 16},
    {"dimensionality": 1000, "decay_midpoint": 0.5, "decay_steepness": 100, "component_count": 32},
    # {"dimensionality": 10000, "decay_midpoint": 0.5, "decay_steepness": 100, "component_count": 64},
]

# Dimensionality and sample sizes for experimentation
sample_sizes = [500, 1000, 5000]

import numpy as np

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loop through the groundtruth configurations
    for gt_param in gt_params:
        print(f"Running experiment against groundtruth: {gt_param}")

        # Generate the groundtruth based on parameters
        groundtruth = create_groundtruth_from_params(gt_param, device=device)
        
        for num_samples in sample_sizes:
            samples, _ = groundtruth.sample(num_samples)

            print(f"Running experiment for {num_samples} samples...")

            # Initialize models
            print("Creating models")
            pcem_model = ComponentGMM(n_gaussians=1, n_dimensions=gt_param["dimensionality"], max_iterations=100, device=device)
            pca_gmm_model = GaussianMixture(n_components=1, covariance_type='full', max_iter=100)
            gmm_model = GaussianMixture(n_components=1, covariance_type='full', max_iter=100)
            
            # # Run PCEM-GMM experiment
            # results_pcem = experiment(pcem_model, samples=samples, num_samples=num_samples, dimensionality=gt_param["dimensionality"], groundtruth=groundtruth, device=device)
            # print(f"PCEM-GMM Results:", results_pcem)

            # Run PCA-GMM experiment
            print("Moving samples to CPU")
            cpu_samples = samples.cpu().numpy()
            print("Running PCA-GMM experiment")
            results_pca_gmm = experiment(pca_gmm_model, samples=cpu_samples, num_samples=num_samples, dimensionality=gt_param["dimensionality"], groundtruth=groundtruth)
            print_results(f"PCA-GMM Results", results_pca_gmm)

            # Run standard GMM experiment
            print("Running standard GMM experiment")
            results_gmm = experiment(gmm_model, samples=cpu_samples, num_samples=num_samples, dimensionality=gt_param["dimensionality"], groundtruth=groundtruth)
            print_results(f"GMM Results", results_gmm)
