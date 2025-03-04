import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
from pcem import ComponentGaussian, ComponentGMM
from metrics import compare_means, compare_eigenvalues, compare_eigenvectors, calculate_ari, calculate_silhouette_score

def create_groundtruth():
    # Create a ground truth GMM with 2 overlapping components in 2D
    groundtruth = ComponentGMM(n_components=2, n_dimensions=2, device=device)

    # Configure the first component (slightly shifted mean, different covariance)
    groundtruth.components[0].set_mean(torch.tensor([-1.0, 1.0], device=device))
    groundtruth.components[0].set_eigenvalues(torch.tensor([3.0, 0.5], device=device))

    # Configure the second component (overlapping mean, different shape)
    groundtruth.components[1].set_mean(torch.tensor([1.0, -1.0], device=device))
    groundtruth.components[1].set_eigenvalues(torch.tensor([2.5, 1.5], device=device))

    return groundtruth

def compute_clustering_metrics(true_labels, predicted_labels):
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)  # Solve for best alignment
    reordered_conf_matrix = conf_matrix[row_ind][:, col_ind]  # Apply optimal label assignment
    accuracy = reordered_conf_matrix.trace() / reordered_conf_matrix.sum()
    return reordered_conf_matrix, accuracy

def print_model_parameters(name, means, covariances):
    print(f"\n{name} Parameters:")
    for i, (mean, cov) in enumerate(zip(means, covariances)):
        print(f"Component {i+1}:")
        print(f"  Mean: {mean}")
        print(f"  Eigenvalues: {np.linalg.eigvalsh(cov)}")
        print(f"  Eigenvectors:\n{np.linalg.eigh(cov)[1]}")

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_samples = 1000

    # Create a ground truth GMM and get samples
    groundtruth = create_groundtruth()
    samples, sample_source = groundtruth.sample(num_samples)
    samples = samples.cpu().numpy()
    sample_source = sample_source.cpu().numpy()

    print("\n[Experiment] Running PCEM-GMM on overlapping 2D Gaussian mixture data...\n")

    # Initialize ComponentGMM with 2 components in 2D
    model = ComponentGMM(
        n_components=2, 
        n_dimensions=2, 
        max_iterations=50, 
        tolerance=1e-6, 
        device=device
    )

    # # Simulate k-means initialization
    # model.components[0].set_mean(torch.tensor([-1.0, 1.0], device=device))
    # model.components[1].set_mean(torch.tensor([1.0, -1.0], device=device))

    # K-means initialization for means
    kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(samples)
    initial_means = torch.tensor(kmeans.cluster_centers_, device=device)
    model.components[0].set_mean(initial_means[0])
    model.components[1].set_mean(initial_means[1])

    # Fit the model to the data
    model.fit(torch.tensor(samples, device=device))

    # ====== Groundtruth: CEM-GMM ======
    print(" ========================= Groundtruth: CEM-GMM ========================= ")
    # Compute clustering metrics for Groundtruth
    responsibilities, _ = groundtruth.e_step(torch.tensor(samples, device=device))
    component_assignments = torch.argmax(responsibilities, dim=1).cpu().numpy()
    conf_matrix, accuracy = compute_clustering_metrics(sample_source, component_assignments)
    print("Groundtruth Confusion Matrix:\n", conf_matrix)
    print("Groundtruth Clustering Accuracy:", accuracy)

    # Extract ground truth parameters
    ground_means = np.array([c.mean.cpu().numpy() for c in groundtruth.components])
    ground_covariances = [
        c.eigenvectors.cpu().numpy() @ np.diag(c.eigenvalues.cpu().numpy()) @ c.eigenvectors.T.cpu().numpy()
        for c in groundtruth.components
    ]    
    print_model_parameters("Ground Truth GMM", ground_means, ground_covariances)

    # ====== Experiment: CEM-GMM ======
    print(" ========================= Experiment: CEM-GMM ========================= ")
    # Compute clustering metrics for PCEM-GMM
    responsibilities, _ = model.e_step(torch.tensor(samples, device=device))
    component_assignments = torch.argmax(responsibilities, dim=1).cpu().numpy()
    conf_matrix, accuracy = compute_clustering_metrics(sample_source, component_assignments)
    print("PCEM-GMM Confusion Matrix:\n", conf_matrix)
    print("PCEM-GMM Clustering Accuracy:", accuracy)

    # Extract PCEM-GMM parameters
    pcem_means = np.array([c.mean.cpu().numpy() for c in model.components])
    pcem_covariances = [
        c.eigenvectors.cpu().numpy() @ np.diag(c.eigenvalues.cpu().numpy()) @ c.eigenvectors.T.cpu().numpy()
        for c in model.components
    ]
    print_model_parameters("PCEM-GMM", pcem_means, pcem_covariances)

    # Compare means, eigenvalues, and eigenvectors using metrics
    for i, component in enumerate(model.components):
        print(f"\nComparing Component {i+1}:")
        mean_diff = compare_means(groundtruth.components[i].mean.cpu().numpy(), component.mean.cpu().numpy())
        print(f"Mean Difference (Component {i+1}): {mean_diff}")

        eigenvalue_diff = compare_eigenvalues(groundtruth.components[i].eigenvalues.cpu().numpy(), component.eigenvalues.cpu().numpy())
        print(f"Eigenvalue Difference (Component {i+1}): {eigenvalue_diff}")

        eigenvector_diff = compare_eigenvectors(groundtruth.components[i].eigenvectors.T.cpu().numpy(), component.eigenvectors.T.cpu().numpy())
        print(f"Eigenvector Difference (Component {i+1}): {eigenvector_diff}")

    # Clustering metrics: ARI and Silhouette Score
    ari_score = calculate_ari(sample_source, component_assignments)
    silhouette_score = calculate_silhouette_score(samples, component_assignments)

    print(f"\nAdjusted Rand Index (ARI): {ari_score}")
    print(f"Silhouette Score: {silhouette_score}")

    # ====== Baseline: Standard GMM ======
    print(" ========================= Baseline: Standard GMM  ========================= ")
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    gmm.fit(samples)
    gmm_assignments = gmm.predict(samples)
    conf_matrix_gmm, accuracy_gmm = compute_clustering_metrics(sample_source, gmm_assignments)
    print("\nStandard GMM Confusion Matrix:\n", conf_matrix_gmm)
    print("Standard GMM Clustering Accuracy:", accuracy_gmm)
    print_model_parameters("Standard GMM", gmm.means_, gmm.covariances_)


    # Calculate Silhouette Score for Standard GMM
    silhouette_score_gmm = calculate_silhouette_score(samples, gmm_assignments)
    print(f"Standard GMM Silhouette Score: {silhouette_score_gmm}")

    # ====== Baseline: PCA-GMM ======
    print(" ========================= Baseline: PCA-GMM  ========================= ")
    pca = PCA(n_components=2)
    reduced_samples = pca.fit_transform(samples)
    pca_gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    pca_gmm.fit(reduced_samples)
    pca_gmm_assignments = pca_gmm.predict(reduced_samples)
    conf_matrix_pca_gmm, accuracy_pca_gmm = compute_clustering_metrics(sample_source, pca_gmm_assignments)
    print("\nPCA-GMM Confusion Matrix:\n", conf_matrix_pca_gmm)
    print("PCA-GMM Clustering Accuracy:", accuracy_pca_gmm)
    print_model_parameters("PCA-GMM", pca_gmm.means_, pca_gmm.covariances_)

    # Calculate Silhouette Score for PCA-GMM
    silhouette_score_pca_gmm = calculate_silhouette_score(reduced_samples, pca_gmm_assignments)
    print(f"PCA-GMM Silhouette Score: {silhouette_score_pca_gmm}")

    # # Plot results
    plt.figure(figsize=(10, 8))
    plt.scatter(samples[:, 0], samples[:, 1], c=sample_source, cmap='coolwarm', alpha=0.5, label='True Labels')
    plt.scatter(samples[:, 0], samples[:, 1], c=component_assignments, cmap='coolwarm', alpha=0.3, label='PCEM-GMM Assignments')
    plt.legend()
    plt.title("PCEM-GMM Clustering on Overlapping 2D Gaussian Mixture")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()
