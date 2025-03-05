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
    # Create a ground truth GMM with 2 overlapping gaussians in 2D
    groundtruth = ComponentGMM(n_gaussians=2, n_dimensions=2, device=device)

    # Configure the first component (slightly shifted mean, different covariance)
    groundtruth.gaussians[0].set_mean(torch.tensor([-1.0, 1.0], device=device))
    groundtruth.gaussians[0].set_eigenvalues(torch.tensor([3.0, 0.5], device=device))

    # Configure the second component (overlapping mean, different shape)
    groundtruth.gaussians[1].set_mean(torch.tensor([1.0, -1.0], device=device))
    groundtruth.gaussians[1].set_eigenvalues(torch.tensor([2.5, 1.5], device=device))

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

    # Initialize ComponentGMM with 2 gaussians in 2D
    model = ComponentGMM(
        n_gaussians=2, 
        n_dimensions=2, 
        max_iterations=50, 
        tolerance=1e-6, 
        device=device
    )

    # # Simulate k-means initialization
    # model.gaussians[0].set_mean(torch.tensor([-1.0, 1.0], device=device))
    # model.gaussians[1].set_mean(torch.tensor([1.0, -1.0], device=device))

    # K-means initialization for means
    kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(samples)
    initial_means = torch.tensor(kmeans.cluster_centers_, device=device)
    model.gaussians[0].set_mean(initial_means[0])
    model.gaussians[1].set_mean(initial_means[1])

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
    ground_means = np.array([c.mean.cpu().numpy() for c in groundtruth.gaussians])
    ground_covariances = [
        c.eigenvectors.cpu().numpy() @ np.diag(c.eigenvalues.cpu().numpy()) @ c.eigenvectors.T.cpu().numpy()
        for c in groundtruth.gaussians
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
    pcem_means = np.array([c.mean.cpu().numpy() for c in model.gaussians])
    pcem_covariances = [
        c.eigenvectors.cpu().numpy() @ np.diag(c.eigenvalues.cpu().numpy()) @ c.eigenvectors.T.cpu().numpy()
        for c in model.gaussians
    ]
    print_model_parameters("PCEM-GMM", pcem_means, pcem_covariances)

    # Compare means, eigenvalues, and eigenvectors using metrics
    for i, component in enumerate(model.gaussians):
        print(f"\nComparing Component {i+1}:")
        mean_diff = compare_means(groundtruth.gaussians[i].mean.cpu().numpy(), component.mean.cpu().numpy())
        print(f"Mean Difference (Component {i+1}): {mean_diff}")

        eigenvalue_diff = compare_eigenvalues(groundtruth.gaussians[i].eigenvalues.cpu().numpy(), component.eigenvalues.cpu().numpy())
        print(f"Eigenvalue Difference (Component {i+1}): {eigenvalue_diff}")

        eigenvector_diff = compare_eigenvectors(groundtruth.gaussians[i].eigenvectors.T.cpu().numpy(), component.eigenvectors.T.cpu().numpy())
        print(f"Eigenvector Difference (Component {i+1}): {eigenvector_diff}")

    # Clustering metrics: ARI and Silhouette Score
    ari_score = calculate_ari(sample_source, component_assignments)
    silhouette_score = calculate_silhouette_score(samples, component_assignments)

    print(f"\nAdjusted Rand Index (ARI): {ari_score}")
    print(f"Silhouette Score: {silhouette_score}")

    # After fitting, before comparison
    def match_components(groundtruth, model):
        from scipy.spatial.distance import cdist
        means_true = torch.stack([c.mean for c in groundtruth.gaussians]).cpu().numpy()
        means_est = torch.stack([c.mean for c in model.gaussians]).cpu().numpy()
        dists = cdist(means_true, means_est)
        idx = np.argmin(dists, axis=1)  # Match each true to closest estimated
        return [model.gaussians[i] for i in idx], idx

    # Replace model1, model2 assignment
    matched_components, match_idx = match_components(groundtruth, model)
    model1, model2 = matched_components
    print(f"Component Matching Order: {match_idx}")

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
    def plot_cov_ellipse(pos, cov, nstd=2, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        evals, evecs = eigh(cov)
        angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
        width, height = 2 * nstd * np.sqrt(evals)
        ellip = Ellipse(xy=pos, width=width, height=height, angle=angle, **kwargs)
        ax.add_patch(ellip)
        return ellip
    
    plt.figure(figsize=(10, 8))
    plt.scatter(samples[:, 0], samples[:, 1], c=sample_source, cmap='coolwarm', alpha=0.5, label='True Labels')
    plt.scatter(samples[:, 0], samples[:, 1], c=component_assignments, cmap='coolwarm', alpha=0.3, label='PCEM-GMM Assignments')

    # Ground truth ellipses
    for i, c in enumerate(groundtruth.gaussians):
        cov = c.eigenvectors.cpu().numpy() @ np.diag(c.eigenvalues.cpu().numpy()) @ c.eigenvectors.T.cpu().numpy()
        plot_cov_ellipse(c.mean.cpu().numpy(), cov, alpha=0.3, color='green', label=f'True Cov {i+1}' if i == 0 else None)

    # PCEM ellipses
    for i, c in enumerate(matched_components):
        cov = c.eigenvectors.cpu().numpy() @ np.diag(c.eigenvalues.cpu().numpy()) @ c.eigenvectors.T.cpu().numpy()
        plot_cov_ellipse(c.mean.cpu().numpy(), cov, alpha=0.3, color='black', label=f'PCEM Cov {i+1}' if i == 0 else None)

    plt.legend()
    plt.title("PCEM-GMM Clustering on Overlapping 2D Gaussian Mixture")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()
