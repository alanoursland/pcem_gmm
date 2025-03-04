import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.patches import Ellipse
from pcem import ComponentGaussian, ComponentGMM
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from metrics import compare_means, compare_eigenvalues, compare_eigenvectors, calculate_ari, calculate_silhouette_score

def create_groundtruth():
    # Create a ground truth GMM with 2 components in 2D
    groundtruth = ComponentGMM(n_components=2, n_dimensions=2, device=device)

    # Configure the first component (elongated shape)
    groundtruth.components[0].set_mean(torch.tensor([-5.0, 2.0], device=device))
    groundtruth.components[0].set_eigenvalues(torch.tensor([6.0, 1.0], device=device))

    # Configure the second component (spherical shape)
    groundtruth.components[1].set_mean(torch.tensor([5.0, -3.0], device=device))
    groundtruth.components[1].set_eigenvalues(torch.tensor([2.0, 2.0], device=device))

    return groundtruth

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_samples = 1000

    # Create a ground truth GMM and get samples
    groundtruth = create_groundtruth()
    samples, labels = groundtruth.sample(1000)

    print("\n[Experiment] Running PCEM-GMM on synthetic 2D Gaussian mixture data...\n")

    # Initialize ComponentGMM with 2 components in 2D
    model = ComponentGMM(
        n_components=2, 
        n_dimensions=2, 
        max_iterations=50, 
        tolerance=1e-6, 
        device=device
    )

    # K-means initialization for means
    kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(samples.cpu().numpy())
    initial_means = torch.tensor(kmeans.cluster_centers_, device=device)
    model.components[0].set_mean(initial_means[0])
    model.components[1].set_mean(initial_means[1])

    # Fit the model to the data
    model.fit(samples)

    # model components for convenience
    ground1 = groundtruth.components[0]
    ground2 = groundtruth.components[1]
    model1 = model.components[0]
    model2 = model.components[1]

    # Print iteration details and log likelihood
    print(f"\n[EM Iteration]\t[Log Likelihood]\t[Change in Log Likelihood]")
    for i, log_likelihood in enumerate(model.log_likelihood_history):
        if i > 0:
            change = log_likelihood - model.log_likelihood_history[i-1]
            print(f"{i+1}\t{log_likelihood:.4f}\t{change:.4f}")
        else:
            print(f"{i+1}\t{log_likelihood:.4f}\tN/A")

    if model.converged_:
        print(f"\nConverged after {model.n_iterations_} iterations!")
    else:
        print(f"\nReached maximum iterations ({model.max_iterations}) without convergence.")

    # Compare Recovered vs. Ground Truth
    print("\n=== Final Comparison ===")
    print("True Means:")
    print("Gaussian 1:", ground1.mean.cpu().numpy())
    print("Gaussian 2:", ground2.mean.cpu().numpy())
    print("Estimated Means:")
    print("Model 1:", model1.mean.cpu().numpy())
    print("Model 2:", model2.mean.cpu().numpy())

    print("\nTrue Eigenvalues:")
    print("Gaussian 1:", ground1.eigenvalues.cpu().numpy())
    print("Gaussian 2:", ground2.eigenvalues.cpu().numpy())
    print("Estimated Eigenvalues:")
    print("Model 1:", model1.eigenvalues.cpu().numpy())
    print("Model 2:", model2.eigenvalues.cpu().numpy())

    print("\nTrue Eigenvectors (row based):")
    print("Gaussian 1:", ground1.eigenvectors.T.cpu().numpy())
    print("Gaussian 2:", ground2.eigenvectors.T.cpu().numpy())
    print("Estimated Eigenvectors (row based):")
    print("Model 1:", model1.eigenvectors.T.cpu().numpy())
    print("Model 2:", model2.eigenvectors.T.cpu().numpy())

    # Compare means, eigenvalues, and eigenvectors using metrics
    mean_diff = compare_means(ground1.mean.cpu().numpy(), model1.mean.cpu().numpy())
    print(f"Mean Difference (Model 1): {mean_diff}")
    mean_diff = compare_means(ground2.mean.cpu().numpy(), model2.mean.cpu().numpy())
    print(f"Mean Difference (Model 2): {mean_diff}")

    eigenvalue_diff = compare_eigenvalues(ground1.eigenvalues.cpu().numpy(), model1.eigenvalues.cpu().numpy())
    print(f"Eigenvalue Difference (Model 1): {eigenvalue_diff}")
    eigenvalue_diff = compare_eigenvalues(ground2.eigenvalues.cpu().numpy(), model2.eigenvalues.cpu().numpy())
    print(f"Eigenvalue Difference (Model 2): {eigenvalue_diff}")

    eigenvector_diff = compare_eigenvectors(ground1.eigenvectors.T.cpu().numpy(), model1.eigenvectors.T.cpu().numpy())
    print(f"Eigenvector Difference (Model 1): {eigenvector_diff}")
    eigenvector_diff = compare_eigenvectors(ground2.eigenvectors.T.cpu().numpy(), model2.eigenvectors.T.cpu().numpy())
    print(f"Eigenvector Difference (Model 2): {eigenvector_diff}")

    # Clustering metrics: ARI and Silhouette Score
    responsibilities, _ = groundtruth.e_step(samples)
    sample_source = torch.argmax(responsibilities, dim=1)
    ari_score = calculate_ari(sample_source.cpu().numpy(), model.get_cluster_assignments(samples).cpu().numpy())
    silhouette_score = calculate_silhouette_score(samples.cpu(), model.get_cluster_assignments(samples).cpu().numpy())

    print(f"\nAdjusted Rand Index (ARI): {ari_score}")
    print(f"Silhouette Score: {silhouette_score}")

    # Plot results
    # Get model data assignments
    responsibilities, _ = groundtruth.e_step(samples)
    component_assignments = torch.argmax(responsibilities, dim=1)

    # ======== Create the Plot ========
    plt.figure(figsize=(10, 8))

    # Plot data points colored by original source for reference
    plt.scatter(samples[:, 0].cpu().numpy(), samples[:, 1].cpu().numpy(), 
                c=sample_source.cpu().numpy(), cmap='coolwarm', alpha=0.5, label='Data Points (True Source)')

    # Overlay estimated component assignments
    plt.scatter(samples[:, 0].cpu().numpy(), samples[:, 1].cpu().numpy(), 
                c=component_assignments.cpu().numpy(), cmap='coolwarm', alpha=0.3, label='Estimated Assignments')

    # ======== Plot Means ========
    # Estimated means (model)
    plt.scatter(model1.mean[0].cpu().numpy(), model1.mean[1].cpu().numpy(), 
                color='black', marker='x', s=200, label='Estimated Mean 1', linewidths=2)
    plt.scatter(model2.mean[0].cpu().numpy(), model2.mean[1].cpu().numpy(), 
                color='yellow', marker='x', s=200, label='Estimated Mean 2', linewidths=2)

    # True means (ground truth)
    plt.scatter(ground1.mean[0].cpu().numpy(), ground1.mean[1].cpu().numpy(), 
                color='green', marker='+', s=200, label='True Mean 1', linewidths=2)
    plt.scatter(ground2.mean[0].cpu().numpy(), ground2.mean[1].cpu().numpy(), 
                color='red', marker='+', s=200, label='True Mean 2', linewidths=2)

    # ======== Ellipse Plotting Functions ========
    def plot_cov_ellipse(pos, cov, nstd=2, ax=None, **kwargs):
        """
        Plots an ellipse representing the covariance matrix cov centered at pos.
        :param pos: Center of the ellipse (mean).
        :param cov: Covariance matrix.
        :param nstd: Number of standard deviations.
        :param ax: Axis object.
        :param kwargs: Additional arguments for Ellipse.
        """
        if ax is None:
            ax = plt.gca()
            
        evals, evecs = eigh(cov)  # Eigen decomposition of covariance matrix
        angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))  # Convert to degrees

        width, height = 2 * nstd * np.sqrt(evals)  # Compute ellipse width & height
        ellip = Ellipse(xy=pos, width=width, height=height, angle=angle, **kwargs)
        ax.add_patch(ellip)
        return ellip

    def eigen_to_cov(eigenvectors, eigenvalues):
        """
        Convert eigen decomposition (eigenvectors & eigenvalues) into covariance matrix.
        """
        return sum(eigenvalues[i] * np.outer(eigenvectors[:, i], eigenvectors[:, i]) for i in range(len(eigenvalues)))

    # ======== Plot Covariance Ellipses ========
    try:
        # True covariance matrices
        cov1_true = eigen_to_cov(ground1.eigenvectors.cpu().numpy(), ground1.eigenvalues.cpu().numpy())
        cov2_true = eigen_to_cov(ground2.eigenvectors.cpu().numpy(), ground2.eigenvalues.cpu().numpy())

        # Estimated covariance matrices
        cov1_est = eigen_to_cov(model1.eigenvectors.cpu().numpy(), model1.eigenvalues.cpu().numpy())
        cov2_est = eigen_to_cov(model2.eigenvectors.cpu().numpy(), model2.eigenvalues.cpu().numpy())

        # Plot true covariance ellipses
        plot_cov_ellipse(ground1.mean.cpu().numpy(), cov1_true, alpha=0.3, color='green', label='True Cov 1')
        plot_cov_ellipse(ground2.mean.cpu().numpy(), cov2_true, alpha=0.3, color='red', label='True Cov 2')

        # Plot estimated covariance ellipses
        plot_cov_ellipse(model1.mean.cpu().numpy(), cov1_est, alpha=0.3, color='black', label='Est Cov 1')
        plot_cov_ellipse(model2.mean.cpu().numpy(), cov2_est, alpha=0.3, color='yellow', label='Est Cov 2')

    except Exception as e:
        print(f"Warning: Could not plot covariance ellipses. Error: {e}")

    # ======== Step 6: Final Plot Customization & Show ========
    plt.legend()
    plt.title("PCEM-GMM Clustering on Synthetic 2D Gaussian Mixture")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()
