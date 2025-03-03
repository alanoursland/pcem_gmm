import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from matplotlib.patches import Ellipse
from pcem import ComponentGaussian
from scipy.linalg import eigh

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d = 2  # Dimensionality
    num_samples = 500

    # Define first Gaussian (elongated shape)
    groundtruth1 = ComponentGaussian(2, device=device)
    groundtruth1.set_mean(torch.tensor([-5.0, 2.0], device=device))
    groundtruth1.set_eigenvalues(torch.tensor([6.0, 1.0], device=device))

    # Define second Gaussian (spherical shape)
    groundtruth2 = ComponentGaussian(2, device=device)
    groundtruth2.set_mean(torch.tensor([5.0, -3.0], device=device))
    groundtruth2.set_eigenvalues(torch.tensor([2.0, 2.0], device=device))

    # Generate samples
    samples1 = groundtruth1.sample(num_samples)
    samples2 = groundtruth2.sample(num_samples)
    
    # Combine samples into dataset
    samples = torch.cat([samples1, samples2], dim=0)
    sample_source = torch.cat([torch.zeros(num_samples), torch.ones(num_samples)])  # For visualization only: 0 for Gaussian 1, 1 for Gaussian 2

    print("\n[Experiment] Running PCEM-GMM on synthetic 2D Gaussian mixture data...\n")

    # Initialize two-component PCEM-GMM
    model1 = ComponentGaussian(2, device=device)
    model2 = ComponentGaussian(2, device=device)

    # Initialize mixing coefficients (priors)
    mixing_coeffs = torch.tensor([0.5, 0.5], device=device)  # Equal initial mixture weights

    # EM algorithm
    max_iterations = 20
    stop_tolerance = 1e-4
    prev_log_likelihood = float('-inf')

    for iteration in range(max_iterations):
        print(f"\n[EM Iteration {iteration+1}]")

        # ======== E-Step: Compute responsibilities ========
        likelihoods1 = model1.calculate_responsibilities(samples)  # (n_samples, 1)
        likelihoods2 = model2.calculate_responsibilities(samples)  # (n_samples, 1)

        # Apply mixture weights
        weighted_likelihoods1 = likelihoods1 * mixing_coeffs[0]
        weighted_likelihoods2 = likelihoods2 * mixing_coeffs[1]

        # Compute total likelihood per sample (avoid zero division)
        total_likelihoods = weighted_likelihoods1 + weighted_likelihoods2  # (n_samples, 1)

        # Compute responsibilities (normalize across components)
        responsibilities1 = weighted_likelihoods1 / (total_likelihoods + 1e-6)
        responsibilities2 = weighted_likelihoods2 / (total_likelihoods + 1e-6)

        # ======== M-step: Update component parameters ========
        model1.fit(samples, responsibilities1)
        model2.fit(samples, responsibilities2)

        # Update mixing coefficients
        mixing_coeffs[0] = responsibilities1.mean()
        mixing_coeffs[1] = responsibilities2.mean()

        # Compute log likelihood for convergence check (ensure stability)
        current_log_likelihood = torch.log(torch.clamp(total_likelihoods, min=1e-6)).sum()
        log_likelihood_change = current_log_likelihood - prev_log_likelihood

        print(f"  Log Likelihood: {current_log_likelihood.item():.4f}")
        print(f"  Change in Log Likelihood: {log_likelihood_change.item() if iteration > 0 else 'N/A'}")

        # Check for convergence
        if iteration > 0 and abs(log_likelihood_change.item()) < stop_tolerance * abs(prev_log_likelihood):
            print(f"\nConverged after {iteration+1} iterations!")
            break

        prev_log_likelihood = current_log_likelihood


    # Step 3: Compare Recovered vs. Ground Truth
    print("\n=== Final Comparison ===")
    print("True Means:")
    print("Gaussian 1:", groundtruth1.mean.cpu().numpy())
    print("Gaussian 2:", groundtruth2.mean.cpu().numpy())
    print("Estimated Means:")
    print("Model 1:", model1.mean.cpu().numpy())
    print("Model 2:", model2.mean.cpu().numpy())

    print("\nTrue Eigenvalues:")
    print("Gaussian 1:", groundtruth1.eigenvalues.cpu().numpy())
    print("Gaussian 2:", groundtruth2.eigenvalues.cpu().numpy())
    print("Estimated Eigenvalues:")
    print("Model 1:", model1.eigenvalues.cpu().numpy())
    print("Model 2:", model2.eigenvalues.cpu().numpy())

    print("\nTrue Eigenvectors:")
    print("Gaussian 1:", groundtruth1.eigenvectors.cpu().numpy())
    print("Gaussian 2:", groundtruth2.eigenvectors.cpu().numpy())
    print("Estimated Eigenvectors:")
    print("Model 1:", model1.eigenvectors.cpu().numpy())
    print("Model 2:", model2.eigenvectors.cpu().numpy())

    # ======== Compute Component Assignments ========
    # Assign each sample to the component with the highest responsibility
    component_assignments = torch.argmax(torch.cat([responsibilities1, responsibilities2], dim=1), dim=1)

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
                color='black', marker='x', s=200, label='Estimated Mean 1', edgecolors='black', linewidths=2)
    plt.scatter(model2.mean[0].cpu().numpy(), model2.mean[1].cpu().numpy(), 
                color='yellow', marker='x', s=200, label='Estimated Mean 2', edgecolors='black', linewidths=2)

    # True means (ground truth)
    plt.scatter(groundtruth1.mean[0].cpu().numpy(), groundtruth1.mean[1].cpu().numpy(), 
                color='green', marker='+', s=200, label='True Mean 1', edgecolors='black', linewidths=2)
    plt.scatter(groundtruth2.mean[0].cpu().numpy(), groundtruth2.mean[1].cpu().numpy(), 
                color='red', marker='+', s=200, label='True Mean 2', edgecolors='black', linewidths=2)

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
        cov1_true = eigen_to_cov(groundtruth1.eigenvectors.cpu().numpy(), groundtruth1.eigenvalues.cpu().numpy())
        cov2_true = eigen_to_cov(groundtruth2.eigenvectors.cpu().numpy(), groundtruth2.eigenvalues.cpu().numpy())

        # Estimated covariance matrices
        cov1_est = eigen_to_cov(model1.eigenvectors.cpu().numpy(), model1.eigenvalues.cpu().numpy())
        cov2_est = eigen_to_cov(model2.eigenvectors.cpu().numpy(), model2.eigenvalues.cpu().numpy())

        # Plot true covariance ellipses
        plot_cov_ellipse(groundtruth1.mean.cpu().numpy(), cov1_true, alpha=0.3, color='green', label='True Cov 1')
        plot_cov_ellipse(groundtruth2.mean.cpu().numpy(), cov2_true, alpha=0.3, color='red', label='True Cov 2')

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