import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from matplotlib.patches import Ellipse
from pcem import ComponentGaussian
from scipy.linalg import eigh

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Union

from pcem import ComponentGaussian

from typing import Optional, Tuple
import torch
import numpy as np
from pcem import ComponentGaussian

class ComponentGMM:
    """
    Gaussian Mixture Model implementation using Principal Component EM (PCEM)
    with ComponentGaussian objects.
    """
    
    def __init__(self, n_components: int, n_dimensions: int, 
                 max_iterations: int = 100, 
                 tolerance: float = 1e-4,
                 random_state: Optional[int] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize the ComponentGMM model.
        
        Args:
            n_components: Number of Gaussian components
            n_dimensions: Dimensionality of the data
            max_iterations: Maximum number of EM iterations
            tolerance: Convergence tolerance for log-likelihood
            random_state: Random seed for reproducibility
            device: Device to run computations on (CPU/GPU)
        """
        self.n_components = n_components
        self.n_dimensions = n_dimensions
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # Set random seed if provided
        if random_state is not None:
            torch.manual_seed(random_state)
            
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components and mixing coefficients
        self.components = [ComponentGaussian(n_dimensions, device=self.device) for _ in range(n_components)]
        self.mixing_coeffs = torch.ones(n_components, device=self.device) / n_components
        
        # For convergence tracking
        self.log_likelihood_history = []
        self.n_iterations_ = 0
        self.converged_ = False
        
    def fit(self, x: torch.Tensor) -> 'ComponentGMM':
        """
        Fit the GMM to the data using the EM algorithm.
        
        Args:
            x: Data tensor of shape [n_samples, n_dimensions]
            
        Returns:
            self: The fitted model
        """
        # Initial log likelihood
        prev_log_likelihood = float('-inf')
        self.n_iterations_ = 0
        
        # EM loop
        for iteration in range(self.max_iterations):
            # E-step: compute responsibilities
            responsibilities, current_log_likelihood = self.e_step(x)
            
            # Store log likelihood
            self.log_likelihood_history.append(current_log_likelihood.item())
            
            # Check for convergence
            if self._check_convergence(current_log_likelihood, prev_log_likelihood):
                self.converged_ = True
                break
                
            # M-step: update parameters
            self.m_step(x, responsibilities)
            
            # Update previous log likelihood
            prev_log_likelihood = current_log_likelihood
            
            # Update iteration count
            self.n_iterations_ = iteration + 1
            
        return self
        
    def e_step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform the E-step: compute responsibilities.
        
        Args:
            x: Data tensor of shape [n_samples, n_dimensions]
            
        Returns:
            responsibilities: Tensor of shape [n_samples, n_components]
            log_likelihood: Current log-likelihood value
        """
        n_samples = x.shape[0]
        
        # Calculate likelihoods for each component (will be of shape [n_samples, 1])
        component_likelihoods = []
        weighted_likelihoods = []
        
        for i, component in enumerate(self.components):
            # Get likelihoods from component - keep the (n_samples, 1) shape
            likelihood = component.calculate_density(x)  # or the original method name
            component_likelihoods.append(likelihood)
            
            # Apply mixing coefficient (still keeping the shape)
            weighted_likelihood = likelihood * self.mixing_coeffs[i]
            weighted_likelihoods.append(weighted_likelihood)
        
        # Stack weighted likelihoods for easier operations
        all_weighted_likelihoods = torch.cat(weighted_likelihoods, dim=1)
        
        # Compute total likelihood per sample (sum across components)
        total_likelihoods = all_weighted_likelihoods.sum(dim=1, keepdim=True)
        
        # Compute responsibilities for all components
        responsibilities = torch.cat([
            weighted_likelihood / (total_likelihoods + 1e-6) 
            for weighted_likelihood in weighted_likelihoods
        ], dim=1)
        
        # Compute log likelihood
        log_likelihood = torch.log(torch.clamp(total_likelihoods, min=1e-10)).sum()
        
        return responsibilities, log_likelihood

    def m_step(self, x: torch.Tensor, responsibilities: torch.Tensor) -> None:
        """
        Perform the M-step: update component parameters.
        
        Args:
            x: Data tensor of shape [n_samples, n_dimensions]
            responsibilities: Tensor of shape [n_samples, n_components]
        """
        # Update parameters for each component
        for i, component in enumerate(self.components):
            # Get responsibilities for this component
            resp_i = responsibilities[:, i].unsqueeze(1)
            
            # Update component parameters
            component.fit(x, resp_i)
            
            # Update mixing coefficient
            self.mixing_coeffs[i] = resp_i.mean()
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Generate random samples from the mixture model.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Samples of shape [n_samples, n_dimensions]
        """
        # Determine how many samples to generate from each component
        component_samples = torch.multinomial(
            self.mixing_coeffs, 
            n_samples, 
            replacement=True
        ).bincount(minlength=self.n_components)
        
        samples_list = []
        
        # Generate samples from each component
        for i, n in enumerate(component_samples):
            if n > 0:
                component_samples = self.components[i].sample(int(n))
                samples_list.append(component_samples)
        
        # Combine all samples
        if samples_list:
            samples = torch.cat(samples_list, dim=0)
            
            # Shuffle the samples
            perm = torch.randperm(samples.shape[0])
            return samples[perm]
        else:
            return torch.zeros((0, self.n_dimensions), device=self.device)
    
    def calculate_log_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the log-likelihood of the data under the current model.
        
        Args:
            x: Data tensor of shape [n_samples, n_dimensions]
            
        Returns:
            Log-likelihood value
        """
        n_samples = x.shape[0]
        
        # Calculate likelihoods for each component
        weighted_likelihoods = torch.zeros((n_samples, self.n_components), device=self.device)
        
        for i, component in enumerate(self.components):
            likelihoods = component.calculate_density(x)
            weighted_likelihoods[:, i] = likelihoods.squeeze() * self.mixing_coeffs[i]
        
        # Sum likelihoods across components
        total_likelihoods = weighted_likelihoods.sum(dim=1)
        
        # Compute log likelihood with stabilization
        log_likelihood = torch.log(torch.clamp(total_likelihoods, min=1e-10)).sum()
        
        return log_likelihood
    
    def _check_convergence(self, current_ll: float, previous_ll: float) -> bool:
        """
        Check if the algorithm has converged.
        
        Args:
            current_ll: Current log-likelihood
            previous_ll: Previous log-likelihood
            
        Returns:
            True if converged, False otherwise
        """
        if previous_ll == float('-inf'):
            return False
        
        # Calculate absolute and relative change
        abs_change = current_ll - previous_ll
        rel_change = abs(abs_change / (previous_ll + 1e-10))
        
        # Check if relative change is below tolerance
        return rel_change < self.tolerance
    
    def summary(self) -> None:
        """
        Print a summary of the model parameters.
        """
        print(f"\n=== ComponentGMM Summary ===")
        print(f"Number of components: {self.n_components}")
        print(f"Dimensions: {self.n_dimensions}")
        print(f"Iterations run: {self.n_iterations_}")
        print(f"Converged: {self.converged_}")
        
        if len(self.log_likelihood_history) > 0:
            print(f"Final log-likelihood: {self.log_likelihood_history[-1]:.4f}")
        
        print("\nMixing coefficients:")
        for i, coef in enumerate(self.mixing_coeffs):
            print(f"  Component {i+1}: {coef.item():.4f}")
        
        print("\nComponent means:")
        for i, component in enumerate(self.components):
            print(f"  Component {i+1}: {component.mean.cpu().numpy()}")
        
        print("\nComponent eigenvalues:")
        for i, component in enumerate(self.components):
            print(f"  Component {i+1}: {component.eigenvalues.cpu().numpy()}")

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d = 2  # Dimensionality
    num_samples = 1000

    # Create a ground truth GMM with 2 components in 2D
    groundtruth = ComponentGMM(n_components=2, n_dimensions=2, device=device)

    # Configure the first component (elongated shape)
    groundtruth.components[0].set_mean(torch.tensor([-5.0, 2.0], device=device))
    groundtruth.components[0].set_eigenvalues(torch.tensor([6.0, 1.0], device=device))

    # Configure the second component (spherical shape)
    groundtruth.components[1].set_mean(torch.tensor([5.0, -3.0], device=device))
    groundtruth.components[1].set_eigenvalues(torch.tensor([2.0, 2.0], device=device))

    # Set mixing coefficients to 0.5 each (equal proportions)
    groundtruth.mixing_coeffs = torch.tensor([0.5, 0.5], device=device)

    # Generate samples from the mixture model
    samples = groundtruth.sample(num_samples)

    # For visualization, we need to know which component generated each sample
    # This isn't directly available from sample(), so we can compute it afterwards
    responsibilities, _ = groundtruth.e_step(samples)
    sample_source = torch.argmax(responsibilities, dim=1)

    print("\n[Experiment] Running PCEM-GMM on synthetic 2D Gaussian mixture data...\n")

    # Initialize ComponentGMM with 2 components in 2D
    model = ComponentGMM(
        n_components=2, 
        n_dimensions=2, 
        max_iterations=20, 
        tolerance=1e-4, 
        device=device
    )

    # EM algorithm
    max_iterations = 20
    stop_tolerance = 1e-4
    # prev_log_likelihood = float('-inf')

    # for iteration in range(max_iterations):
    #     print(f"\n[EM Iteration {iteration+1}]")

    #     # ======== E-Step: Compute responsibilities ========
    #     # likelihoods1 = model1.calculate_density(samples)  # (n_samples, 1)
    #     # likelihoods2 = model2.calculate_density(samples)  # (n_samples, 1)

    #     # # Apply mixture weights
    #     # weighted_likelihoods1 = likelihoods1 * mixing_coeffs[0]
    #     # weighted_likelihoods2 = likelihoods2 * mixing_coeffs[1]

    #     # # Compute total likelihood per sample (avoid zero division)
    #     # total_likelihoods = weighted_likelihoods1 + weighted_likelihoods2  # (n_samples, 1)

    #     # # Compute responsibilities (normalize across components)
    #     # responsibilities1 = weighted_likelihoods1 / (total_likelihoods + 1e-6)
    #     # responsibilities2 = weighted_likelihoods2 / (total_likelihoods + 1e-6)
    #     # print(responsibilities1.size())
    #     # print(responsibilities2.size())

    #     responsibilities, current_log_likelihood = model.e_step(samples)
    #     # responsibilities1 = responsibilities[:,0].unsqueeze(1)
    #     # responsibilities2 = responsibilities[:,1].unsqueeze(1)

    #     # ======== M-step: Update component parameters ========
    #     # model1.fit(samples, responsibilities1)
    #     # mixing_coeffs[0] = responsibilities1.mean()

    #     # model2.fit(samples, responsibilities2)
    #     # mixing_coeffs[1] = responsibilities2.mean()

    #     # Compute log likelihood for convergence check (ensure stability)
    #     # current_log_likelihood = torch.log(torch.clamp(total_likelihoods, min=1e-6)).sum()
    #     log_likelihood_change = current_log_likelihood - prev_log_likelihood
    #     rel_change = abs(log_likelihood_change / (prev_log_likelihood + 1e-10))

    #     # Check for convergence
    #     if rel_change < stop_tolerance:
    #         print(f"\nConverged after {iteration+1} iterations!")
    #         break

    #     # Update mixing coefficients
    #     model.m_step(samples, responsibilities)

    #     print(f"  Log Likelihood: {current_log_likelihood.item():.4f}")
    #     print(f"  Change in Log Likelihood: {log_likelihood_change.item()}")
    #     print(f"  Relative change in Log Likelihood: {rel_change.item()}")

    #     prev_log_likelihood = current_log_likelihood

    # Fit the model to the data
    model.fit(samples)

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

    # Step 3: Compare Recovered vs. Ground Truth
    print("\n=== Final Comparison ===")
    print("True Means:")
    print("Gaussian 1:", groundtruth.components[0].mean.cpu().numpy())
    print("Gaussian 2:", groundtruth.components[1].mean.cpu().numpy())
    print("Estimated Means:")
    print("Model 1:", model.components[0].mean.cpu().numpy())
    print("Model 2:", model.components[1].mean.cpu().numpy())

    print("\nTrue Eigenvalues:")
    print("Gaussian 1:", groundtruth.components[0].eigenvalues.cpu().numpy())
    print("Gaussian 2:", groundtruth.components[1].eigenvalues.cpu().numpy())
    print("Estimated Eigenvalues:")
    print("Model 1:", model.components[0].eigenvalues.cpu().numpy())
    print("Model 2:", model.components[1].eigenvalues.cpu().numpy())

    print("\nTrue Eigenvectors (row based):")
    print("Gaussian 1:", groundtruth.components[0].eigenvectors.T.cpu().numpy())
    print("Gaussian 2:", groundtruth.components[1].eigenvectors.T.cpu().numpy())
    print("Estimated Eigenvectors (row based):")
    print("Model 1:", model.components[0].eigenvectors.T.cpu().numpy())
    print("Model 2:", model.components[1].eigenvectors.T.cpu().numpy())

    # ======== Compute Component Assignments ========
    # Assign each sample to the component with the highest responsibility
    responsibilities, _ = groundtruth.e_step(samples)
    component_assignments = torch.argmax(responsibilities, dim=1)
    model1 = model.components[0]
    model2 = model.components[1]

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
    plt.scatter(groundtruth.components[0].mean[0].cpu().numpy(), groundtruth.components[0].mean[1].cpu().numpy(), 
                color='green', marker='+', s=200, label='True Mean 1', linewidths=2)
    plt.scatter(groundtruth.components[1].mean[0].cpu().numpy(), groundtruth.components[1].mean[1].cpu().numpy(), 
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
        cov1_true = eigen_to_cov(groundtruth.components[0].eigenvectors.cpu().numpy(), groundtruth.components[0].eigenvalues.cpu().numpy())
        cov2_true = eigen_to_cov(groundtruth.components[1].eigenvectors.cpu().numpy(), groundtruth.components[1].eigenvalues.cpu().numpy())

        # Estimated covariance matrices
        cov1_est = eigen_to_cov(model1.eigenvectors.cpu().numpy(), model1.eigenvalues.cpu().numpy())
        cov2_est = eigen_to_cov(model2.eigenvectors.cpu().numpy(), model2.eigenvalues.cpu().numpy())

        # Plot true covariance ellipses
        plot_cov_ellipse(groundtruth.components[0].mean.cpu().numpy(), cov1_true, alpha=0.3, color='green', label='True Cov 1')
        plot_cov_ellipse(groundtruth.components[1].mean.cpu().numpy(), cov2_true, alpha=0.3, color='red', label='True Cov 2')

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