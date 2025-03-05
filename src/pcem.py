import torch
import torch.nn as nn
from typing import Optional, Tuple

class ComponentGaussian(nn.Module):
    def __init__(self, dimensionality=None, component_count=None, mean=None, eigenvectors=None, eigenvalues=None, requires_grad=False, device="cpu"):
        """
        ComponentGaussian represents a single Gaussian distribution parameterized by
        its mean, eigenvectors, and eigenvalues (variances along principal directions).
        
        Can be initialized either with:
        1. dimensionality (and optionally component_count) for random initialization
        2. explicit mean, eigenvectors, and eigenvalues
        
        :param dimensionality: Integer 'd' representing the dimensionality of the space.
        :param component_count: Integer 'k' for the number of principal components to use.
        :param mean: Tensor of shape (d,) representing the mean vector.
        :param eigenvectors: Tensor of shape (d, k) where each column is a principal component.
        :param eigenvalues: Tensor of shape (k,) representing variances along principal components.
        :param device: Device to run computations on ("cpu" or "cuda").
        """
        super(ComponentGaussian, self).__init__()
        
        # Case 1: Random initialization based on dimensionality
        if dimensionality is not None and mean is None:
            if component_count is None:
                component_count = dimensionality
            elif component_count > dimensionality:
                raise ValueError(f"Component count ({component_count}) cannot exceed dimensionality ({dimensionality})")
            
            # Generate random parameters
            mean = torch.randn(dimensionality, device=device)
            
            # Random orthonormal eigenvectors in column reprsentation
            random_matrix = torch.randn(dimensionality, dimensionality, device=device)
            eigenvectors_full, _ = torch.linalg.qr(random_matrix)
            eigenvectors = eigenvectors_full[:, :component_count]
            
            # Random positive eigenvalues
            eigenvalues = torch.abs(torch.randn(component_count, device=device))
            
        # Case 2: Explicit initialization with given parameters
        elif mean is not None and eigenvectors is not None and eigenvalues is not None:
            # Parameters will be converted to nn.Parameter below
            pass
        else:
            raise ValueError("Either provide dimensionality for random initialization or all three parameters: mean, eigenvectors, eigenvalues")
        
        # Create parameters directly
        self.mean = nn.Parameter(mean)
        self.eigenvectors = nn.Parameter(eigenvectors)
        self.eigenvalues = nn.Parameter(eigenvalues)
        self.use_grad(requires_grad)

    def fit(self, data, responsibilities):
        """
        Modified M-step: Extract principal components iteratively.
        Updates this ComponentGaussian instance with parameters estimated from data.
        
        :param data: Tensor of shape (n_samples, dimensionality)
        :param responsibilities: Tensor of shape (n_samples, 1) of weights for each sample
        :return: self (for method chaining)
        """
        device = self.mean.device
        # print("\n[PCEM M-Step] Computing weighted mean...")
        
        # Compute weighted mean
        weights = responsibilities.sum(dim=0)
        mu = (responsibilities.T @ data) / weights.unsqueeze(1)

        # print(f"  Estimated Mean: {mu.cpu().numpy()}")

        # Center data
        centered_data = data - mu

        # Compute the scatter matrix (without explicit covariance computation)
        # this is a weighted, unnormalized covariance matrix (i.e., not divided by the total number of samples)
        scatter_matrix = (centered_data.T @ (responsibilities * centered_data)) / weights

        # Extract principal components iteratively
        eigenvectors = []
        eigenvalues = []

        # Get dimensionality and component count from current instance
        d = self.mean.shape[0]
        k = self.eigenvalues.shape[0]

        for comp in range(k):  # Extract k components 
            # print(f"\n[PCEM M-Step] Extracting Principal Component {comp+1}...")
            
            v, eigenvalue = self._power_iteration(scatter_matrix)
            eigenvectors.append(v)
            eigenvalues.append(eigenvalue)

            # old method for subtracting variance -- test for speed and stability
            # projection = (centered_data @ v).unsqueeze(1) * v.unsqueeze(0)
            # centered_data -= projection  # Deflate the data

            # Deflate the data along the discovered eigenvector using projection
            # projection_matrix = torch.eye(v.shape[0], device=device) - torch.outer(v, v)
            # centered_data = centered_data @ projection_matrix  # Project onto hyperplane
            # scatter_matrix = (centered_data.T @ (responsibilities * centered_data)) / weights

            # # Deflate the data along the discovered eigenvector using Gram-Schmidt 
            # centered_data -= (centered_data @ v.unsqueeze(1)) * v.unsqueeze(0)
            # scatter_matrix = (centered_data.T @ (responsibilities * centered_data)) / weights

            # Subtract the discovered component from the scatter_matrix
            scatter_matrix -= eigenvalue * torch.outer(v, v)

            # print(eigenvectors)
            # print(f"  Variance removed: {eigenvalue:.6f}")
            # print(f"  Remaining scatter matrix:\n{scatter_matrix.cpu().numpy()}\n")

        # Update the model parameters
        self.set_mean(mu.squeeze(0))
        stacked_eigenvectors = torch.stack(eigenvectors, dim=1) # column storage
        self.set_eigenvectors(stacked_eigenvectors)
        self.set_eigenvalues(torch.tensor(eigenvalues, device=data.device))
        
        return self

    def _power_iteration(self, scatter_matrix, num_iters=50, tol=1e-6):
        """
        Power Iteration to find the dominant eigenvector of matrix scatter_matrix.
        """
        device = self.mean.device

        # random start
        v = torch.randn(scatter_matrix.shape[0], device=device)
        v /= torch.norm(v)

        # initialize v as the mean direction
        # v = torch.mean(centered_data, dim=0)
        # v /= torch.norm(v)

        # print("\n[Power Iteration] Starting...")
        
        for i in range(num_iters):
            v_new = scatter_matrix @ v
            v_new /= torch.norm(v_new)

            diff = torch.norm(v_new - v)
            # print(f"  Iter {i+1}: Change in eigenvector = {diff:.6f}")

            if diff < tol:
                # print("  Converged.")
                break

            v = v_new

        eigenvalue = (v @ scatter_matrix @ v).item()
        # print(f"  Dominant eigenvalue found: {eigenvalue:.6f}")
        # print(f"  Dominant eigenvector: {v.cpu().numpy()}\n")

        return v, eigenvalue

    def _rayleigh_quotient_iteration(self, scatter_matrix, num_iters=50, tol=1e-6):
        """
        Rayleigh Quotient Iteration to find the dominant eigenvector and eigenvalue.
        """
        device = self.mean.device
        v = torch.randn(scatter_matrix.shape[0], device=device)
        v /= torch.norm(v)

        # print("\n[Rayleigh Quotient Iteration] Starting...")

        for i in range(num_iters):
            v_new = scatter_matrix @ v
            v_new /= torch.norm(v_new)  # Normalization (same as power iteration)

            eigenvalue = (v_new @ scatter_matrix @ v_new).item()  # Rayleigh Quotient
            
            diff = torch.norm(v_new - v)
            # print(f"  Iter {i+1}: Eigenvalue = {eigenvalue:.6f}, Change in eigenvector = {diff:.6f}")

            if diff < tol:
                # print("  Converged.")
                break

            v = v_new

        # print(f"  Dominant eigenvalue found: {eigenvalue:.6f}")
        # print(f"  Dominant eigenvector: {v.cpu().numpy()}\n")

        return v, eigenvalue

    def use_grad(self, requires_grad=True):
        """
        Enable or disable gradient tracking for all parameters in this module.
        
        :param requires_grad: Boolean indicating whether gradients should be tracked
        :return: self (for method chaining)
        """
        for param in self.parameters():
            param.requires_grad_(requires_grad)
        return self

    def set_mean(self, mean_tensor):
        """Set new values for the mean parameter"""
        self.mean.data.copy_(mean_tensor)
        return self
        
    def set_eigenvectors(self, eigenvectors_tensor):
        """Set new values for the eigenvectors parameter"""
        self.eigenvectors.data.copy_(eigenvectors_tensor)
        return self
        
    def set_eigenvalues(self, eigenvalues_tensor):
        """Set new values for the eigenvalues parameter"""
        self.eigenvalues.data.copy_(eigenvalues_tensor)
        return self

    def forward(self, num_samples):
        """
        Generate samples from the Gaussian distribution.
        This is equivalent to the sample method but follows PyTorch convention.
        
        :param num_samples: Number of samples to generate.
        :return: Tensor of shape (num_samples, d) containing generated samples.
        """
        raise "NotImplemented" 
    
    def sample(self, num_samples):
        """
        Generate samples from the Gaussian distribution using its principal components.
        
        :param num_samples: Number of samples to generate.
        :return: Tensor of shape (num_samples, d) containing generated samples.
        """
        d = self.mean.shape[0]  # Dimensionality
        
        # Generate standard normal samples
        z = torch.randn(num_samples, d, device=self.mean.device)
        
        # Scale by sqrt of eigenvalues (standard deviations along principal axes)
        scaled_samples = z * torch.sqrt(self.eigenvalues).unsqueeze(0)
        
        # Transform into original space using eigenvectors
        transformed_samples = scaled_samples @ self.eigenvectors.T
        
        # Shift by mean
        samples = transformed_samples + self.mean
        
        return samples

    def calculate_squared_mahalanobis_distance(self, data):
        """
        Calculate the squared Mahalanobis distance for a batch of samples based on this Gaussian component.

        :param data: A batch of sample to calculate the Mahalanobis distance for (1D tensor of size [dimensionality])
        :return: Mahalanobis distance for the sample
        """
        # Step 1: Center the samples by subtracting the mean
        centered_data = data - self.mean

        # Step 2: Project the centered data onto the eigenvector space (principal component space)
        projected_data = torch.matmul(centered_data, self.eigenvectors)  # (1D sample x eigenvectors matrix)

        # Step 3: Compute the Mahalanobis distance squared (using the eigenvalues for scaling)
        mahalanobis_sq = torch.sum((projected_data ** 2) / (self.eigenvalues.unsqueeze(0) + 1e-6))  # Adding epsilon for stability
        
        # Return the squared Mahalanobis distance
        return mahalanobis_sq
    
    def calculate_mahalanobis_distance(self, data):
        """
        Calculate the Mahalanobis distance for a batch of samples based on this Gaussian component.

        :param data: A batch of sample to calculate the Mahalanobis distance for (1D tensor of size [dimensionality])
        :return: Mahalanobis distance for the sample
        """
        # Return the Mahalanobis distance (not squared)
        return torch.sqrt(self.calculate_squared_mahalanobis_distance(data))
    
    def calculate_likelihood(self, data):
        """
        Calculate the unnormalized responsibilities (likelihoods) for each data point
        belonging to this Gaussian component.

        :param data: Tensor of shape (n_samples, dimensionality)
        :return: Tensor of shape (n_samples, 1) containing unnormalized responsibilities
        """
        d = self.mean.shape[0]
        device = self.mean.device
        
        # Compute Mahalanobis distance squared
        mahalanobis_sq = self.calculate_squared_mahalanobis_distance(data)

        # Compute log determinant of covariance (with stability term)
        log_det = torch.log(self.eigenvalues + 1e-6).sum()

        # Compute log likelihood
        log_likelihood = -0.5 * (d * torch.log(2 * torch.tensor(torch.pi, device=device)) 
                                + log_det 
                                + mahalanobis_sq)

        # Convert to likelihoods (unnormalized responsibilities)
        likelihoods = torch.exp(log_likelihood).unsqueeze(1)

        return likelihoods  # Should be normalized externally in GMM

class ComponentGMM(nn.Module):
    """
    Gaussian Mixture Model implementation using Principal Component EM (PCEM)
    with ComponentGaussian objects.
    """
    
    def __init__(self, n_gaussians: int, n_dimensions: int, 
                 max_iterations: int = 100, 
                 tolerance: float = 1e-4,
                 random_state: Optional[int] = None,
                 requires_grad=False,
                 device: Optional[torch.device] = None):
        """
        Initialize the ComponentGMM model.
        
        Args:
            n_gaussians: Number of Gaussian components
            n_dimensions: Dimensionality of the data
            max_iterations: Maximum number of EM iterations
            tolerance: Convergence tolerance for log-likelihood
            random_state: Random seed for reproducibility
            device: Device to run computations on (CPU/GPU)
        """
        super(ComponentGMM, self).__init__()

        self.n_gaussians = n_gaussians
        self.n_dimensions = n_dimensions
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # Set random seed if provided
        if random_state is not None:
            torch.manual_seed(random_state)
            
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components and mixing coefficients
        self.gaussians = nn.ModuleList([ComponentGaussian(n_dimensions, device=self.device) for _ in range(n_gaussians)])
        self.mixing_coeffs = nn.Parameter(torch.ones(n_gaussians, device=self.device) / n_gaussians)
        # print(f"self.mixing_coeffs {self.mixing_coeffs.size()}")
        
        # For convergence tracking
        self.log_likelihood_history = []
        self.n_iterations_ = 0
        self.converged_ = False
        self.use_grad(requires_grad)

    def use_grad(self, requires_grad=True):
        """
        Enable or disable gradient tracking for all parameters in this module.
        
        :param requires_grad: Boolean indicating whether gradients should be tracked
        :return: self (for method chaining)
        """
        for param in self.parameters():
            param.requires_grad_(requires_grad)
        return self

    def forward(self, x: torch.Tensor):
        """
        Override forward method. For now, this could be a dummy method, as PCEM-GMM works through fit and e_step.
        """
        raise "NotImplemented"

    def to(self, device: torch.device):
        """
        Override the to() method inherited from nn.Module to move the entire model to the specified device.
        """
        super(ComponentGMM, self).to(device)  # Call the parent class' to method
        self.device = device
        return self

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
            responsibilities: Tensor of shape [n_samples, n_gaussians]
            log_likelihood: Current log-likelihood value
        """
        n_samples = x.shape[0]
        
        # Calculate likelihoods for each component (will be of shape [n_samples, 1])
        component_likelihoods = []
        weighted_likelihoods = []
        
        for i, gaussian in enumerate(self.gaussians):
            # Get likelihoods from gaussian - keep the (n_samples, 1) shape
            likelihood = gaussian.calculate_likelihood(x)  # or the original method name
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
        Perform the M-step: update gaussian parameters.
        
        Args:
            x: Data tensor of shape [n_samples, n_dimensions]
            responsibilities: Tensor of shape [n_samples, n_gaussians]
        """
        # Update parameters for each gaussian
        for i, gaussian in enumerate(self.gaussians):
            # Get responsibilities for this gaussian
            resp_i = responsibilities[:, i].unsqueeze(1)
            
            # Update gaussian parameters
            gaussian.fit(x, resp_i)
            
            # Update mixing coefficient
            self.mixing_coeffs[i] = resp_i.mean()
    
    def sample(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate random samples from the mixture model while keeping track of true gaussian labels.

        Args:
            n_samples: Number of samples to generate

        Returns:
            samples: Tensor of shape [n_samples, n_dimensions] containing generated samples.
            labels: Tensor of shape [n_samples] with true gaussian labels.
        """
        # Determine how many samples to generate from each gaussian
        component_samples = torch.multinomial(
            self.mixing_coeffs, 
            n_samples, 
            replacement=True
        ).bincount(minlength=self.n_gaussians)

        samples_list = []
        labels_list = []

        # Generate samples from each gaussian and record labels
        for i, n in enumerate(component_samples):
            if n > 0:
                component_samples_i = self.gaussians[i].sample(int(n))
                samples_list.append(component_samples_i)
                labels_list.append(torch.full((int(n),), i, dtype=torch.long, device=self.device))  

        # Combine all samples and labels
        if samples_list:
            samples = torch.cat(samples_list, dim=0)
            labels = torch.cat(labels_list, dim=0)  # Fully torch-based

            # Shuffle the samples and labels together
            perm = torch.randperm(samples.shape[0], device=self.device)
            return samples[perm], labels[perm]
        else:
            return torch.zeros((0, self.n_dimensions), device=self.device), torch.zeros((0,), dtype=torch.long, device=self.device)

    def get_cluster_assignments(self, data):
            """
            Assign each data point to the cluster with the highest responsibility.
            This function assumes that the model's parameters are already fitted.
            
            :param data: Tensor of shape [n_samples, n_dimensions] representing the data
            :return: Tensor of shape [n_samples] containing the index of the most likely cluster for each data point
            """
            # Compute the responsibilities (E-step)
            responsibilities, _ = self.e_step(data)
            
            # Return the index of the cluster with the highest responsibility for each data point
            return torch.argmax(responsibilities, dim=1)

    def calculate_log_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the log-likelihood of the data under the current model.
        
        Args:
            x: Data tensor of shape [n_samples, n_dimensions]
            
        Returns:
            Log-likelihood value
        """
        n_samples = x.shape[0]
        
        # Calculate likelihoods for each gaussian
        weighted_likelihoods = torch.zeros((n_samples, self.n_gaussians), device=self.device)
        
        for i, gaussian in enumerate(self.gaussians):
            likelihoods = gaussian.calculate_likelihood(x)
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
        print(f"Number of components: {self.n_gaussians}")
        print(f"Dimensions: {self.n_dimensions}")
        print(f"Iterations run: {self.n_iterations_}")
        print(f"Converged: {self.converged_}")
        
        if len(self.log_likelihood_history) > 0:
            print(f"Final log-likelihood: {self.log_likelihood_history[-1]:.4f}")
        
        print("\nMixing coefficients:")
        for i, coef in enumerate(self.mixing_coeffs):
            print(f"  Component {i+1}: {coef.item():.4f}")
        
        print("\Gaussian means:")
        for i, gaussian in enumerate(self.gaussians):
            print(f"  Gaussian {i+1}: {gaussian.mean.cpu().numpy()}")
        
        print("\Gaussian eigenvalues:")
        for i, gaussian in enumerate(self.gaussians):
            print(f"  Gaussian {i+1}: {gaussian.eigenvalues.cpu().numpy()}")

