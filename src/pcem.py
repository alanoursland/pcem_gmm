import torch
import torch.nn as nn

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
            
            # Random orthonormal eigenvectors
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

    def _power_iteration(self, S, num_iters=50, tol=1e-6):
        """
        Power Iteration to find the dominant eigenvector of matrix S.
        """
        device = self.mean.device
        v = torch.randn(S.shape[0], device=device)
        v /= torch.norm(v)

        print("\n[Power Iteration] Starting...")
        
        for i in range(num_iters):
            v_new = S @ v
            v_new /= torch.norm(v_new)

            diff = torch.norm(v_new - v)
            print(f"  Iter {i+1}: Change in eigenvector = {diff:.6f}")

            if diff < tol:
                print("  Converged.")
                break

            v = v_new

        eigenvalue = (v @ S @ v).item()
        print(f"  Dominant eigenvalue found: {eigenvalue:.6f}")
        print(f"  Dominant eigenvector: {v.cpu().numpy()}\n")

        return v, eigenvalue

    def calculate_responsibilities(self, data):
        """
        Calculate the unnormalized responsibilities (likelihoods) for each data point
        belonging to this Gaussian component.

        :param data: Tensor of shape (n_samples, dimensionality)
        :return: Tensor of shape (n_samples, 1) containing unnormalized responsibilities
        """
        d = self.mean.shape[0]
        device = self.mean.device
        
        # Center the data
        centered_data = data - self.mean

        # Project data onto eigenvector space
        projected_data = centered_data @ self.eigenvectors

        # Compute Mahalanobis distance squared
        scaled_distances = (projected_data ** 2) / (self.eigenvalues.unsqueeze(0) + 1e-6)
        mahalanobis_sq = scaled_distances.sum(dim=1)

        # Compute log determinant of covariance (with stability term)
        log_det = torch.log(self.eigenvalues + 1e-6).sum()

        # Compute log likelihood
        log_likelihood = -0.5 * (d * torch.log(2 * torch.tensor(torch.pi, device=device)) 
                                + log_det 
                                + mahalanobis_sq)

        # Convert to likelihoods (unnormalized responsibilities)
        likelihoods = torch.exp(log_likelihood).unsqueeze(1)

        return likelihoods  # Should be normalized externally in GMM

    def fit(self, data, responsibilities):
        """
        Modified M-step: Extract principal components iteratively.
        Updates this ComponentGaussian instance with parameters estimated from data.
        
        :param data: Tensor of shape (n_samples, dimensionality)
        :param responsibilities: Tensor of shape (n_samples, 1) of weights for each sample
        :return: self (for method chaining)
        """
        device = self.mean.device
        print("\n[PCEM M-Step] Computing weighted mean...")
        
        # Compute weighted mean
        weights = responsibilities.sum(dim=0)
        mu = (responsibilities.T @ data) / weights.unsqueeze(1)

        print(f"  Estimated Mean: {mu.cpu().numpy()}")

        # Center data
        centered_data = data - mu

        # Compute the scatter matrix (without explicit covariance computation)
        S = (centered_data.T @ (responsibilities * centered_data)) / weights

        # Extract principal components iteratively
        eigenvectors = []
        eigenvalues = []

        # Get dimensionality and component count from current instance
        d = self.mean.shape[0]
        k = self.eigenvalues.shape[0]

        for comp in range(k):  # Extract k components 
            print(f"\n[PCEM M-Step] Extracting Principal Component {comp+1}...")
            
            v, eigenvalue = self._power_iteration(S)
            eigenvectors.append(v)
            eigenvalues.append(eigenvalue)

            # Deflate the data along the discovered eigenvector
            projection_matrix = torch.eye(v.shape[0], device=device) - torch.outer(v, v)
            centered_data = centered_data @ projection_matrix  # Project onto hyperplane

            # old method for subtracting variance -- test for speed and stability
            # projection = (centered_data @ v).unsqueeze(1) * v.unsqueeze(0)
            # centered_data -= projection  # Deflate the data

            # Recompute S after deflation
            S = (centered_data.T @ (responsibilities * centered_data)) / weights

            print(f"  Variance removed: {eigenvalue:.6f}")
            print(f"  Remaining scatter matrix:\n{S.cpu().numpy()}\n")

        # Update the model parameters
        self.set_mean(mu.squeeze(0))
        self.set_eigenvectors(torch.stack(eigenvectors, dim=1))
        self.set_eigenvalues(torch.tensor(eigenvalues, device=data.device))
        
        return self

