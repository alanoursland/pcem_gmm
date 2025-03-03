import torch

# Ensure PyTorch runs on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Generate 2D Gaussian Data
torch.manual_seed(42)

# Generate a random unit vector
v1 = torch.randn(2, device=device)
v1 /= torch.norm(v1)

# Compute an orthogonal unit vector
v2 = torch.tensor([-v1[1], v1[0]], device=device)  # 90-degree rotation

# Assign eigenvalues (variances along each principal direction)
eigenvalues = torch.tensor([4.0, 1.0], device=device)  # Variances along v1, v2

# Select a mean
mean = torch.tensor([5.0, -3.0], device=device)

# Generate samples using the principal components directly
num_samples = 1000
z = torch.randn(num_samples, 2, device=device)  # Standard normal samples
samples = mean + z[:, 0].unsqueeze(1) * torch.sqrt(eigenvalues[0]) * v1 + z[:, 1].unsqueeze(1) * torch.sqrt(eigenvalues[1]) * v2

# Step 2: Implement PCEM-GMM for 2D Data

def power_iteration(S, num_iters=50, tol=1e-6):
    """
    Power Iteration to find the dominant eigenvector of matrix S.
    """
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

def pcem_m_step(data, responsibilities):
    """
    Modified M-step: Extract principal components iteratively.
    """
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

    for comp in range(2):  # Extract both components for 2D data
        print(f"\n[PCEM M-Step] Extracting Principal Component {comp+1}...")
        
        v, eigenvalue = power_iteration(S)
        eigenvectors.append(v)
        eigenvalues.append(eigenvalue)

        # Deflate the data instead of covariance matrix
        projection = (centered_data @ v).unsqueeze(1) * v.unsqueeze(0)
        centered_data -= projection  # Deflate the data

        # Recompute S after deflation
        S = (centered_data.T @ (responsibilities * centered_data)) / weights

        print(f"  Variance removed: {eigenvalue:.6f}")
        print(f"  Remaining scatter matrix:\n{S.cpu().numpy()}\n")

    return mu, torch.stack(eigenvectors), torch.tensor(eigenvalues, device=device)

# Step 3: Run PCEM-GMM on the synthetic data

print("\n[Experiment] Running PCEM-GMM on synthetic 2D Gaussian data...\n")

# Initial responsibilities (random assignment for now)
responsibilities = torch.ones(num_samples, 1, device=device)

# Estimate components using our modified EM
mu_est, eigenvectors_est, eigenvalues_est = pcem_m_step(samples, responsibilities)

# Step 4: Compare Recovered vs. Ground Truth
print("\n=== Final Comparison ===")
print("True Mean:", mean.cpu().numpy())
print("Estimated Mean:", mu_est.cpu().numpy())

print("\nTrue Eigenvalues:", eigenvalues.cpu().numpy())
print("Estimated Eigenvalues:", eigenvalues_est.cpu().numpy())

print("\nTrue Eigenvectors:\n", torch.stack([v1, v2]).cpu().numpy())
print("Estimated Eigenvectors:\n", eigenvectors_est.cpu().numpy())
