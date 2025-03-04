import torch
import torch.nn as nn
from pcem import ComponentGaussian
from metrics import compare_means, compare_eigenvalues, compare_eigenvectors

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Generate 2D Gaussian Data
    d = 2  # Dimensionality
    ground_truth = ComponentGaussian(2, device=device)
    ground_truth.set_mean(torch.tensor([5.0, -3.0], device=device))
    ground_truth.set_eigenvalues(torch.tensor([4.0, 1.0], device=device))
    # ComponentGaussian inits with random orthonormal eigenvectors

    # Generate samples
    num_samples = 1000
    samples = ground_truth.sample(num_samples)

    # Step 2: Implement PCEM-GMM for 2D Data
    # Step 3: Run PCEM-GMM on the synthetic data

    print("\n[Experiment] Running PCEM-GMM on synthetic 2D Gaussian data...\n")

    # Initial responsibilities (random assignment for now)
    model = ComponentGaussian(2, device=device)
    responsibilities = torch.ones(num_samples, 1, device=device)

    # Estimate components using our modified EM
    model.fit(samples, responsibilities)

    # Step 4: Compare Recovered vs. Ground Truth
    print("\n=== Final Comparison ===")
    print("True Mean:", ground_truth.mean)
    print("Estimated Mean:", model.mean)

    print("\nTrue Eigenvalues:", ground_truth.eigenvalues)
    print("Estimated Eigenvalues:", model.eigenvalues)

    print("\nTrue Eigenvectors:\n", ground_truth.eigenvectors)
    print("Estimated Eigenvectors:\n", model.eigenvectors)

    print()
    # Compare means
    mean_diff = compare_means(ground_truth.mean.cpu().numpy(), model.mean.cpu().numpy())
    print(f"Mean Difference: {mean_diff}")

    # Compare eigenvalues
    eigenvalue_diff = compare_eigenvalues(ground_truth.eigenvalues.cpu().numpy(), model.eigenvalues.cpu().numpy())
    print(f"Eigenvalue Difference: {eigenvalue_diff}")

    # Compare eigenvectors
    eigenvector_diff = compare_eigenvectors(ground_truth.eigenvectors.cpu().numpy(), model.eigenvectors.cpu().numpy())
    print(f"Eigenvector Difference (radians): {eigenvector_diff}")
