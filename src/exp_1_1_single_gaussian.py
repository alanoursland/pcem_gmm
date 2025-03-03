import torch
import torch.nn as nn

from pcem import ComponentGaussian

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
