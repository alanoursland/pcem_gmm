import torch
import numpy as np
import time
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from pcem import ComponentGMM, ComponentGaussian
from metrics import compare_means, compare_eigenvalues, compare_eigenvectors, calculate_ari, calculate_silhouette_score, compute_clustering_metrics

def sigmoid(x, midpoint=0.5, steepness=10):
    """Sigmoid function for eigenvalue decay."""
    return 1 / (1 + np.exp(-steepness * (x - midpoint)))

def generate_eigenvalues(dim, midpoint=0.5, steepness=10):
    """Generate eigenvalues for each Gaussian using a sigmoid decay."""
    eigenvalues = sigmoid(np.linspace(0, 1, dim), midpoint, steepness)
    eigenvalues = eigenvalues / eigenvalues.sum()  # Normalize to avoid too large eigenvalues
    return eigenvalues

def random_rotation_matrix(dim):
    """Generate a random orthonormal matrix for eigenvectors."""
    mat = np.random.randn(dim, dim)
    q, _ = np.linalg.qr(mat)  # QR decomposition gives orthonormal matrix
    return torch.tensor(q, dtype=torch.float32)

def print_model_parameters(name, means, covariances):
    """
    Print the parameters (means, eigenvalues, eigenvectors) of the GMM model.
    :param name: The name of the model (e.g., "Groundtruth", "PCEM-GMM", etc.)
    :param means: List or array of means for each component
    :param covariances: List or array of covariance matrices (or their eigenvalues/eigenvectors) for each component
    """
    print(f"\n{name} Parameters:")
    for i, (mean, cov) in enumerate(zip(means, covariances)):
        print(f"Component {i+1}:")
        print(f"  Mean: {mean}")
        print(f"  Eigenvalues: {np.linalg.eigvalsh(cov)}")  # Compute eigenvalues of the covariance matrix
        print(f"  Eigenvectors:\n{np.linalg.eigh(cov)[1]}")  # Eigenvectors of the covariance matrix

def create_groudtruth(num_gaussians, num_samples, dim):
    """Generate a synthetic dataset with multiple Gaussians using principal components."""
    means = [np.random.uniform(-5, 5, dim) for _ in range(num_gaussians)]
    eigenvalues = [generate_eigenvalues(dim, midpoint=np.random.uniform(0.2, 0.8), steepness=np.random.uniform(5, 15)) for _ in range(num_gaussians)]
    eigenvectors = [random_rotation_matrix(dim) for _ in range(num_gaussians)]
    
    # Set up ComponentGMM
    model = ComponentGMM(n_gaussians=num_gaussians, n_dimensions=dim, device=torch.device("cpu"))
    
    for i in range(num_gaussians):
        gaussian = model.gaussians[i]
        gaussian.set_mean(torch.tensor(means[i], dtype=torch.float32))
        gaussian.set_eigenvectors(eigenvectors[i].clone().detach().to(torch.float32))
        gaussian.set_eigenvalues(torch.tensor(eigenvalues[i], dtype=torch.float32))

    # # Adding background noise (1 large, low amplitude Gaussian)
    # noise_component = ComponentGaussian(dim, device=torch.device("cpu"))
    # noise_component.set_mean(torch.zeros(dim, dtype=torch.float32))
    # noise_component.set_eigenvectors(torch.eye(dim, dtype=torch.float32))  # Identity matrix, no rotation
    # noise_component.set_eigenvalues(torch.full((dim,), 0.01, dtype=torch.float32))  # Low amplitude
    
    # model.gaussians.append(noise_component)  # Add the noise gaussian to the model
    model.use_grad(False)
    return model

# Experiment Settings
# dimensions_list = [10, 50]  # Test 10D and 50D cases
dimensions_list = [10]
num_gaussians = 3  # Number of Gaussians
num_samples = 5000  # Fewer samples than dimensions to stress-test PCEM-GMM
max_iterations = 100
tolerance = 1e-6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for dim in dimensions_list:
    print(f"\n=== Running Experiment 1.4 in {dim}D ===")
    groundtruth = create_groudtruth(num_gaussians, num_samples, dim).to(device)
    x, y = groundtruth.sample(5000)

    # # PCEM-GMM
    # pcem_model = ComponentGMM(num_gaussians, dim, max_iterations, tolerance, device=device)
    # start_time = time.time()
    # pcem_model.fit(x)
    # pcem_time = time.time() - start_time
    
    # Standard GMM (Full Covariance)
    start_time = time.time()
    try:
        gmm = GaussianMixture(n_components=num_gaussians, covariance_type='full', max_iter=max_iterations, tol=tolerance, random_state=42)
        gmm.fit(x.cpu().numpy())
        gmm_time = time.time() - start_time
    except Exception as e:
        gmm_time = "FAILED"
        print(f"Standard GMM failed in {dim}D: {e}")
    
    # PCA-GMM
    start_time = time.time()
    pca = PCA(n_components=5)  # Reduce dimensionality before GMM
    reduced_samples = pca.fit_transform(x.cpu().numpy())
    pca_gmm = GaussianMixture(n_components=num_gaussians, covariance_type='full', max_iter=max_iterations, tol=tolerance, random_state=42)
    pca_gmm.fit(reduced_samples)
    pca_time = time.time() - start_time
    
    print(f"\nResults for {dim}D:")
    # print(f"PCEM-GMM Time: {pcem_time:.2f} sec")
    print(f"Standard GMM Time: {gmm_time} sec")
    print(f"PCA-GMM Time: {pca_time:.2f} sec")
    # print(f"PCEM-GMM Converged in {pcem_model.n_iterations_} iterations")

    # ====== Groundtruth: CEM-GMM ======
    print(" ========================= Groundtruth: CEM-GMM ========================= ")
    responsibilities, _ = groundtruth.e_step(x)
    component_assignments = torch.argmax(responsibilities, dim=1).cpu().numpy()
    conf_matrix, accuracy = compute_clustering_metrics(y.cpu(), component_assignments)
    print("Groundtruth Confusion Matrix:\n", conf_matrix)
    print("Groundtruth Clustering Accuracy:", accuracy)

    # Extract ground truth parameters
    ground_means = np.array([c.mean.cpu().numpy() for c in groundtruth.gaussians])
    ground_covariances = [
        c.eigenvectors.cpu().numpy() @ np.diag(c.eigenvalues.cpu().numpy()) @ c.eigenvectors.T.cpu().numpy()
        for c in groundtruth.gaussians
    ]    
    print_model_parameters("Ground Truth GMM", ground_means, ground_covariances)

    # # ====== Experiment: CEM-GMM ======
    # print(" ========================= Experiment: CEM-GMM ========================= ")
    # responsibilities, _ = groundtruth.e_step(x)
    # component_assignments = torch.argmax(responsibilities, dim=1).cpu().numpy()
    # conf_matrix, accuracy = compute_clustering_metrics(y, component_assignments)
    # print("PCEM-GMM Confusion Matrix:\n", conf_matrix)
    # print("PCEM-GMM Clustering Accuracy:", accuracy)

    # # Compare means, eigenvalues, and eigenvectors using metrics
    # for i, gaussian in enumerate(groundtruth.gaussians):
    #     print(f"\nComparing Component {i+1}:")
    #     mean_diff = compare_means(gaussian.mean.cpu().numpy(), groundtruth.gaussians[i].mean.cpu().numpy())
    #     print(f"Mean Difference (Component {i+1}): {mean_diff}")

    #     eigenvalue_diff = compare_eigenvalues(gaussian.eigenvalues.cpu().numpy(), groundtruth.gaussians[i].eigenvalues.cpu().numpy())
    #     print(f"Eigenvalue Difference (Component {i+1}): {eigenvalue_diff}")

    #     eigenvector_diff = compare_eigenvectors(gaussian.eigenvectors.T.cpu().numpy(), groundtruth.gaussians[i].eigenvectors.T.cpu().numpy())
    #     print(f"Eigenvector Difference (Component {i+1}): {eigenvector_diff}")

    # # Clustering metrics: ARI and Silhouette Score for Standard GMM
    # ari_score = calculate_ari(y, component_assignments)
    # silhouette_score = calculate_silhouette_score(x, component_assignments)

    # print(f"\nAdjusted Rand Index (ARI): {ari_score}")
    # print(f"Silhouette Score: {silhouette_score}")
    
    # ====== Baseline: Standard GMM ======
    print(" ========================= Baseline: Standard GMM  ========================= ")
    gmm_assignments = gmm.predict(x.cpu().numpy())
    conf_matrix_gmm, accuracy_gmm = compute_clustering_metrics(y.cpu(), gmm_assignments)
    print("\nStandard GMM Confusion Matrix:\n", conf_matrix_gmm)
    print("Standard GMM Clustering Accuracy:", accuracy_gmm)
    print(f"Standard GMM Silhouette Score: {calculate_silhouette_score(x.cpu(), gmm_assignments)}")

    # ====== Baseline: PCA-GMM ======
    print(" ========================= Baseline: PCA-GMM  ========================= ")
    pca_gmm_assignments = pca_gmm.predict(reduced_samples)
    conf_matrix_pca_gmm, accuracy_pca_gmm = compute_clustering_metrics(y.cpu(), pca_gmm_assignments)
    print("\nPCA-GMM Confusion Matrix:\n", conf_matrix_pca_gmm)
    print("PCA-GMM Clustering Accuracy:", accuracy_pca_gmm)
    print(f"PCA-GMM Silhouette Score: {calculate_silhouette_score(reduced_samples, pca_gmm_assignments)}")
