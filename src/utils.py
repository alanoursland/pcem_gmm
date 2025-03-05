import time
import torch
import torch.nn as nn
import torch.optim as optim
import random

from torchvision import datasets, transforms 

class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.last_tick = self.start_time
    
    def tick(self):
        now = time.time()
        tick = now - self.last_tick
        self.last_tick = now
        return tick
    
    def elapsed(self):
        return time.time() - self.start_time

# Sampler class for random batch sampling
class Sampler:
    def __init__(self, data, labels, batch_size, device='cuda'):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.device = device
        self.num_samples = data.size(0)

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        indices = torch.randint(0, self.num_samples, (batch_size,), device=self.device)
        inputs = self.data[indices]
        targets = self.labels[indices]
        return inputs, targets

    def all(self):
        return self.data, self.labels

# Visualization function
def visualize_gaussian_components(model, data_samples, targets=None):
    # Project data samples to the same latent space as the means
    with torch.no_grad():
        projected_data = model(data_samples)
    variances = model.variances()

    plt.figure(figsize=(8, 6))
    if targets is not None:
        scatter = plt.scatter(projected_data[:, 0].detach().cpu().numpy(), 
                              projected_data[:, 1].detach().cpu().numpy(), 
                              c=targets.detach().cpu().numpy(), cmap='tab10', s=10, alpha=0.6, label='Projected Data Samples')
        plt.colorbar(scatter, label='Target Class')
    else:
        plt.scatter(projected_data[:, 0].detach().cpu().numpy(), projected_data[:, 1].detach().cpu().numpy(), s=10, alpha=0.6, label='Projected Data Samples')
    # plt.scatter(means.detach().cpu().numpy(), 
    #             np.zeros_like(means.detach().cpu().numpy()), 
    #             color='red', marker='x', s=100, label='Gaussian Means')
    plt.title('Visualization of Gaussian Components in Latent Space')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.legend()
    plt.show()

# Cosine similarity visualization function
def plot_cosine_similarity_histogram(model):
    weight_vectors = model.linear.weight.detach()
    cosine_similarities = []
    for i in range(weight_vectors.size(0)):
        for j in range(i + 1, weight_vectors.size(0)):
            cos_sim = torch.nn.functional.cosine_similarity(weight_vectors[i].unsqueeze(0), weight_vectors[j].unsqueeze(0)).item()
            cosine_similarities.append(cos_sim)

    plt.figure(figsize=(8, 6))
    plt.hist(cosine_similarities, bins=50, alpha=0.75, color='blue')
    plt.title('Histogram of Cosine Similarities Between Weight Vectors')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.show()

class Abs(nn.Module):
    def forward(self, x):
        return torch.abs(x)

class OffsetClassifierModel(nn.Module):
    def __init__(self, input_model, output_dim, latent_dim, activation=nn.ReLU()):
        super(ClassifierModel, self).__init__()
        self.input_model = input_model
        self.bias_offsets = nn.Parameter(torch.zeros(output_dim, latent_dim))
        self.activation = activation
        self.linear = nn.Linear(latent_dim*output_dim, output_dim)

    def forward(self, x):
        with torch.no_grad():
            x = self.input_model(x)
        x = x.unsqueeze(1) + self.bias_offsets
        x = x.flatten(-2)
        x = self.activation(x)
        x = self.linear(x)
        return x

class SimpleClassifierModel2(nn.Module):
    def __init__(self, input_model, output_dim, latent0_dim, activation=nn.ReLU()):
        super(SimpleClassifierModel2, self).__init__()
        for param in input_model.parameters():
            param.requires_grad = False
        self.model = nn.Sequential(
            input_model,
            activation,
            nn.Linear(latent0_dim, output_dim),
        )
    def forward(self, x):
        return self.model(x)

class SimpleClassifierModel3(nn.Module):
    def __init__(self, input_model, output_dim, latent0_dim, latent1_dim, activation=nn.ReLU()):
        super(SimpleClassifierModel3, self).__init__()
        for param in input_model.parameters():
            param.requires_grad = False
        self.model = nn.Sequential(
            input_model,
            activation,
            nn.Linear(latent0_dim, latent1_dim),
            activation,
            nn.Linear(latent1_dim, output_dim)
        )
    def forward(self, x):
        return self.model(x)

class MLP2(nn.Module):
    def __init__(self, input_dim, latent0, output_dim, activation=nn.ReLU()):
        super(MLP2, self).__init__()
        self.activation = activation
        self.linear0 = nn.Linear(input_dim, latent0)
        self.linear1 = nn.Linear(latent0, output_dim)
        self.model = nn.Sequential(
            self.linear0,
            self.activation,
            self.linear1
        )

    def forward(self, x):
        return self.model(x)

class MLP3(nn.Module):
    def __init__(self, input_dim, latent0, latent1, output_dim, activation=nn.ReLU()):
        super(MLP3, self).__init__()
        self.activation = activation
        self.linear0 = nn.Linear(input_dim, latent0)
        self.linear1 = nn.Linear(latent0, latent1)
        self.linear2 = nn.Linear(latent1, output_dim)
        self.model = nn.Sequential(
            self.linear0,
            self.activation,
            self.linear1,
            self.activation,
            self.linear2
        )

    def forward(self, x):
        return self.model(x)

# Supervised training function
def train_supervised(model, sampler, num_outputs, num_epochs=100, lr=0.001, log_every=100, device="cuda"):
    timer = Timer()

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.9)
    criterion = nn.CrossEntropyLoss()

    print('Starting supervised training')
    timer.tick()
    for update in range(num_epochs):
        inputs, targets = sampler.sample()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        elapsed = timer.tick()

        # with torch.no_grad():
        #     train_data, train_labels = sampler.all()
        #     y = model(train_data)
        #     prediction = torch.argmax(y, dim=1)
        #     correct = (prediction == train_labels)
        #     train_accuracy = correct.sum().float() / correct.size(0)
        train_accuracy = 0

        if (update+1) % log_every == 0:
            print(f"Update: {update + 1}/{num_epochs} | Loss: {loss.item():.4f} | Train Acc: {100*train_accuracy:.2f}% | Elapsed Time: {timer.elapsed():.2f}s | Step Time: {elapsed:.4f}s")
    print('Supervised training complete.')
    return model

def evaluate_model(model, sampler, batch_size=None, device='cuda'):
    """
    Evaluate a model on the entire dataset provided by the sampler.
    
    Args:
        model: The model to evaluate
        sampler: Sampler containing the evaluation data
        batch_size: Batch size for evaluation. If None, process all data at once.
        device: Device to run evaluation on
        
    Returns:
        accuracy: Classification accuracy as a percentage
    """
    model.eval()  # Set model to evaluation mode
    
    # Get all data at once
    data, labels = sampler.all()
    total_samples = data.size(0)
    
    with torch.no_grad():  # No need to track gradients during evaluation
        if batch_size is None:
            # Process all data at once
            y = model(data)
            prediction = torch.argmax(y, dim=1)
            correct = (prediction == labels)
            accuracy = correct.sum().float().item() / total_samples
        else:
            # Process in batches
            correct = 0
            for i in range(0, total_samples, batch_size):
                batch_data = data[i:i+batch_size]
                batch_labels = labels[i:i+batch_size]
                
                outputs = model(batch_data)
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == batch_labels).sum().float().item()
            accuracy = correct / total_samples

    return accuracy

def load_mnist(root='e:/ml_datasets', batch_size=512, device='cuda'):
    # Define the transform to normalize the data
    transform = lambda x: x.float().view(-1, 28 * 28) / 255.0
    
    # Load MNIST datasets (both train and test)
    mnist_train = datasets.MNIST(root=root, train=True, download=True)
    mnist_test = datasets.MNIST(root=root, train=False, download=True)
    
    # Convert to tensors and move to device
    train_data = transform(mnist_train.data).to(device)
    train_labels = mnist_train.targets.to(device)
    
    test_data = transform(mnist_test.data).to(device)
    test_labels = mnist_test.targets.to(device)
    
    # Create samplers for both train and test
    train_sampler = Sampler(train_data, train_labels, batch_size, device)
    test_sampler = Sampler(test_data, test_labels, batch_size, device)
    
    return train_sampler, test_sampler, 28*28  # Return both samplers and input dimension

def load_cifar10(root='e:/ml_datasets', batch_size=512, device='cuda'):
    # Define the transform to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Flatten the images to 1D vectors
        transforms.Lambda(lambda x: x.view(-1))
    ])
    
    # Load CIFAR-10 datasets (both train and test)
    cifar_train = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    cifar_test = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    
    # Convert to tensors and move to device
    train_data = torch.stack([sample[0] for sample in cifar_train]).to(device)
    train_labels = torch.tensor([sample[1] for sample in cifar_train], dtype=torch.long).to(device)
    
    test_data = torch.stack([sample[0] for sample in cifar_test]).to(device)
    test_labels = torch.tensor([sample[1] for sample in cifar_test], dtype=torch.long).to(device)
    
    # Create samplers for both train and test
    train_sampler = Sampler(train_data, train_labels, batch_size, device)
    test_sampler = Sampler(test_data, test_labels, batch_size, device)
    
    return train_sampler, test_sampler, 3072  # Return both samplers and input dimension

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

