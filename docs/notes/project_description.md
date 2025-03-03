# Project Description: PCEM-GMM (Principal Component Expectation-Maximization for Gaussian Mixture Models)

## 1. Project Overview
### PCEM-GMM: A Principal Component-Based Expectation-Maximization Algorithm for Gaussian Mixture Models

### Purpose
Gaussian Mixture Models (GMMs) are widely used for probabilistic clustering, density estimation, and generative modeling. However, their applicability in high-dimensional spaces is hindered by computational and numerical stability issues, primarily due to the estimation of full covariance matrices. PCEM-GMM (Principal Component Expectation-Maximization for GMMs) proposes a modification to the standard Expectation-Maximization (EM) algorithm to address these challenges.

Instead of explicitly estimating full covariance matrices in the M-step, PCEM-GMM **iteratively constructs the covariance structure** using principal component extraction. This process avoids direct computation and inversion of large covariance matrices, making the algorithm more stable and scalable in high-dimensional settings.

### Key Idea
- **Traditional GMMs estimate full covariance matrices**, which become ill-conditioned or computationally expensive in high dimensions.
- **PCEM-GMM constructs covariance representations iteratively** by extracting principal components one at a time, deflating variance after each extraction.
- **This approach retains full covariance information** while improving numerical stability and reducing memory requirements.

---

## 2. Background Theory

### 2.1 Gaussian Mixture Models (GMMs)
A Gaussian Mixture Model represents data as a weighted sum of multiple Gaussian distributions:

\[
p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)
\]

where:
- \( K \) is the number of Gaussian components.
- \( \pi_k \) represents the weight (mixing coefficient) of the \( k \)th Gaussian.
- \( \mu_k \) is the mean vector of the \( k \)th Gaussian.
- \( \Sigma_k \) is the covariance matrix of the \( k \)th Gaussian.

A GMM is a powerful tool because it can approximate any continuous probability distribution given a sufficient number of components. It is widely used in applications such as speech recognition, anomaly detection, and image segmentation.

### 2.2 Expectation-Maximization (EM) for GMMs
The **Expectation-Maximization (EM) algorithm** is an iterative method used to estimate the parameters of GMMs. It consists of two main steps:

1. **E-Step (Expectation):** Compute the posterior probabilities (or responsibilities) \( \gamma_{ik} \) that each data point \( x_i \) belongs to each Gaussian component \( k \):

   \[
   \gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)}
   \]

2. **M-Step (Maximization):** Update the parameters of each Gaussian component based on the responsibilities:
   - **Mean Update:**
     \[
     \mu_k = \frac{\sum_{i=1}^{N} \gamma_{ik} x_i}{\sum_{i=1}^{N} \gamma_{ik}}
     \]
   - **Covariance Update:**
     \[
     \Sigma_k = \frac{\sum_{i=1}^{N} \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^T}{\sum_{i=1}^{N} \gamma_{ik}}
     \]

This process is repeated until convergence, typically measured by the change in log-likelihood of the data.

---

## 3. Challenges of GMMs in High Dimensions

### 3.1 The Curse of Dimensionality
As the dimensionality \( d \) of the data increases, several problems arise:
- **Quadratic Growth in Covariance Parameters:** A full covariance matrix \( \Sigma_k \) requires storing and updating \( O(d^2) \) values per Gaussian component. This becomes computationally prohibitive in high-dimensional settings.
- **Numerical Instability:** If the number of data points \( N \) is less than the dimensionality \( d \), the covariance matrix \( \Sigma_k \) is singular or nearly singular, making it difficult to compute \( \Sigma_k^{-1} \).
- **Inversion of Large Matrices:** The Mahalanobis distance, essential for responsibility computation, requires computing \( \Sigma_k^{-1} \), which is slow and unstable for large \( d \).

### 3.2 The Need for Dimensionality Reduction
A common strategy to handle high-dimensional data in GMMs is to **reduce the dimensionality before training**. Popular techniques include:
- **Principal Component Analysis (PCA) + GMM:** PCA is applied to project data into a lower-dimensional space before fitting a GMM.
- **Autoencoder + GMM:** Deep learning-based autoencoders compress data into a lower-dimensional latent space, where a GMM is then applied.
- **t-SNE / UMAP + GMM:** Dimensionality reduction techniques are used to create embeddings before applying a GMM.

While these approaches can help, they also have limitations:
- **Potential Information Loss:** If dimensionality reduction removes relevant structure, GMM performance suffers.
- **Preprocessing Overhead:** Running PCA or training an autoencoder is an additional computational step.
- **Static Transformations:** Once reduced, the data remains fixed, limiting adaptive learning.

PCEM-GMM eliminates the need for **external** dimensionality reduction by incorporating **iterative covariance construction directly into the EM process**.

---

## 4.3 Advantages of PCEM-GMM

### Numerical Stability
- Avoids explicit computation and inversion of large covariance matrices, preventing issues with singular or near-singular matrices.
- Power iteration provides a stable way to extract variance directions sequentially.

### Scalability
- Works efficiently in high-dimensional spaces where the number of data points \( N \) is smaller than the dimensionality \( d \).
- Since covariance matrices grow quadratically in standard GMMs, PCEM-GMM dramatically reduces memory requirements by storing only the principal variance directions.

### Interpretability
- Extracted principal components provide **meaningful variance directions** for each Gaussian component.
- Enables analysis of **how each Gaussian adapts to the data**, unlike standard covariance estimation, which provides limited interpretability.

### Retains Full Covariance Information
- Unlike dimensionality reduction methods that project data into a lower space **before** training, PCEM-GMM preserves full-rank covariance information **as it is learned**.
- Can approximate full-rank covariance matrices while storing only the most significant components.

---

## 5. Mathematical Formulation

### 5.1 Standard EM Formulation
The Expectation-Maximization (EM) algorithm for GMMs optimizes the **log-likelihood function**:

\[
\log p(X | \theta) = \sum_{i=1}^{N} \log \sum_{k=1}^{K} \pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)
\]

where \( X \) is the dataset, \( \theta = \{ \pi_k, \mu_k, \Sigma_k \} \) are the parameters, and \( \mathcal{N}(x | \mu_k, \Sigma_k) \) is the Gaussian density function.

#### **E-Step**
The responsibility of component \( k \) for point \( x_i \) is computed as:

\[
\gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)}
\]

#### **M-Step**
Parameters are updated as follows:

- **Mean update:**
  
  \[
  \mu_k = \frac{\sum_{i=1}^{N} \gamma_{ik} x_i}{\sum_{i=1}^{N} \gamma_{ik}}
  \]

- **Covariance update:**
  
  \[
  \Sigma_k = \frac{\sum_{i=1}^{N} \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^T}{\sum_{i=1}^{N} \gamma_{ik}}
  \]

This standard approach becomes **numerically unstable in high dimensions**, motivating our modification.

---

### 5.2 Modified M-Step in PCEM-GMM

Instead of computing \( \Sigma_k \) explicitly, PCEM-GMM **iteratively constructs** the covariance structure.

#### **Step 1: Compute Component Means**
\[
\mu_k = \frac{\sum_{i=1}^{N} \gamma_{ik} x_i}{\sum_{i=1}^{N} \gamma_{ik}}
\]

#### **Step 2: Iterative Principal Component Extraction**
1. **Compute the scatter matrix (without forming it explicitly):**
   \[
   S_k = \sum_{i=1}^{N} \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^T
   \]

2. **Find the dominant eigenvector \( v_1 \) using power iteration:**
   - Initialize \( v_1 \) randomly.
   - Iterate:
     \[
     v_{t+1} = \frac{S_k v_t}{\| S_k v_t \|}
     \]
   - Stop when \( v_t \) converges.

3. **Compute the corresponding eigenvalue (variance magnitude):**
   \[
   \lambda_1 = \frac{1}{\sum_{i=1}^{N} \gamma_{ik}} \sum_{i=1}^{N} \gamma_{ik} ((x_i - \mu_k)^T v_1)^2
   \]

4. **Deflate the data along \( v_1 \):**
   \[
   x_i' = x_i - ((x_i - \mu_k)^T v_1) v_1
   \]

5. **Repeat for additional principal components** until variance is sufficiently captured.

6. **Store covariance as a set of ranked eigenvectors and eigenvalues**:
   \[
   \Sigma_k \approx \sum_{j=1}^{r} \lambda_j v_j v_j^T
   \]
   where \( r \leq d \), reducing computational complexity.

This eliminates the need for direct **covariance matrix inversion**, enhancing numerical stability.

---

## 6. Experimental Validation

### 6.1 Comparison with Standard GMM
To demonstrate the effectiveness of PCEM-GMM, we compare it against standard GMMs using:
- **Log-likelihood** on test data.
- **Convergence speed** (number of iterations and runtime).
- **Cluster quality** (Adjusted Rand Index, Silhouette Score).

### 6.2 PCEM-GMM vs. PCA-GMM
We test whether **applying PCA first** (PCA-GMM) vs. **PCEM-GMM’s on-the-fly covariance learning** retains more variance and structure.

### 6.3 High-Dimensional Synthetic Data
We construct datasets where \( d > N \) to show that standard GMMs fail due to singular covariance matrices, while PCEM-GMM remains stable.

### 6.4 Real-World Applications
PCEM-GMM is tested on:
- **High-dimensional text data (TF-IDF vectors).**
- **Financial data (correlated stock returns).**
- **Genomics data (gene expression profiles).**

---

## 7. Conclusion

- **PCEM-GMM introduces a novel way of learning covariance matrices iteratively**, improving stability in high dimensions.
- It avoids **explicit covariance computation and inversion**, making it memory-efficient.
- The model performs well compared to standard GMMs, especially in settings where \( d > N \).

PCEM-GMM represents a **practical alternative** for using GMMs in domains where standard covariance estimation is problematic.

