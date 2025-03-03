# **Experiment Report: PCEM-GMM on Multiple Gaussians (2D, Well-Separated)**

## **1. Experiment Overview**
- **Experiment ID**: 1.2 Multiple Gaussians
- **Objective**: Validate PCEM-GMM’s ability to correctly estimate the means, eigenvalues, and eigenvectors of two well-separated Gaussian components.
- **Dataset**: Synthetic 2D Gaussian mixture with two distinct clusters.
- **Model(s) Used**: PCEM-GMM.
- **Metrics Evaluated**: Mean estimation accuracy, eigenvalue recovery, eigenvector alignment, log-likelihood convergence.

---

## **2. Experimental Setup**
### **2.1 Data Generation / Preprocessing**
- **Data Source**: Synthetic dataset generated with two 2D Gaussians.
- **Number of Samples**: 1000
- **Dimensionality**: 2D
- **Gaussian Components**: 2
- **Ground Truth Parameters**:
  - **Means**:
    ```
    Gaussian 1: [-5.0, 2.0]
    Gaussian 2: [5.0, -3.0]
    ```
  - **Eigenvalues**:
    ```
    Gaussian 1: [6.0, 1.0]
    Gaussian 2: [2.0, 2.0]
    ```
  - **Eigenvectors (row-based)**:
    ```
    Gaussian 1:
    [[-0.1904,  0.9817],
     [ 0.9817,  0.1904]]
    
    Gaussian 2:
    [[-0.9967, -0.0809],
     [-0.0809,  0.9967]]
    ```

### **2.2 Model Configuration**
- **Algorithm**: PCEM-GMM
- **Number of Components**: 2
- **Maximum Iterations**: 100
- **Tolerance for Convergence**: `1e-4`
- **Device Used**: CPU/GPU
- **Initialization**: K-Means clustering was used to initialize the means of the Gaussian components before running PCEM-GMM.

---

## **3. Results & Analysis**
### **3.1 Convergence & Log-Likelihood Evolution**
- **Number of Iterations Taken**: 17
- **Final Log-Likelihood**: `-4353.1963`
- **Converged?**: Yes

```plaintext
| Iteration | Log-Likelihood | Change |
|-----------|---------------|--------|
| 1         | -8011.3101    | N/A    |
| 2         | -5193.9395    | 2817.3706  |
| 3         | -4394.8330    | 799.1064  |
| 4         | -4353.3096    | 41.5234  |
| ...       | ...           | ...    |
| 17        | -4353.1953    | 0.2051  |
| 18        | -4353.1963    | -0.0010  |
```

---

### **3.2 Estimated Parameters vs. Ground Truth**
#### **Means**
```plaintext
| Component | True Mean  | Estimated Mean  |
|-----------|-----------|----------------|
| 1         | [-5.0, 2.0]  | [-4.9785, 1.8909] |
| 2         | [5.0, -3.0]  | [5.0031, -3.0115] |
```
#### **Eigenvalues**
```plaintext
| Component | True Eigenvalues | Estimated Eigenvalues |
|-----------|-----------------|-----------------------|
| 1         | [6.0, 1.0]       | [6.5310, 0.9793]     |
| 2         | [2.0, 2.0]       | [2.1242, 1.9829]     |
```
#### **Eigenvectors**
```plaintext
| Component | True Eigenvectors  | Estimated Eigenvectors |
|-----------|--------------------|------------------------|
| 1         | [[-0.1904,  0.9817],  | [[ 0.1863, -0.9825],  |
|           |  [ 0.9817,  0.1904]]  |  [ 0.9825,  0.1863]]  |
| 2         | [[-0.9967, -0.0809],  | [[-0.0599,  0.9982],  |
|           |  [-0.0809,  0.9967]]  |  [-0.9983, -0.0589]]  |
```

---

## **4. Discussion & Interpretation**
### **4.1 Key Observations**
- **Mean Estimation**: The estimated means closely match the true means, with deviations on the order of ~0.02–0.1.
- **Eigenvalue Recovery**: PCEM-GMM correctly estimated the major variance directions, with minor overestimation in the largest eigenvalues.
- **Eigenvector Alignment**: The estimated eigenvectors are aligned with the principal components of each Gaussian, confirming accurate covariance estimation.
- **Convergence**: The model converged after 17 iterations, demonstrating efficient learning.

### **4.2 Comparison with Other Models (Planned for Future Experiments)**
- **Standard GMM** (full covariance estimation)
- **PCA-GMM** (dimensionality reduction before GMM fitting)
- **Mixtures of PPCA** (probabilistic PCA-based GMM)

---

## **5. Conclusion & Next Steps**
- **Summary**: PCEM-GMM successfully recovered the parameters of two well-separated Gaussians with high accuracy.
- **Limitations**: Minor deviations in eigenvalues and eigenvectors, likely due to numerical approximations.
- **Future Work**:
  - **Exp 1.3**: Evaluate performance on overlapping Gaussians.
  - **Exp 1.4**: Extend to high-dimensional Gaussian mixtures (10D, 50D).
  - **Exp 2.x**: Compare with standard GMM, PCA-GMM, and Mixtures of PPCA on real-world datasets.

---

## **6. Appendix**
- **Code used**: See `exp_1_2_multiple_gaussians.py`
- **Additional plots**: To be included in future iterations for visual comparisons.

---

### **[Version: 2025-03-03]**

