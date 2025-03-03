# **Experiment Report: PCEM-GMM on Single Gaussian (2D)**

## **1. Experiment Overview**
- **Experiment ID**: 1.1 Single Gaussian
- **Objective**: Validate PCEM-GMM's ability to estimate the mean, eigenvalues, and eigenvectors of a single 2D Gaussian distribution.
- **Dataset**: Synthetic 2D Gaussian with predefined parameters.
- **Model(s) Used**: PCEM-GMM.
- **Metrics Evaluated**: Mean estimation accuracy, eigenvalue recovery, eigenvector alignment.

---

## **2. Experimental Setup**
### **2.1 Data Generation / Preprocessing**
- **Data Source**: Synthetic dataset generated from a single 2D Gaussian.
- **Number of Samples**: 1000
- **Dimensionality**: 2D
- **Gaussian Components**: 1
- **Ground Truth Parameters**:
  - **Mean**: `[5.0, -3.0]`
  - **Eigenvalues**: `[4.0, 1.0]`
  - **Eigenvectors**:
    ```
    [[-0.1904,  0.9817],
     [ 0.9817,  0.1904]]
    ```

### **2.2 Model Configuration**
- **Algorithm**: PCEM-GMM
- **Number of Components**: 1
- **Maximum Iterations**: N/A (single-component estimation)
- **Tolerance for Convergence**: N/A
- **Device Used**: GPU (CUDA)

---

## **3. Results & Analysis**
### **3.1 Estimated Parameters vs. Ground Truth**
#### **Means**
```plaintext
| Component | True Mean  | Estimated Mean  |
|-----------|-----------|----------------|
| 1         | [5.0, -3.0]  | [5.0006, -3.0769] |
```
#### **Eigenvalues**
```plaintext
| Component | True Eigenvalues | Estimated Eigenvalues |
|-----------|-----------------|-----------------------|
| 1         | [4.0, 1.0]       | [4.1197, 1.0575]     |
```
#### **Eigenvectors**
```plaintext
| Component | True Eigenvectors  | Estimated Eigenvectors |
|-----------|--------------------|------------------------|
| 1         | [[-0.1904,  0.9817],  | [[-0.1889,  0.9820],  |
|           |  [ 0.9817,  0.1904]]  |  [ 0.9820,  0.1889]]  |
```

---

## **4. Discussion & Interpretation**
### **4.1 Key Observations**
- **Mean Estimation**: The estimated mean is very close to the true mean, with a minor deviation of approximately `0.0769` in the second dimension.
- **Eigenvalue Recovery**: The estimated eigenvalues closely match the true eigenvalues, with a small overestimation.
- **Eigenvector Alignment**: The estimated eigenvectors align well with the true principal components, confirming accurate principal component extraction.
- **PCEM-GMM Stability**: The model successfully learned the covariance structure iteratively, demonstrating numerical stability.

### **4.2 Comparison with Other Models**
This experiment focused solely on PCEM-GMM. Future experiments will compare against:
- **Standard GMM** (full covariance estimation)
- **PCA-GMM** (dimensionality reduction before GMM fitting)

---

## **5. Conclusion & Next Steps**
- **Summary**: PCEM-GMM successfully estimated the mean, eigenvalues, and eigenvectors of a single 2D Gaussian with high accuracy.
- **Limitations**: Minor deviations in estimated parameters, though within an acceptable range.
- **Future Work**: Extend to multiple Gaussian components (Exp 1.2), higher-dimensional data (Exp 1.4), and real-world datasets.

---

## **6. Appendix**
- **Code used**: See `exp_1_1_single_gaussian.py`
- **Additional plots**: To be included in future reports when visualizations are necessary.

---

### **[Version: 2025-03-03]**