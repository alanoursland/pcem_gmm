# **Experiment Report: PCEM-GMM on Overlapping Gaussians (2D)**

## **1. Experiment Overview**
- **Experiment ID**: 1.3 Overlapping Gaussians
- **Objective**: Evaluate PCEM-GMM’s ability to estimate and distinguish overlapping Gaussian components.
- **Dataset**: Synthetic 2D Gaussian mixture with two overlapping clusters.
- **Model(s) Used**: PCEM-GMM, Standard GMM, PCA-GMM.
- **Metrics Evaluated**: Clustering accuracy, confusion matrix, mean estimation, eigenvalue recovery, eigenvector alignment.

---

## **2. Experimental Setup**
### **2.1 Data Generation / Preprocessing**
- **Data Source**: Synthetic dataset generated with two overlapping 2D Gaussians.
- **Number of Samples**: 1000
- **Dimensionality**: 2D
- **Gaussian Components**: 2
- **Ground Truth Parameters**:
  - **Means**:
    ```
    Component 1: [-1.0, 1.0]
    Component 2: [1.0, -1.0]
    ```
  - **Eigenvalues**:
    ```
    Component 1: [0.5, 3.0]
    Component 2: [1.5, 2.5]
    ```
  - **Eigenvectors**:
    ```
    Component 1:
    [[-0.9817, -0.1904],
     [-0.1904,  0.9817]]
    
    Component 2:
    [[ 0.0809, -0.9967],
     [-0.9967, -0.0809]]
    ```

### **2.2 Model Configuration**
- **Algorithm**: PCEM-GMM
- **Number of Components**: 2
- **Maximum Iterations**: 100
- **Tolerance for Convergence**: `1e-4` (tuned for improved accuracy)
- **Device Used**: CPU/GPU
- **Initialization**: K-Means clustering was used to initialize the means. Without K-Means, accuracy was in the 60s; initializing means to ground truth yielded results nearly identical to the theoretical maximum.

---

## **3. Results & Analysis**
### **3.1 Clustering Performance**
#### **Confusion Matrices & Accuracy**
```plaintext
| Model       | Accuracy |
|------------|----------|
| Ground Truth GMM | 0.853 |
| PCEM-GMM   | 0.846 |
| Standard GMM | 0.843 |
| PCA-GMM    | 0.843 |
```

```plaintext
Ground Truth GMM Confusion Matrix:
 [[417  72]
 [ 75 436]]

PCEM-GMM Confusion Matrix:
 [[391  98]
 [ 56 455]]

Standard GMM Confusion Matrix:
 [[396  93]
 [ 64 447]]

PCA-GMM Confusion Matrix:
 [[396  93]
 [ 64 447]]
```

**Note:** The theoretical maximum classification accuracy is given by the ground truth model, as even it "misclassifies" data near the mixture point due to the inherent overlap in distributions.

---

### **3.2 Estimated Parameters vs. Ground Truth**
#### **Means**
```plaintext
| Component | True Mean  | PCEM-GMM Estimated Mean |
|-----------|-----------|------------------------|
| 1         | [-1.0, 1.0]  | [-1.0609, 1.1526]    |
| 2         | [1.0, -1.0]  | [0.9170, -1.0536]    |
```
#### **Eigenvalues**
```plaintext
| Component | True Eigenvalues | PCEM-GMM Estimated Eigenvalues |
|-----------|-----------------|-------------------------------|
| 1         | [0.5, 3.0]       | [0.4610, 2.8127]             |
| 2         | [1.5, 2.5]       | [1.5457, 2.4435]             |
```
#### **Eigenvectors**
```plaintext
| Component | True Eigenvectors  | PCEM-GMM Estimated Eigenvectors |
|-----------|--------------------|--------------------------------|
| 1         | [[-0.9817, -0.1904],  | [[-0.9877, -0.1564],  |
|           |  [-0.1904,  0.9817]]  |  [-0.1564,  0.9877]]  |
| 2         | [[ 0.0809, -0.9967],  | [[ 0.1452, -0.9894],  |
|           |  [-0.9967, -0.0809]]  |  [-0.9894, -0.1452]]  |
```

---

## **4. Discussion & Interpretation**
### **4.1 Key Observations**
- **Clustering Accuracy**: PCEM-GMM achieved `84.6%` accuracy, comparable to Standard GMM and PCA-GMM, but slightly lower than the ground truth model (`85.3%`).
- **Effect of Initialization**: Without K-Means, PCEM-GMM accuracy was in the 60s. When initialized with the true means, accuracy was only slightly lower than the ground truth.
- **Mean Estimation**: PCEM-GMM estimated means closely match the true means, with deviations of ~0.05-0.15.
- **Eigenvalue Recovery**: PCEM-GMM successfully estimated major variance directions, with some deviation in smaller eigenvalues.
- **Eigenvector Alignment**: The estimated eigenvectors are well-aligned with the principal components of each Gaussian.
- **Comparison with Other Models**:
  - Standard GMM and PCA-GMM achieved slightly lower clustering accuracy than PCEM-GMM.
  - PCEM-GMM provides a numerically stable covariance estimation, making it preferable for high-dimensional extensions.

---

## **5. Conclusion & Next Steps**
- **Summary**: PCEM-GMM effectively estimated the parameters of overlapping Gaussians with competitive clustering accuracy.
- **Limitations**: Slight deviations in eigenvalues and eigenvectors, possibly due to overlapping distributions.
- **Future Work**:
  - **Exp 1.4**: Extend to high-dimensional Gaussian mixtures (10D, 50D) to stress-test numerical stability.
  - **Exp 2.x**: Apply PCEM-GMM to real-world datasets for further evaluation.
  - **Optimize** initialization strategies for improved convergence in overlapping distributions.

---

## **6. Appendix**
- **Code used**: See `exp_1_3_overlapping_gaussians.py`
- **Additional plots**: To be included in future iterations for visual comparisons.

---

### **[Version: 2025-03-03]**

