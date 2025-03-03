### **PCEM-GMM Experiment Plan**  
This plan is structured **iteratively**, starting with simple synthetic cases and progressing to complex real-world datasets. Each step will serve both **software validation** and **academic rigor** for the final paper.  

---

## **1. Baseline Experiments: Synthetic Data (Validation & Debugging)**
### **1.1 Single Gaussian (2D) [Completed]**
- **Purpose**: Verify that PCEM-GMM correctly extracts principal components from a **single Gaussian**.
- **Metrics**: Compare estimated **means, eigenvectors, and eigenvalues** to ground truth.
- **Expected Outcome**: Near-exact recovery of true parameters.  

✅ **Completed**  
🔄 Can be expanded to **higher dimensions (e.g., 10D)** to test numerical stability.  

### **1.2 Multiple Gaussians (2D, Well-Separated)**
- **Purpose**: Test **full EM behavior**, ensuring correct **component assignments**.
- **Design**:
  - Two well-separated Gaussians in 2D.
  - Different eigenvalue scales (e.g., one elongated, one spherical).
- **Expected Outcome**:
  - Each Gaussian's **principal components are correctly estimated**.
  - Assignments **match ground truth labels**.
- **Comparison**: Compare against **standard GMM (full covariance)** and **PCA-GMM**.

---

### **1.3 Multiple Gaussians (2D, Overlapping)**
- **Purpose**: Evaluate **cluster separation ability** when Gaussians have significant overlap.
- **Design**:
  - Two Gaussians with similar means but different covariances.
  - Some points will be ambiguous in assignment.
- **Metrics**: **Log-likelihood, clustering accuracy, and confusion matrices**.
- **Expected Outcome**:  
  - PCEM-GMM should **adapt dynamically** rather than assuming a **fixed subspace (PCA-GMM).**
  - Compare against PCA-GMM and Mixtures of PPCA.

---

### **1.4 Higher-Dimensional Synthetic Gaussians (10D, 50D)**
- **Purpose**: Stress-test numerical stability and efficiency in **high dimensions**.
- **Design**:
  - Multiple Gaussians in 10D and 50D.
  - Randomized eigenvalue spectra (some directions high variance, others low variance).
- **Metrics**:
  - **Convergence speed** of power iteration.
  - **Recovery error** in eigenvalues.
- **Comparison**: Standard GMM, PCA-GMM, and Mixtures of PPCA.
- **Expected Outcome**:  
  - PCEM-GMM should remain stable while standard GMMs **struggle with singular covariances**.

---

## **2. Real-World Datasets for Gaussian Mixture Models**
We now test **canonical datasets for GMMs**, with a focus on **datasets where standard methods fail**.

### **2.1 Gaussian Toy Datasets (Sanity Check)**
- **Scikit-Learn "Blobs"**: Well-separated Gaussians (should be trivial).
- **Scikit-Learn "Moons" & "Circles"**: Non-Gaussian data, useful for failure cases.

Expected: PCEM-GMM should work well for **Blobs**, but **struggle on Moons/Circles** (since they are **not Gaussian-like**).

---

### **2.2 High-Dimensional GMM Benchmarks**
#### **2.2.1 Statlog Landsat Satellite Dataset**
- **Purpose**: Tests GMM clustering in high-dimensional structured data.
- **Details**:
  - 36-dimensional feature space (spectral bands from satellite imagery).
  - 6 classes.
- **Why it's good**:  
  - Standard GMM **overfits** due to high dimensionality.
  - PCEM-GMM can potentially **learn lower-rank covariance structures**.

🔗 Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Statlog+(Landsat+Satellite))

---

#### **2.2.2 EMNIST (Handwritten Digits)**
- **Purpose**: Test **unsupervised clustering** on **image data** (28x28 pixels → 784D).
- **Why it's good**:
  - PCA-GMM **loses too much information**.
  - Standard GMM **is numerically unstable** in 784D.
- **Experiment**:
  - Use **flattened raw pixel intensities** (no feature engineering).
  - Evaluate cluster quality per digit.
  - Compare against **PCA-GMM and Variational GMMs**.

🔗 Dataset: [EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)

---

#### **2.2.3 MNIST with Feature Representations**
- **Purpose**: Compare PCEM-GMM against PCA-GMM on more structured **feature-extracted data**.
- **Experiment Variants**:
  - **Raw Pixels (784D)**.
  - **PCA-Reduced (50D, 100D)** → Check if PCEM-GMM learns comparable structures **without needing PCA first**.
  - **Autoencoder Latents (32D, 64D)** → More structured than raw pixels.
- **Expected Outcome**:
  - PCEM-GMM should outperform standard GMM in **high-dimensional clustering stability**.
  - PCA-GMM may **perform comparably if PCA is well-tuned**, but PCEM-GMM should **adapt dynamically**.

🔗 Dataset: [MNIST](http://yann.lecun.com/exdb/mnist/)

---

#### **2.2.4 High-Dimensional Financial Data (Stock Correlations)**
- **Purpose**: Validate PCEM-GMM on **real-world continuous Gaussian-like data**.
- **Details**:
  - Use **daily returns** of **S&P 500 stocks** (500D feature space).
  - Fit GMMs to cluster **correlated stocks**.
- **Why it's good**:
  - Stock return data has **correlated Gaussian-like structure**.
  - Standard GMM **overfits and is numerically unstable**.
- **Expected Outcome**:
  - PCEM-GMM should **automatically learn lower-rank covariance models**, clustering correlated stocks efficiently.

🔗 Dataset: [Yahoo Finance API](https://pypi.org/project/yfinance/)

---

#### **2.2.5 Gene Expression Data (RNA-Seq)**
- **Purpose**: Test PCEM-GMM in **biomedical high-dimensional clustering**.
- **Details**:
  - Data is often **highly structured, Gaussian-like, and sparse**.
  - Standard GMMs fail due to **extreme dimensionality (10,000+ genes)**.
- **Experiment**:
  - Compare PCEM-GMM against:
    - **Standard GMM** (full covariance).
    - **Sparse GMMs** (e.g., HDDC).
    - **PCA-GMM**.
- **Expected Outcome**:
  - PCEM-GMM should provide a **balance between full covariance models and low-rank PCA-GMMs**.

🔗 Dataset: [TCGA Gene Expression](https://www.cancer.gov/tcga)

---

## **3. Final Experiments: Comparisons & Robustness Tests**
### **3.1 Benchmark Against Other Methods**
We compare **PCEM-GMM vs. Standard GMM vs. PCA-GMM vs. Mixtures of PPCA** on:
- **Synthetic 50D Gaussians**
- **EMNIST & MNIST (high-dimensional clustering)**
- **Stock Market Data**
- **Gene Expression Data**

### **3.2 Hyperparameter Sensitivity Analysis**
- **Power Iteration Convergence Tolerance**
- **Stopping Criterion for Principal Component Extraction**
- **Effect of Sample Size on Estimation Stability**

### **3.3 Computational Efficiency Benchmarks**
- **Compare runtime** of:
  - Standard GMM
  - PCA-GMM
  - PCEM-GMM
- **Test on CPU vs. GPU performance.**

---

## **Conclusion & Next Steps**
**This experiment plan allows us to:**
1. **Validate PCEM-GMM** with simple synthetic data.
2. **Compare against existing GMM variants** in well-known **real-world datasets**.
3. **Highlight cases where standard methods fail** due to **numerical instability**.
4. **Demonstrate practical applications** in finance, genomics, and image processing.

