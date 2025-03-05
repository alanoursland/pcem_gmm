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

### **1.4 Single High-Dimensional Gaussian with Thresholding for Principal Component Extraction**

- **Purpose**:  
  - Test PCEM-GMM on high-dimensional (e.g., 50D, 100D, or 1000D) synthetic Gaussian data to evaluate its ability to iteratively extract principal components and effectively manage dimensionality reduction. The goal is to assess whether the model can handle increasing dimensions while maintaining computational efficiency.
  - Introduce a **thresholding parameter** that stops the extraction of additional principal components when their contribution to the total variance drops below a predefined threshold. This allows for early termination of the extraction process, improving both computational efficiency and model performance.
  - Introduce a **spherical background variance parameter** to account for dropped principal components, modeling the residual variance when components with small contributions are excluded. This helps to preserve important variance from the original high-dimensional Gaussian while maintaining numerical stability.

- **Design**:
  - **Data Generation**: Generate synthetic Gaussian data in varying dimensions: 10D, 50D, 100D, and 1000D. Each dataset will have a mean vector and eigenvalues assigned to the Gaussian distribution, simulating the true variance structure.
  - **Threshold Experimentation**:  
    - Implement **direct thresholding** for individual eigenvalue contributions (e.g., stop extracting components when the eigenvalue is below a certain value).
    - Use a **percentage thresholding** based on the cumulative eigenvalue sum (e.g., stop when the cumulative variance explained exceeds 95%).
  - **Sample Size Variations**: Experiment with different sample counts (e.g., 1000, 5000, 10000) to evaluate how sample size affects the accuracy and convergence speed of the algorithm, particularly in higher dimensions.
  - **Batch Training**: Test batch-wise training with smaller subsets of data, comparing its impact on convergence speed and memory efficiency.
  - **Baseline Comparison**: Compare PCEM-GMM to standard **GMM** (Gaussian Mixture Model with full covariance estimation) and **PCA-GMM** (PCA-based dimensionality reduction followed by GMM fitting) for performance in terms of:
    - Eigenvalue recovery
    - Log-likelihood
    - Convergence speed

- **Metrics**:
  - **Parameter Estimation Accuracy**:  
    - **Eigenvalue Recovery**: Compare estimated eigenvalues to ground truth to evaluate the accuracy of the principal component extraction.
    - **Eigenvector Estimation**: Compare the estimated principal eigenvectors to the true eigenvectors. This will measure how well the model recovers the directions of maximum variance.
  - **Calculation Time**: Measure the time taken for model fitting with different dimensionalities and thresholds. Track time-to-convergence, as well as time spent per iteration, to evaluate the scalability of the approach.
  - **Convergence Behavior**:  
    - Track the number of iterations needed for convergence with different thresholds and sample sizes.
    - Analyze the log-likelihood change per iteration to monitor whether the algorithm converges to a stable solution.
  - **Threshold Sensitivity**:  
    - Analyze the impact of varying threshold values (for both direct and percentage-based thresholds) on the final model parameters and computational efficiency.
    - Measure how different stopping criteria influence the accuracy of recovered eigenvalues and eigenvectors compared to the true values.
  - **Variance Capture**:  
    - Calculate the proportion of total variance explained by the selected components for each thresholding method. Evaluate how well the model captures the majority of the variance when fewer components are retained.
    - Compare how much variance is retained when spherical background variance is used, to see how much residual variance is captured and its effect on the model's stability.
  - **Sample Size and Dimensionality Sensitivity**:  
    - Evaluate how different sample sizes influence convergence accuracy, particularly as dimensionality increases.
    - Assess the relationship between sample size and accuracy of the eigenvalue recovery.

- **Expected Outcome**:  
  - **Effective Component Extraction**: PCEM-GMM should successfully extract the most significant components (those with the largest eigenvalues) and stop the extraction process when the remaining components contribute negligibly to the overall variance. The thresholding mechanism should help the algorithm converge faster in higher-dimensional spaces without losing significant variance.
  - **Stable and Efficient Performance**: The model should remain numerically stable even in high-dimensional settings (e.g., 1000D), while demonstrating improved computational efficiency with thresholding. The spherical background variance parameter will allow the model to handle residual variance effectively without overfitting.
  - **Comparison to Baseline Methods**:  
    - PCEM-GMM should outperform standard GMM in terms of computational efficiency in high-dimensional spaces, as it avoids full covariance matrix estimation.  
    - It should perform comparably or better than PCA-GMM by retaining more variance in the principal components dynamically (without requiring pre-dimensionality reduction).  
    - The threshold-based stopping criterion should strike a balance between model accuracy and computational speed, particularly in terms of convergence time and variance capture.

---

### **1.5 Higher-Dimensional Synthetic Gaussians (10D, 50D)**
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

