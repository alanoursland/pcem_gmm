## **Outline for PCEM-GMM Documentation**

### **1. Introduction**
   - Brief overview of **Gaussian Mixture Models (GMMs)**.
   - Importance of GMMs in **density estimation, clustering, and generative modeling**.
   - Challenges of applying **GMMs in high-dimensional spaces**.
   - Our contribution: **Principal Component Expectation-Maximization (PCEM-GMM)**.

---

### **2. Background and Theory**
   - **2.1 Gaussian Mixture Models (GMMs)**
     - Definition of a **GMM** as a weighted sum of multivariate Gaussians.
     - Explanation of **GMM parameters**: means, covariances, and mixture weights.
     - Mathematical formulation of **GMM probability density function**.

   - **2.2 Expectation-Maximization (EM) for GMMs**
     - Role of **Expectation (E-step)**: Computing soft assignments of points to Gaussian components.
     - Role of **Maximization (M-step)**: Updating means, covariances, and mixture weights.
     - Derivation of the standard **M-step covariance update**.

   - **2.3 Problems with GMMs in High Dimensions**
     - Curse of Dimensionality: Covariance estimation becomes **unstable** when dimensions are high.
     - **Overfitting**: Full-rank covariance matrices require \( O(d^2) \) parameters per Gaussian.
     - **Singular Covariances**: If the number of data points is smaller than the dimensionality, covariance matrices become **ill-conditioned**.
     - **Dimensionality Reduction as a Standard Fix**:
       - **PCA + GMM**: Reduce dimensionality before applying GMM.
       - **VAE + GMM**: Learn latent embeddings for GMM clustering.
       - **T-SNE + GMM**: Reduce to 2D or 3D for visualization.

---

### **3. Principal Component Expectation-Maximization (PCEM-GMM)**
   - **3.1 Motivation**
     - Instead of reducing dimensionality before GMM, we propose **modifying the EM algorithm itself**.
     - PCEM-GMM **iteratively constructs the covariance structure** rather than estimating it in full.

   - **3.2 The PCEM-GMM Algorithm**
     - **E-Step**: Identical to standard GMM (compute responsibilities using Mahalanobis distance).
     - **M-Step Modification**:
       - **Step 1: Compute Component Means** \( \mu_k \).
       - **Step 2: Iteratively Extract Principal Components**:
         - Find the **direction of greatest variance** using **power iteration**.
         - Compute the **corresponding eigenvalue (variance)**.
         - **Deflate** the data along this direction.
         - Repeat until the full covariance structure is extracted.
       - Store covariance as **a set of principal components and eigenvalues** rather than a full matrix.

   - **3.3 Advantages of PCEM-GMM**
     - **Numerical Stability**: No need to invert full covariance matrices.
     - **Memory Efficiency**: Store only **rank-reduced covariance representations**.
     - **High-Dimensional Scalability**: Works better in datasets where \( d > N \).
     - **Interpretable Components**: Learns **principal variance directions** instead of raw covariance estimates.

---

### **4. Mathematical Formulation**
   - **4.1 Standard EM Formulation**
     - Likelihood function for GMMs.
     - Derivation of **E-step and M-step updates**.

   - **4.2 Modified M-Step in PCEM-GMM**
     - Step-by-step derivation of **power iteration** for dominant eigenvector extraction.
     - Proof that this method **recovers full covariance structure** after enough iterations.
     - Computational complexity analysis.

---

### **5. Experimental Validation**
   - **5.1 Comparison with Standard GMM**
     - Dataset: Synthetic Gaussian mixtures in **low and high dimensions**.
     - Evaluation: **Log-likelihood, clustering accuracy, runtime efficiency**.

   - **5.2 PCEM-GMM vs. PCA-GMM**
     - Compare **applying PCA before GMM** vs. **learning the covariance incrementally**.
     - Show how PCEM-GMM retains **more variance information**.

   - **5.3 Real-World Applications**
     - High-dimensional datasets where standard GMMs fail.

---

### **6. Discussion and Future Work**
   - **Limitations of PCEM-GMM**
     - What happens when the number of PCs extracted is **too low or too high**?
     - Edge cases: Highly **non-Gaussian data**.
   
   - **Potential Extensions**
     - **Regularization techniques** for PC extraction.
     - **Non-linear PC extraction** using kernel methods.
     - Applications to **other mixture models** (e.g., Student-t Mixture Models).

---

### **7. Conclusion**
   - Summary of contributions.
   - Why PCEM-GMM is a useful alternative to traditional GMMs.
   - Open questions for further research.

---

### **8. References**
   - Include relevant academic papers on:
     - Expectation-Maximization (EM).
     - Gaussian Mixture Models (GMMs).
     - Power Iteration and Principal Component Analysis.
     - High-dimensional clustering techniques.

