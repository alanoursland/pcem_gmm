Does PCEM-GMM give a big enough improvement over PCA-GMM to justify the extra computation?

Does power iteration still work well when sample size N is small compared to dimension d?

Does PCEM-GMM handle covariance estimation well when multiple clusters exist?

When to stop based on eigenvalue drop-off?

Which deflation method is better?
```
    projection_matrix = torch.eye(v.shape[0], device=device) - torch.outer(v, v)
    centered_data = centered_data @ projection_matrix  # Project onto hyperplane
```
```
    projection = (centered_data @ v).unsqueeze(1) * v.unsqueeze(0)
    centered_data -= projection  # Deflate the data
```
