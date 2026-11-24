# FIGURE PLACEMENT GUIDE: REGRESSION (Q2) AND K-MEANS CLUSTERING (Q3(a))

This document provides explicit insertion instructions, filenames, captions, and analytical explanations for all figures associated with Question 2 (Supervised Regression) and Question 3(a) (Unsupervised K-Means Clustering).

---

## 1. Question 2 – Supervised Regression (Graduate Admissions Prediction)

### Figure 2.1: Gradient Descent Optimization Dynamics
- **Filename**: `regression_gd_mse_vs_epochs.png`
- **Path**: `reports/regression_kmeans/figures/regression/regression_gd_mse_vs_epochs.png`
- **Report Section**: Question 2, Section D ("Implementation and Training Dynamics").
- **Insertion Placement**: Immediately after the explanation of the custom `gradient_descent()` function and parameter convergence.
- **Suggested Caption**: *Figure 2.1. Batch Gradient Descent Training Loss ($J_{\text{MSE}}$) vs. Optimization Epochs.*
- **Analytical Explanation**: Illustrates the monotonic decline of Mean Squared Error across 5,000 optimization epochs with learning rate $\alpha = 0.05$. The rapid descent in the initial 200 epochs followed by asymptotic stabilization confirms robust convergence to the global convex minimum without oscillations or gradient explosion.

---

### Figure 2.2: Linear Regression Actual vs. Predicted
- **Filename**: `regression_linear_actual_vs_predicted.png`
- **Path**: `reports/regression_kmeans/figures/regression/regression_linear_actual_vs_predicted.png`
- **Report Section**: Question 2, Section E ("Experimental Results - Linear Model Evaluation").
- **Insertion Placement**: Directly following the presentation of the Linear Regression test metrics (MSE: 0.00462, RMSE: 0.06795, $R^2$: 0.82121).
- **Suggested Caption**: *Figure 2.2. Actual vs. Predicted Chance of Admit for Linear Regression (Batch Gradient Descent) on Held-Out Test Set ($N_{\text{test}}=80$).*
- **Analytical Explanation**: Plots predicted admission probability against actual applicant ground truth along the dashed identity reference line ($y = x$). Observations cluster tightly around the diagonal across the $0.5$ to $0.95$ probability range, validating that linear regression captures the primary predictive signal.

---

### Figure 2.3: Linear Regression Residual Analysis
- **Filename**: `regression_linear_residuals.png`
- **Path**: `reports/regression_kmeans/figures/regression/regression_linear_residuals.png`
- **Report Section**: Question 2, Section E ("Residual Analysis").
- **Insertion Placement**: Immediately after Figure 2.2.
- **Suggested Caption**: *Figure 2.3. Residuals vs. Fitted Values for Linear Regression (Gradient Descent).*
- **Analytical Explanation**: Evaluates the homoscedasticity assumption by plotting residual errors ($e_i = y_i - \hat{y}_i$) against predicted values. Residuals are roughly centered symmetrically around zero with no severe heteroscedastic funneling, though minor negative residuals at the lower tail hint at slight non-linearity.

---

### Figure 2.4: Polynomial Degrees Training vs. Testing Error
- **Filename**: `regression_polynomial_degrees_error.png`
- **Path**: `reports/regression_kmeans/figures/regression/regression_polynomial_degrees_error.png`
- **Report Section**: Question 2, Section E ("Polynomial Expansion and Bias-Variance Trade-off").
- **Insertion Placement**: Directly following the Polynomial Degrees Benchmark table (Table 2.1).
- **Suggested Caption**: *Figure 2.4. Training MSE and Testing MSE Across Polynomial Expansion Degrees ($d \in \{1, 2, 3, 4\}$).*
- **Analytical Explanation**: Demonstrates the classic bias-variance dilemma. As polynomial degree increases from 1 to 2, test error reaches its minimum ($0.00451$). Beyond degree 2, training error continues to plummet toward zero ($0.00021$ at $d=4$) while test error diverges sharply ($0.87718$ at $d=4$), illustrating catastrophic overfitting.

---

### Figure 2.5: Polynomial Degrees 1–4 Prediction Scatter Grid
- **Filename**: `regression_polynomial_degrees_scatter_grid.png`
- **Path**: `reports/regression_kmeans/figures/regression/regression_polynomial_degrees_scatter_grid.png`
- **Report Section**: Question 2, Section E ("Polynomial Overfitting Visual Assessment").
- **Insertion Placement**: Immediately following Figure 2.4.
- **Suggested Caption**: *Figure 2.5. Actual vs. Predicted Response Grids Across Polynomial Degrees 1 through 4 on Held-Out Test Data.*
- **Analytical Explanation**: Compares the 4 test scatter distributions against the ideal $y=x$ red line. Subplots for Degree 1 and Degree 2 display compact variance along the line. Degree 3 displays increased point dispersion, while Degree 4 demonstrates wild predictive oscillations with extreme negative and outlier predictions exceeding 2.0, providing visual confirmation of Runge's phenomenon.

---

### Figure 2.6: Degree-2 Polynomial (Optimal Model) Fit and Residuals
- **Filename**: `regression_poly2_actual_vs_predicted.png` and `regression_poly2_residuals.png`
- **Path**: 
  - `reports/regression_kmeans/figures/regression/regression_poly2_actual_vs_predicted.png`
  - `reports/regression_kmeans/figures/regression/regression_poly2_residuals.png`
- **Report Section**: Question 2, Section F ("Discussion and Best Model Selection").
- **Insertion Placement**: Inside the discussion subsection establishing Degree 2 as the optimal regression model.
- **Suggested Caption**: *Figure 2.6. Prediction Concordance and Residual Distribution for the Top-Performing Degree-2 Polynomial Estimator ($R^2 = 0.8254, \text{RMSE} = 0.0672$).*
- **Analytical Explanation**: Shows that the degree-2 polynomial model achieves the tightest residual distribution around zero, effectively modeling non-linear synergisms between CGPA and GRE scores while avoiding boundary volatility.

---

### Figure 2.7: Ridge and Lasso Cross-Validation Curves vs. $\alpha$
- **Filename**: `regression_ridge_lasso_cv_alpha.png`
- **Path**: `reports/regression_kmeans/figures/regression/regression_ridge_lasso_cv_alpha.png`
- **Report Section**: Question 2, Section E ("Regularization Analysis (Ridge vs. Lasso)").
- **Insertion Placement**: Immediately following the discussion of 5-fold cross-validation grid search across regularization penalty $\alpha \in [10^{-4}, 10^2]$.
- **Suggested Caption**: *Figure 2.7. 5-Fold Cross-Validation Mean Squared Error vs. Regularization Hyperparameter $\alpha$ for Ridge ($L_2$) and Lasso ($L_1$).*
- **Analytical Explanation**: Plots validation error curves on a logarithmic $\alpha$ scale. Both estimators exhibit an optimal error plateau for small $\alpha$ values ($\alpha < 0.1$), with MSE escalating rapidly once $\alpha > 1.0$ as excessive penalty drives weights toward zero, causing underfitting.

---

### Figure 2.8: Comprehensive Model Performance Comparison (R² and MSE)
- **Filename**: `regression_model_comparison_r2.png` and `regression_model_comparison_train_test_mse.png`
- **Path**:
  - `reports/regression_kmeans/figures/regression/regression_model_comparison_r2.png`
  - `reports/regression_kmeans/figures/regression/regression_model_comparison_train_test_mse.png`
- **Report Section**: Question 2, Section E ("Consolidated Benchmark Comparison").
- **Insertion Placement**: Accompanying the master regression comparison table.
- **Suggested Caption**: *Figure 2.8. Comparative Evaluation of All Six Regression Models Across Coefficient of Determination ($R^2$) and Generalization Gap (Train vs. Test MSE).*
- **Analytical Explanation**: Visually highlights the contrast between stable models ($R^2 \approx 0.81 - 0.83$) and the catastrophic failure of Degree 4 ($R^2 = -32.97$).

---

## 2. Question 3(a) – Unsupervised K-Means Clustering (Bike Sharing Telemetry)

### Figure 3.1: K-Means Elbow Method Curve
- **Filename**: `kmeans_elbow_curve.png`
- **Path**: `reports/regression_kmeans/figures/kmeans/kmeans_elbow_curve.png`
- **Report Section**: Question 3(a), Section E ("Elbow Method and Selection of Optimal K").
- **Insertion Placement**: Immediately following Table 3.1 (WCSS Inertia across $K \in \{2, 3, 4, 5, 6\}$).
- **Suggested Caption**: *Figure 3.1. Within-Cluster Sum of Squares (WCSS) vs. Number of Clusters $K$ on Standardized Bike Sharing Telemetry ($N=17,379$).*
- **Analytical Explanation**: Plots inertia over $K \in \{2, 3, 4, 5, 6\}$. Sharp marginal reductions occur from $K=2$ ($65,562.23$) through $K=3$ ($55,764.07$) to $K=4$ ($49,113.85$), after which the curve levels off ($K=5, 45,186.36$), confirming $K=4$ as the optimal mathematical elbow.

---

### Figure 3.2: K-Means (K=4) Bivariate Cluster Distribution
- **Filename**: `kmeans_clusters_scatter_temp_hum.png`
- **Path**: `reports/regression_kmeans/figures/kmeans/kmeans_clusters_scatter_temp_hum.png`
- **Report Section**: Question 3(a), Section E ("Cluster Visualizations and Geometric Partitions").
- **Insertion Placement**: Immediately following the elbow analysis.
- **Suggested Caption**: *Figure 3.2. Two-Dimensional Scatter Representation of the $K=4$ Cluster Assignments in Standardized Temperature vs. Humidity Space.*
- **Analytical Explanation**: Illustrates the geometric clustering of the 17,379 observations in the temperature and humidity feature plane, displaying clear separation between warm/dry periods, cold/humid periods, and transitional weather.

---

### Figure 3.3: K-Means (K=4) Cluster Centroid Profiles
- **Filename**: `kmeans_centroid_profiles.png`
- **Path**: `reports/regression_kmeans/figures/kmeans/kmeans_centroid_profiles.png`
- **Report Section**: Question 3(a), Section E ("Cluster Centroid Characteristics and Feature Profiles").
- **Insertion Placement**: Immediately following the centroid summary table (Table 3.2).
- **Suggested Caption**: *Figure 3.3. Standardized Feature Coordinates (Z-Scores) Across the Four Optimal Cluster Centroids ($K=4$).*
- **Analytical Explanation**: Compares the multidimensional centroid coordinates (`hour_sin`, `hour_cos`, `temp`, `hum`, `windspeed`). Clearly distinguishes Cluster 1 as warm and dry afternoon travel, Cluster 2 as cold, dry, high-wind conditions, Cluster 3 as early morning humid commute, and Cluster 0 as midday humid travel.

---

### Figure 3.4: K-Means (K=4) Clusters with Marked Centroids
- **Filename**: `kmeans_clusters_with_centroids.png`
- **Path**: `reports/regression_kmeans/figures/kmeans/kmeans_clusters_with_centroids.png`
- **Report Section**: Question 3(a), Section F ("Discussion and Verification").
- **Insertion Placement**: Inside the discussion section illustrating convergence properties of the custom NumPy implementation.
- **Suggested Caption**: *Figure 3.4. High-Resolution Scatter Plot of Final $K=4$ Clusters with Vectorized Mathematical Centroids Overlaid as Bold Markers.*
- **Analytical Explanation**: Overlays the converged cluster centers of mass (marked as bold black crosses) directly on the clustered data scatter, verifying that the centroids coincide with the geometric center of each partition.

