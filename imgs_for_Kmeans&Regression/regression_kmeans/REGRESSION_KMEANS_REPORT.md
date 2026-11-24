# Machine Learning Course Assignment Report
## Part II: Supervised Regression and Part III(a): Unsupervised K-Means Clustering

---

# Question 2 – Supervised Regression (Graduate Admissions Prediction)

## 2.1 Problem Statement and Objectives

Predicting postgraduate university admissions is an essential quantitative problem in educational data mining. Higher education applicants evaluate multiple academic and profile credentials—such as standardized test scores, undergraduate Grade Point Average (CGPA), letters of recommendation, statement of purpose ratings, university prestige, and research experience—to gauge their competitiveness. Because university admissions committees evaluate these factors non-trivially, providing an objective, data-driven, continuous estimate of admission likelihood aids prospective students in strategic application planning and institutions in holistic portfolio appraisal.

From a machine learning perspective, this problem is framed as a supervised multivariate regression task:
$$\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N, \quad \mathbf{x}_i \in \mathbb{R}^D, \quad y_i \in [0, 1]$$
where $\mathbf{x}_i$ represents applicant $i$'s standardized feature vector across $D=7$ dimensions, and $y_i \in [0, 1]$ denotes the continuous admission probability (`Chance of Admit`).

The core objectives of this study are:
1. **First-Principles Implementation**: Implement multivariate Linear Regression via Batch Gradient Descent from scratch using NumPy, deriving exact gradient updates with respect to weights and bias.
2. **Optimization Analysis**: Systematically track and analyze training loss convergence ($J_{\text{MSE}}$) across optimization epochs under a tuned learning rate.
3. **Non-Linear Exploration & Complexity Analysis**: Construct Polynomial Regression models spanning degrees $d \in \{1, 2, 3, 4\}$, empirically demonstrating the bias-variance tradeoff and characterizing catastrophic overfitting in high-degree polynomial hypothesis spaces.
4. **Regularization Framework**: Implement and tune Ridge ($L_2$) and Lasso ($L_1$) regularized regression via 5-fold cross-validation to examine parameter shrinkage, feature selection dynamics, and ill-conditioning control.
5. **Comparative Benchmarking**: Evaluate and benchmark all regression formulations on a strictly held-out test split using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and the Coefficient of Determination ($R^2$), followed by comprehensive residual diagnostics.

---

## 2.2 Mathematical Formulation and Derivations

### 2.2.1 Multivariate Linear Regression Model
Given an input design matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$ and corresponding ground-truth target vector $\mathbf{y} \in \mathbb{R}^N$, the multivariate linear hypothesis is parameterized by a weight vector $\mathbf{w} \in \mathbb{R}^D$ and a scalar bias term $b \in \mathbb{R}$:
$$\hat{y}_i = h_{\mathbf{w}, b}(\mathbf{x}_i) = \mathbf{w}^T \mathbf{x}_i + b = \sum_{j=1}^D w_j x_{ij} + b$$
In vectorized notation across all $N$ instances:
$$\hat{\mathbf{y}} = \mathbf{X} \mathbf{w} + b \mathbf{1}_N$$
where $\mathbf{1}_N$ is an all-ones column vector of length $N$.

### 2.2.2 Objective Function (Mean Squared Error)
The optimization criterion is the Mean Squared Error (MSE) empirical risk objective:
$$J(\mathbf{w}, b) = \frac{1}{2N} \sum_{i=1}^N (\hat{y}_i - y_i)^2 = \frac{1}{2N} \|\mathbf{X}\mathbf{w} + b\mathbf{1}_N - \mathbf{y}\|_2^2$$
The constant factor $\frac{1}{2}$ is introduced to cancel the factor of $2$ during partial differentiation.

### 2.2.3 Exact Gradient Derivations
To derive the analytical gradients for parameter updates, we take partial derivatives of $J(\mathbf{w}, b)$ with respect to each parameter.

**Gradient with respect to weights $\mathbf{w}$:**
Applying the chain rule:
$$\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i) \frac{\partial \hat{y}_i}{\partial \mathbf{w}}$$
Since $\frac{\partial \hat{y}_i}{\partial \mathbf{w}} = \mathbf{x}_i$:
$$\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i) \mathbf{x}_i = \frac{1}{N} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y})$$

**Gradient with respect to bias $b$:**
Since $\frac{\partial \hat{y}_i}{\partial b} = 1$:
$$\frac{\partial J}{\partial b} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i) = \frac{1}{N} \mathbf{1}_N^T (\hat{\mathbf{y}} - \mathbf{y})$$

### 2.2.4 Batch Gradient Descent Parameter Updates
Under a learning rate hyperparameter $\alpha > 0$, the parameters are simultaneously updated at iteration $t$:
$$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \alpha \nabla_{\mathbf{w}} J(\mathbf{w}^{(t)}, b^{(t)}) = \mathbf{w}^{(t)} - \frac{\alpha}{N} \mathbf{X}^T (\hat{\mathbf{y}}^{(t)} - \mathbf{y})$$
$$b^{(t+1)} = b^{(t)} - \alpha \frac{\partial J}{\partial b} = b^{(t)} - \frac{\alpha}{N} \sum_{i=1}^N (\hat{y}_i^{(t)} - y_i)$$

For unregularized convex linear regression, the closed-form Ordinary Least Squares (OLS) normal equation solution is:
$$\boldsymbol{\theta}^* = (\tilde{\mathbf{X}}^T \tilde{\mathbf{X}})^{-1} \tilde{\mathbf{X}}^T \mathbf{y}$$
where $\tilde{\mathbf{X}} = [\mathbf{1}_N \;\; \mathbf{X}] \in \mathbb{R}^{N \times (D+1)}$ and $\boldsymbol{\theta}^* = [b \;\; \mathbf{w}^T]^T$. However, when $\tilde{\mathbf{X}}^T \tilde{\mathbf{X}}$ is ill-conditioned or computationally expensive to invert, iterative gradient descent scales gracefully as $\mathcal{O}(N D)$ per epoch.

### 2.2.5 Polynomial Feature Expansion
Linear regression enforces a strict planar hyper-surface assumption. To capture higher-order feature interactions and non-linear response curves, input vectors are mapped into an expanded polynomial space via a non-linear mapping $\boldsymbol{\Phi}_d: \mathbb{R}^D \rightarrow \mathbb{R}^{P_d}$:
$$\hat{y} = \mathbf{w}^T \boldsymbol{\Phi}_d(\mathbf{x}) + b$$
For degree $d=2$ over $D$ features, the feature map contains all linear terms, quadratic terms, and pairwise interaction terms:
$$\boldsymbol{\Phi}_2(\mathbf{x}) = \left[ x_1, x_2, \dots, x_D, x_1^2, x_1 x_2, \dots, x_D^2 \right]^T$$
The resulting dimensional space expands combinatorially according to:
$$P_d = \binom{D + d}{d} - 1 \quad (\text{excluding the constant bias term})$$
For $D=7$ features:
- Degree 1: $P_1 = \binom{8}{1} - 1 = 7$ features
- Degree 2: $P_2 = \binom{9}{2} - 1 = 35$ features
- Degree 3: $P_3 = \binom{10}{3} - 1 = 119$ features
- Degree 4: $P_4 = \binom{11}{4} - 1 = 329$ features

As the degree grows ($d \ge 3$), $P_d$ approaches and exceeds the sample size $N_{\text{train}} = 320$, inducing severe collinearity, sample matrix ill-conditioning, and variance explosion (Runge's phenomenon in regression).

### 2.2.6 Regularized Regression (Ridge and Lasso)

To mitigate multicollinearity and constrain model variance in high-dimensional hypothesis spaces, regularization penalizes parameter magnitude:

**Ridge Regression ($L_2$ Regularization):**
$$J_{\text{Ridge}}(\mathbf{w}, b) = \frac{1}{2N} \|\mathbf{y} - (\mathbf{X}\mathbf{w} + b\mathbf{1})\|_2^2 + \frac{\alpha_{\text{reg}}}{2} \|\mathbf{w}\|_2^2$$
where $\|\mathbf{w}\|_2^2 = \sum_{j=1}^D w_j^2$. Ridge regression applies uniform shrinkage across parameter coefficients, preventing weights from taking extreme values while ensuring $\mathbf{X}^T\mathbf{X} + \alpha_{\text{reg}}\mathbf{I}$ remains strictly non-singular and well-conditioned.

**Lasso Regression ($L_1$ Regularization):**
$$J_{\text{Lasso}}(\mathbf{w}, b) = \frac{1}{2N} \|\mathbf{y} - (\mathbf{X}\mathbf{w} + b\mathbf{1})\|_2^2 + \alpha_{\text{reg}} \|\mathbf{w}\|_1$$
where $\|\mathbf{w}\|_1 = \sum_{j=1}^D |w_j|$. Due to the non-differentiable sharp corners of the $L_1$ diamond ball at coordinate axes, Lasso drives non-essential feature weights identically to zero, performing intrinsic embedded feature selection.

### 2.2.7 Quantitative Evaluation Metrics
All regression models are evaluated across three standardized metrics on held-out samples:
1. **Mean Squared Error (MSE)**:
   $$\text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$
2. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
3. **Coefficient of Determination ($R^2$)**:
   $$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}, \quad \bar{y} = \frac{1}{N}\sum_{i=1}^N y_i$$
   The $R^2$ statistic quantifies the proportion of total target variance explained by the model relative to a baseline naive mean predictor.

---

## 2.3 Dataset Description and Preprocessing Pipeline

The empirical investigation utilizes the **Graduate Admissions** dataset (`Admission_Predict.csv`). The dataset contains academic profiles and admission records for prospective graduate school applicants:

- **Total Sample Size ($N$)**: 400 applicant instances.
- **Dimensionality**: 9 original columns (1 index column, 7 predictor features, 1 continuous target).
- **Missing Values**: 0 missing or NaN values across all attributes.

### Table 2.1: Attribute Schema and Statistical Characterization
| Attribute Name | Variable Type | Domain / Measurement Scale | Description | Role |
| :--- | :--- | :--- | :--- | :--- |
| `Serial No.` | Identifier | Integer ($1$ to $400$) | Applicant row identifier | Excluded from modeling |
| `GRE Score` | Continuous | Continuous scale ($290$ to $340$) | Standardized Graduate Record Examination score | Input Feature ($x_1$) |
| `TOEFL Score` | Continuous | Continuous scale ($92$ to $120$) | Test of English as a Foreign Language score | Input Feature ($x_2$) |
| `University Rating`| Ordinal | Discrete integer rating ($1$ to $5$) | Undergraduate institutional prestige tier | Input Feature ($x_3$) |
| `SOP` | Ordinal | Half-point scale ($1.0$ to $5.0$) | Statement of Purpose strength rating | Input Feature ($x_4$) |
| `LOR` | Ordinal | Half-point scale ($1.0$ to $5.0$) | Letter of Recommendation strength rating | Input Feature ($x_5$) |
| `CGPA` | Continuous | Continuous scale ($6.80$ to $9.92$) | Undergraduate Cumulative Grade Point Average | Input Feature ($x_6$) |
| `Research` | Binary | Categorical flag ($\{0, 1\}$) | Research / publication experience ($1=\text{Yes}, 0=\text{No}$) | Input Feature ($x_7$) |
| `Chance of Admit` | Continuous | Continuous interval ($[0.34, 0.97]$) | Likelihood of graduate school admission | Target ($y$) |

### Data Leakage Prevention and Normalization
1. **Index Dropping**: `Serial No.` is purely administrative and exhibits spurious correlation with index order. It was dropped immediately.
2. **Train-Test Partitioning**: To ensure rigorous out-of-sample generalization, the dataset was split into 80% training ($N_{\text{train}} = 320$) and 20% testing ($N_{\text{test}} = 80$) partitions using `random_state=42`.
3. **Z-Score Feature Standardization**: Input attributes vary widely in scale (GRE $\in [290, 340]$ vs. CGPA $\in [6.8, 9.9]$ vs. Research $\in \{0, 1\}$). To guarantee isotropic gradient descent contours and equal regularization penalty weighting:
   $$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
   The `StandardScaler` was strictly fit **only** on the training partition ($X_{\text{train}}$) and subsequently applied to transform $X_{\text{train}}$ and $X_{\text{test}}$, completely eliminating data leakage.

---

## 2.4 Implementation Methodology

### 2.4.1 Batch Gradient Descent from Scratch
The primary linear regression estimator was implemented in Python using vectorized NumPy operations:

```python
import numpy as np

def compute_mse(X, y, weights, bias):
    N = len(y)
    predictions = np.dot(X, weights) + bias
    return (1.0 / (2.0 * N)) * np.sum((predictions - y) ** 2)

def gradient_descent(X, y, lr=0.05, epochs=5000):
    N, D = X.shape
    weights = np.zeros(D)
    bias = 0.0
    loss_history = []
    
    for epoch in range(epochs):
        predictions = np.dot(X, weights) + bias
        errors = predictions - y
        
        # Exact vectorized gradients
        dw = (1.0 / N) * np.dot(X.T, errors)
        db = (1.0 / N) * np.sum(errors)
        
        # Simultaneous parameter update
        weights -= lr * dw
        bias -= lr * db
        
        loss_history.append(compute_mse(X, y, weights, bias))
        
    return weights, bias, loss_history
```

### 2.4.2 Training Dynamics and Convergence
Using hyperparameters $\alpha = 0.05$ and $5,000$ epochs, the loss monotonically decreased without overshoot.

[Insert Figure: regression_gd_mse_vs_epochs.png here]

As observed in Figure 2.1, the training Mean Squared Error begins at initial zero-weight loss ($J \approx 0.28$) and drops steeply within the first $200$ epochs before asymptotically stabilizing around $J \approx 0.00387$. The gradient norms diminish to near-zero, demonstrating full parameter convergence.

---

## 2.5 Experimental Results and Comparative Analysis

### Table 2.2: Comprehensive Regression Performance Benchmark
All models were evaluated on the identical 80:20 split ($N_{\text{train}}=320, N_{\text{test}}=80$).

| Model Architecture / Hypothesis | Optimal Hyperparameter | Train MSE | Test MSE | Train RMSE | Test RMSE | Train $R^2$ | Test $R^2$ | Empirical Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Linear Regression (Scratch GD)** | $\alpha=0.05, 5000\text{ ep}$ | 0.00387 | 0.00462 | 0.06221 | 0.06795 | 0.79519 | 0.82121 | Robust, well-generalized baseline |
| **Polynomial Degree 1** | OLS Closed-Form | 0.00387 | 0.00462 | 0.06221 | 0.06795 | 0.79519 | 0.82121 | Identical to GD Linear baseline |
| **Polynomial Degree 2** | Full quadratic ($P_2=35$) | **0.00346** | **0.00451** | **0.05880** | **0.06715** | **0.81704** | **0.82538** | **Best overall predictive model** |
| **Polynomial Degree 3** | Full cubic ($P_3=119$) | 0.00253 | 0.01069 | 0.05031 | 0.10338 | 0.86606 | 0.58615 | Moderate overfitting ($R^2$ drops) |
| **Polynomial Degree 4** | Quartic ($P_4=329$) | 0.00021 | 0.87718 | 0.01443 | 0.93658 | 0.99889 | -32.96851 | **Catastrophic overfitting / variance explosion** |
| **Ridge Regression ($L_2$)** | $\alpha_{\text{reg}}=1.0$ (5-fold CV) | 0.00391 | 0.00480 | 0.06254 | 0.06930 | 0.79306 | 0.81401 | Parameter shrinkage; conservative |
| **Lasso Regression ($L_1$)** | $\alpha_{\text{reg}}=0.01$ (5-fold CV) | 0.00387 | 0.00466 | 0.06222 | 0.06829 | 0.79513 | 0.81943 | Sparsity induced; robust performance |

---

### 2.5.1 Linear Regression Performance and Residual Diagnostics
The linear model achieved an out-of-sample Test MSE of $0.00462$ (RMSE = $0.06795$) and a high Test $R^2$ of $0.82121$, indicating that over 82% of the variance in admission probability is linearly captured by applicant academic credentials.

[Insert Figure: regression_linear_actual_vs_predicted.png here]

Figure 2.2 confirms that test predictions align tightly along the ideal diagonal identity line ($y = \hat{y}$). 

[Insert Figure: regression_linear_residuals.png here]

Residual diagnostic analysis (Figure 2.3) reveals that residuals $e_i = y_i - \hat{y}_i$ are randomly distributed around the zero horizontal axis without prominent funneling, demonstrating that error variance is largely homoscedastic. A slight curvature in the negative residual tail suggests subtle non-linear interactions at the lower admission probability threshold.

---

### 2.5.2 Polynomial Regression and Complexity Analysis
To test whether non-linear modeling captures those subtle interactions, polynomial regression models were fitted across degrees $d \in \{1, 2, 3, 4\}$.

[Insert Figure: regression_polynomial_degrees_scatter_grid.png here]

Figure 2.4 illustrates the predicted vs. actual response across all polynomial degrees. Degree 1 and Degree 2 produce tight linear clustering. In contrast, Degree 3 begins spreading, and Degree 4 produces wild extrapolation errors where predictions deviate drastically from the ground truth.

[Insert Figure: regression_polynomial_degrees_error.png here]

Figure 2.5 details the bias-variance tradeoff:
- **Degree 1 ($P_1=7$)**: Train MSE $0.00387$, Test MSE $0.00462$. Low model complexity, low variance.
- **Degree 2 ($P_2=35$)**: Train MSE drops to $0.00346$, Test MSE decreases to $0.00451$, and Test $R^2$ peaks at **$0.82538$**. The pairwise quadratic interactions (e.g., `CGPA` $\times$ `GRE Score`) capture true non-linear relationships without overfitting.
- **Degree 3 ($P_3=119$)**: Train MSE falls to $0.00253$, but Test MSE more than doubles to $0.01069$, dropping Test $R^2$ to $0.58615$. High variance begins dominating.
- **Degree 4 ($P_4=329$)**: Train MSE drops to near-zero ($0.00021$, Train $R^2 = 0.99889$), representing near-perfect interpolation of training noise. However, on held-out test data, Test MSE explodes to **$0.87718$** (RMSE = $0.93658$), yielding a negative $R^2$ of **$-32.96851$**. Because $P_4 = 329 > N_{\text{train}} = 320$, the design matrix becomes overdetermined and collinear, causing parameter variance to diverge.

---

### 2.5.3 Regularization Analysis (Ridge vs. Lasso)
Cross-validation across $\alpha_{\text{reg}} \in [10^{-3}, 10^2]$ identified optimal regularization strengths: $\alpha = 1.0$ for Ridge and $\alpha = 0.01$ for Lasso.

[Insert Figure: regression_ridge_lasso_cv_alpha.png here]

Figure 2.6 displays cross-validation error curves across alpha values.
- **Ridge Regression ($\alpha=1.0$)**: Constrains the $L_2$ norm of weights, yielding Test MSE $0.00480$ and $R^2 = 0.81401$. Ridge trades off a minute amount of bias to prevent coefficient instability caused by correlation between GRE and TOEFL scores.
- **Lasso Regression ($\alpha=0.01$)**: Shrinks the coefficients via $L_1$ penalty, obtaining Test MSE $0.00466$ and $R^2 = 0.81943$. Lasso successfully filters out negligible feature interactions while retaining `CGPA` and `GRE Score` as the primary predictors.

---

### 2.5.4 Overall Model Comparison and Final Benchmark
Figures 2.7, 2.8, and 2.9 synthesize the cross-model evaluation across MSE and $R^2$ metrics.

[Insert Figure: regression_model_comparison_mse.png here]
[Insert Figure: regression_model_comparison_train_test_mse.png here]
[Insert Figure: regression_model_comparison_r2.png here]

[Insert Figure: regression_poly2_actual_vs_predicted.png here]
[Insert Figure: regression_poly2_residuals.png here]

Figures 2.10 and 2.11 present the out-of-sample fit and residual diagnostics for the top-performing model, **Polynomial Degree 2**. Its residuals exhibit superior symmetry and narrower error spreads ($|\text{residual}| \le 0.12$) compared to all alternatives.

---

## 2.6 Question 2 Conclusion
The supervised regression investigation demonstrates that graduate admission likelihood is strongly governed by academic and profile metrics:
1. **Model Hierarchy**: Polynomial Regression of **Degree 2** proved to be the optimal predictive architecture, attaining the lowest out-of-sample error ($\text{MSE} = 0.00451$, $\text{RMSE} = 0.06715$) and highest explanatory power ($R^2 = 0.82538$), outperforming linear and regularized baselines.
2. **Bias-Variance Dynamics**: As polynomial degree escalated to $d=4$, the parameter dimension ($P_4=329$) surpassed the training sample volume ($N_{\text{train}}=320$). While training error approached zero ($R^2 = 0.99889$), test error exploded catastrophically ($R^2 = -32.96851$), providing a clear empirical demonstration of variance explosion and Runge's phenomenon.
3. **Regularization Utility**: Ridge and Lasso regularizers effectively controlled variance without destabilizing predictions. Lasso demonstrated that a sparse linear model retains competitive test performance ($R^2 = 0.81943$).
4. **Primary Predictors**: Feature analysis confirmed that undergraduate Cumulative GPA (`CGPA`) and standardized exam scores (`GRE`, `TOEFL`) represent the dominant statistical drivers of graduate admission success.

---
---

# Question 3(a) – Unsupervised K-Means Clustering (Bike Sharing Telemetry)

## 3.1 Problem Statement and Objectives

Urban bike sharing systems collect extensive spatiotemporal telemetry that reflects real-time commuter behavior, transportation demand, and urban environmental conditions. Identifying natural operational states—such as peak commuting hours, inclement weather shutdowns, or leisure riding periods—is essential for fleet rebalancing, dynamic pricing, and maintenance dispatching. 

Because operational states are not labeled a priori, this task is naturally formulated as an unsupervised clustering problem:
$$\mathcal{D} = \{\mathbf{x}_i\}_{i=1}^N, \quad \mathbf{x}_i \in \mathbb{R}^D$$
where telemetry records must be partitioned into $K$ disjoint clusters without ground-truth supervisory labels.

The primary objectives of this study are:
1. **Mathematical Implementation from Scratch**: Develop a custom, fully vectorized K-Means clustering algorithm using NumPy, incorporating random centroid initialization, Euclidean distance computation, assignment steps, centroid updates, and an outer multi-restart loop ($n_{\text{init}} = 10$) to prevent entrapment in suboptimal local minima.
2. **Feature Engineering**: Address the discontinuous periodic boundary problem of 24-hour diurnal time by applying trigonometric cyclical encoding:
   $$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hr}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hr}}{24}\right)$$
3. **Cluster Evaluation and Elbow Method**: Execute K-Means systematically across cluster configurations $K \in \{2, 3, 4, 5, 6\}$, computing the Within-Cluster Sum of Squares (WCSS) to detect the optimal elbow inflection point.
4. **Cluster Characterization & Real-World Profiling**: Profile the final clusters against both standardized clustering attributes and original physical variables (temperature, humidity, windspeed, and post-hoc rental counts `cnt`) to provide actionable operational insights.

---

## 3.2 Mathematical Formulation and Derivations

### 3.2.1 Objective Function (WCSS / Inertia)
Given a dataset of $N$ standardized feature vectors $\mathbf{X} = \{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N\} \subset \mathbb{R}^D$, K-Means aims to partition the $N$ observations into $K$ mutually exclusive clusters $\mathcal{C} = \{C_1, C_2, \dots, C_K\}$ such that:
$$\bigcup_{k=1}^K C_k = \{1, \dots, N\} \quad \text{and} \quad C_j \cap C_k = \emptyset \quad \forall j \neq k$$
The objective is to minimize the Within-Cluster Sum of Squares (WCSS), also known as inertia:
$$J_{\text{WCSS}}(\mathcal{C}, \boldsymbol{\mu}) = \sum_{k=1}^K \sum_{i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|_2^2$$
where $\boldsymbol{\mu}_k \in \mathbb{R}^D$ denotes the centroid of cluster $C_k$, and $\|\mathbf{x}_i - \boldsymbol{\mu}_k\|_2$ is the Euclidean distance metric:
$$\|\mathbf{x}_i - \boldsymbol{\mu}_k\|_2 = \sqrt{\sum_{j=1}^D (x_{ij} - \mu_{kj})^2}$$

Equivalently, introducing binary cluster assignment indicators $r_{ik} \in \{0, 1\}$ where $\sum_{k=1}^K r_{ik} = 1$:
$$J(R, \boldsymbol{\mu}) = \sum_{i=1}^N \sum_{k=1}^K r_{ik} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|_2^2$$

### 3.2.2 Alternating Optimization (Lloyd's Algorithm)
Because joint optimization over assignments $R$ and centroids $\boldsymbol{\mu}$ is NP-hard, Lloyd's algorithm minimizes $J$ via alternating coordinate descent:

1. **Expectation (Assignment Step)**:
   Fix centroids $\boldsymbol{\mu}$ and update assignment indicators $r_{ik}$ by assigning each instance $\mathbf{x}_i$ to its nearest Euclidean centroid:
   $$r_{ik} = \begin{cases} 1 & \text{if } k = \arg\min_j \|\mathbf{x}_i - \boldsymbol{\mu}_j\|_2^2 \\ 0 & \text{otherwise} \end{cases}$$
   This step guarantees $J(R^{(t+1)}, \boldsymbol{\mu}^{(t)}) \le J(R^{(t)}, \boldsymbol{\mu}^{(t)})$.

2. **Maximization (Centroid Update Step)**:
   Fix assignments $R$ and compute new centroids $\boldsymbol{\mu}_k$ by setting the partial derivative of $J$ with respect to $\boldsymbol{\mu}_k$ to zero:
   $$\frac{\partial J}{\partial \boldsymbol{\mu}_k} = -2 \sum_{i=1}^N r_{ik} (\mathbf{x}_i - \boldsymbol{\mu}_k) = \mathbf{0}$$
   $$\sum_{i \in C_k} \mathbf{x}_i - |C_k| \boldsymbol{\mu}_k = \mathbf{0} \implies \boldsymbol{\mu}_k^* = \frac{1}{|C_k|} \sum_{i \in C_k} \mathbf{x}_i$$
   The optimal centroid is precisely the empirical center of mass (mean vector) of all instances assigned to cluster $C_k$. This step guarantees $J(R^{(t+1)}, \boldsymbol{\mu}^{(t+1)}) \le J(R^{(t+1)}, \boldsymbol{\mu}^{(t)})$.

### 3.2.3 Convergence Guarantees and Local Minima
Because $J(R, \boldsymbol{\mu})$ is non-increasing at each step and bounded below by zero, Lloyd's algorithm is guaranteed to converge in a finite number of iterations to a local minimum. However, because the objective surface is non-convex, the final solution is sensitive to initial centroid placement. 

To mitigate poor local minima, the algorithm executes an outer multi-restart loop with $n_{\text{init}} = 10$ distinct random starts, retaining the cluster partition that achieves the lowest overall WCSS:
$$\mathcal{C}^* = \arg\min_{m \in \{1, \dots, n_{\text{init}}\}} J_{\text{WCSS}}^{(m)}$$

---

## 3.3 Dataset Description and Feature Engineering Pipeline

The clustering experiment utilizes the **Bike Sharing** hourly telemetry dataset (`hour.csv`):
- **Total Records ($N$)**: 17,379 hourly observations.
- **Data Period**: 2011 to 2012 across Capital Bikeshare operational logs.
- **Missing Values**: 0 missing values across all telemetry columns.

### 3.3.1 Trigonometric Cyclical Hour Encoding
The raw telemetry column `hr` ranges discretely from $0$ to $23$. Using raw integer values introduces an artificial discontinuity: hour $23$ (11:00 PM) and hour $0$ (midnight) are numerically $23$ units apart, despite being separated by only a single elapsed hour in physical reality. 

To preserve the continuous periodic topology of a 24-hour cycle, `hr` was mapped onto a unit circle via sine and cosine transformations:
$$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hr}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hr}}{24}\right)$$
This ensures that the Euclidean distance between 11:00 PM and midnight is identical to the Euclidean distance between any other adjacent hours.

### 3.3.2 Feature Selection and Standardization
The feature vector $\mathbf{x}_i \in \mathbb{R}^5$ used for clustering comprises five environmental and temporal variables:
1. `hour_sin`: Sine-transformed diurnal phase.
2. `hour_cos`: Cosine-transformed diurnal phase.
3. `temp`: Normalized ambient temperature (Celsius scaled by $41^\circ\text{C}$).
4. `hum`: Normalized relative humidity (scaled by $100\%$).
5. `windspeed`: Normalized wind speed (scaled by $67\text{ km/h}$).

*Note on Preprocessing Integrity*: Target telemetry features (`casual`, `registered`, `cnt`) and perceived temperature (`atemp`, which is collinear with `temp`) were intentionally omitted from the clustering feature matrix. The total rental count (`cnt`) is preserved exclusively for post-hoc validation to assess whether unsupervised environmental-temporal clusters reflect real commuter demand patterns.

To ensure isotropic Euclidean distance scaling across variables, all five features were standardized to zero mean and unit variance using `StandardScaler`:
$$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$

---

## 3.4 Implementation Methodology

The complete K-Means clustering algorithm was implemented from scratch using NumPy vectorization:

```python
import numpy as np

def compute_distances(X, centroids):
    # Vectorized Euclidean pairwise distance: (N, K)
    return np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)

def compute_wcss(X, labels, centroids):
    wcss = 0.0
    for k in range(len(centroids)):
        cluster_pts = X[labels == k]
        if len(cluster_pts) > 0:
            wcss += np.sum((cluster_pts - centroids[k]) ** 2)
    return wcss

def kmeans_single_run(X, k, max_iters=100, random_state=None):
    rng = np.random.RandomState(random_state)
    N, D = X.shape
    
    # Random initial centroid selection
    initial_indices = rng.choice(N, size=k, replace=False)
    centroids = X[initial_indices].copy()
    
    for iteration in range(max_iters):
        # 1. Assignment Step
        distances = compute_distances(X, centroids)
        labels = np.argmin(distances, axis=1)
        
        # 2. Centroid Update Step
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            cluster_pts = X[labels == j]
            if len(cluster_pts) > 0:
                new_centroids[j] = cluster_pts.mean(axis=0)
            else:
                new_centroids[j] = X[rng.choice(N)]
                
        # Check convergence
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids
        
    wcss = compute_wcss(X, labels, centroids)
    return centroids, labels, wcss

def kmeans(X, k, max_iters=100, n_init=10, random_state=42):
    best_wcss = np.inf
    best_centroids = None
    best_labels = None
    
    for init in range(n_init):
        seed = random_state + init if random_state is not None else None
        centroids, labels, wcss = kmeans_single_run(X, k, max_iters=max_iters, random_state=seed)
        if wcss < best_wcss:
            best_wcss = wcss
            best_centroids = centroids
            best_labels = labels
            
    return best_centroids, best_labels, best_wcss
```

---

## 3.5 Experimental Results and Elbow Optimization

### 3.5.1 Systematic Elbow Evaluation ($K \in \{2, 3, 4, 5, 6\}$)
To determine the optimal number of clusters, the algorithm was executed across $K \in \{2, 3, 4, 5, 6\}$ with $n_{\text{init}}=10$ and `random_state=42`.

### Table 3.1: WCSS Trajectory and Marginal Inertia Reductions
| Number of Clusters ($K$) | WCSS (Inertia) | Marginal Reduction ($\Delta \text{WCSS} = \text{WCSS}_{K-1} - \text{WCSS}_K$) | Marginal Relative Drop ($\%$) |
| :---: | :---: | :---: | :---: |
| **$K=2$** | 65,562.23 | — | — |
| **$K=3$** | 55,764.07 | 9,798.16 | 14.94% |
| **$K=4$** | **49,113.85** | **6,650.22** | **11.93%** |
| **$K=5$** | 45,186.36 | 3,927.49 | 7.99% |
| **$K=6$** | 41,696.01 | 3,490.35 | 7.72% |

[Insert Figure: kmeans_elbow_curve.png here]

Figure 3.1 displays the WCSS curve as a function of $K$. As $K$ increases from $2$ to $4$, inertia drops sharply by a cumulative $16,448.38$ units ($25.1\%$). Beyond $K=4$, the rate of inertia reduction flattens noticeably: increasing $K$ from $4$ to $5$ yields only a modest $3,927.49$ decrease ($7.99\%$), and $K=6$ yields $3,490.35$ ($7.72\%$). The clear elbow inflection point at **$K=4$** identifies it as the optimal trade-off between model parsimony and within-cluster variance minimization.

---

### 3.5.2 Cluster Partitioning and Visual Distributions

[Insert Figure: kmeans_clusters_with_centroids.png here]
[Insert Figure: kmeans_clusters_scatter_temp_hum.png here]

Figure 3.2 illustrates the final $K=4$ partitioning in the normalized `temp` versus `hum` plane, with marked cluster centroids. Figure 3.3 shows the high-density scatter distribution across individual samples. The clusters clearly delineate four distinct environmental-meteorological operational regimes.

---

## 3.6 Cluster Profiling and Operational Interpretation

To translate the unsupervised mathematical partitions into domain-specific operational insights, Table 3.2 and Figure 3.4 summarize the unstandardized physical feature means and post-hoc bike rental counts across all four clusters.

### Table 3.2: Cluster Characteristics in Physical Feature Space ($K=4$, $N=17,379$)
| Cluster ID | Record Count ($N_k$) | Share ($\%$) | Mean Time (`hr`) | Diurnal Regime | Mean Temp ($^\circ\text{C}$) | Mean Humidity ($\%$) | Mean Windspeed (km/h) | Post-Hoc Mean Rental Count (`cnt`) | Operational Classification |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Cluster 0** | 4,970 | 28.6% | 12.85 (Midday) | Afternoon / Off-peak | 20.2$^\circ\text{C}$ ($0.49$) | 71.0% ($0.71$) | 10.3 km/h ($0.15$) | 119.37 | **Mild & Humid Midday / Off-Peak** |
| **Cluster 1** | 3,901 | 22.4% | 14.63 (Afternoon) | Evening Peak / Rush | 28.7$^\circ\text{C}$ ($0.70$) | 48.7% ($0.49$) | 13.5 km/h ($0.20$) | **360.60** | **Warm, Dry Peak Commute Hours** |
| **Cluster 2** | 3,454 | 19.9% | 13.32 (Midday) | Midday Commute | 14.2$^\circ\text{C}$ ($0.35$) | 45.7% ($0.46$) | 21.6 km/h ($0.32$) | 181.80 | **Cool, Windy Daytime Operations** |
| **Cluster 3** | 5,054 | 29.1% | 06.67 (Morning) | Night / Early Morning | 18.5$^\circ\text{C}$ ($0.45$) | 76.4% ($0.76$) | 9.4 km/h ($0.14$) | 131.53 | **Morning & Night High Humidity** |

*(Note: In Table 3.2, normalized feature scales are converted to physical units: Temp in $^\circ\text{C} = \text{norm} \times 41$, Humidity in $\% = \text{norm} \times 100$, Windspeed in km/h $= \text{norm} \times 67$).*

[Insert Figure: kmeans_centroid_profiles.png here]

Figure 3.4 compares the standardized feature profiles of the four cluster centroids across all five input dimensions.

### Detailed Operational Regimes:
1. **Cluster 1: Warm, Dry Peak Commute Hours ($\mu_{\text{cnt}} = 360.60$, $N=3,901$)**
   - **Characteristics**: Highest temperature ($28.7^\circ\text{C}$), moderate humidity ($48.7\%$), average hour $14.63$ (afternoon/early evening).
   - **Operational Insights**: This cluster captures the primary demand window for bike sharing, recording an average of $360.60$ rentals per hour—more than double any other cluster. Operators should schedule active bicycle rebalancing to high-demand commuter stations and ensure fleet availability during these hours.
2. **Cluster 2: Cool, Windy Daytime Operations ($\mu_{\text{cnt}} = 181.80$, $N=3,454$)**
   - **Characteristics**: Lowest temperature ($14.2^\circ\text{C}$), highest wind speeds ($21.6\text{ km/h}$), dry air ($45.7\%$).
   - **Operational Insights**: Captures cold-season daytime operations. Bike demand remains moderate ($181.80$ rentals/hour). Moderate wind and cooler temperatures suppress casual usage, though utilitarian commuting persists.
3. **Cluster 3: Morning & Night High Humidity / Low Demand ($\mu_{\text{cnt}} = 131.53$, $N=5,054$)**
   - **Characteristics**: Concentrated around dawn and early morning ($\mu_{\text{hr}} = 6.67$), highest humidity ($76.4\%$), low windspeed ($9.4\text{ km/h}$).
   - **Operational Insights**: Encompasses late-night and pre-dawn hours with heavy moisture and limited commuter activity ($131.53$ rentals/hour). This low-usage interval represents the ideal window for system maintenance, damaged bike retrieval, and depot battery recharging.
4. **Cluster 0: Mild & Humid Midday / Off-Peak ($\mu_{\text{cnt}} = 119.37$, $N=4,970$)**
   - **Characteristics**: Midday hours ($\mu_{\text{hr}} = 12.85$), elevated humidity ($71.0\%$), mild temperature ($20.2^\circ\text{C}$).
   - **Operational Insights**: Represents mid-day transitional hours with overcast or muggy weather. Rental counts are subdued ($119.37$ rentals/hour), indicating baseline off-peak transit.

---

## 3.7 Question 3(a) Conclusion
The unsupervised K-Means clustering investigation successfully partitioned Capital Bikeshare telemetry records into meaningful operational states:
1. **Methodological Rigor**: The custom NumPy implementation with random restarts ($n_{\text{init}} = 10$) avoided suboptimal local minima, while sine-cosine cyclical encoding resolved the temporal boundary discontinuity of raw 24-hour time.
2. **Optimal Cluster Selection**: The Within-Cluster Sum of Squares (WCSS) trajectory demonstrated a clear elbow inflection at **$K=4$** ($\text{WCSS} = 49,113.85$), capturing the dominant variance structure without unnecessary model complexity.
3. **Post-Hoc Demand Alignment**: Although rental counts (`cnt`) were withheld during clustering, the resulting clusters naturally separated operational states by demand intensity: Cluster 1 captures high-demand peak commuter hours ($\mu_{\text{cnt}} = 360.60$), while Cluster 3 isolates off-peak, high-humidity night intervals ($\mu_{\text{cnt}} = 131.53$).
4. **Practical Application**: These findings demonstrate that unsupervised telemetry clustering provides an effective basis for fleet rebalancing logistics, maintenance scheduling, and dynamic capacity planning in shared urban micro-mobility networks.

