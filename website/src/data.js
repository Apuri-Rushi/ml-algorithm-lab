// All values are taken from the project's result files:
//   classification/classification/*.csv, best_hyperparameters.txt
//   regression/*.ipynb outputs (regression_model_comparison.csv; degree-4 train R2 corrected to 0.98898)
//   unsupervised/K-means/kmeans_elbow_results.csv and notebook outputs
//   unsupervised/PCA/*.csv, pca_summary.txt
// and were re-checked by re-running the code.

export const project = {
  institute: "Indian Institute of Information Technology Sri City",
  title: "Supervised & Unsupervised Machine Learning",
  subtitle: "Classification · Regression · K-Means Clustering · Principal Component Analysis",
  authors: [
    { name: "Apuri Rushi", id: "S20240010012" },
    { name: "Gargi Diware", id: "S20240010074" },
  ],
  section: "Section 2",
};

export const overview = [
  {
    id: "q1", q: "Question 1", task: "Classification", kind: "Supervised",
    dataset: "Chronic Kidney Disease (UCI)", size: "400 patients · 24 attributes",
    models: "KNN, Decision Tree, SVM (RBF), 1D CNN (bonus)",
    result: "ROC-AUC 1.000", note: "SVM selected (KNN–SVM tie broken on accuracy and F1)",
  },
  {
    id: "q2", q: "Question 2", task: "Regression", kind: "Supervised",
    dataset: "Graduate Admissions (Kaggle)", size: "400 applicants · 7 features",
    models: "Linear (gradient descent), Polynomial d = 1–4, Ridge, Lasso",
    result: "Test R² 0.825", note: "Degree-2 polynomial has the lowest test MSE",
  },
  {
    id: "q3", q: "Question 3", task: "Clustering", kind: "Unsupervised",
    dataset: "Bike Sharing, hourly (UCI)", size: "17,379 hours · 5 features",
    models: "K-Means implemented from scratch (NumPy)",
    result: "K = 4", note: "Elbow method; WCSS 49,113.85",
  },
  {
    id: "q4", q: "Question 4", task: "Dimensionality Reduction", kind: "Unsupervised",
    dataset: "Iris (UCI)", size: "150 flowers · 4 measurements",
    models: "PCA via covariance eigen-decomposition (manual)",
    result: "95.81 %", note: "Variance retained by two components",
  },
];

export const protocol = [
  "80 : 20 train–test split with random_state = 42; stratified by class for CKD.",
  "Imputation, encoding and scaling are fitted on training data only and refitted inside every cross-validation fold.",
  "Hyper-parameters are selected by 5-fold cross-validation, never by test-set scores.",
  "K-Means and PCA use the full dataset because they describe structure rather than predict held-out labels.",
];

/* ---------------------------- Question 1 ---------------------------- */
export const q1 = {
  dataset: [
    ["Records", "400 (250 CKD, 150 not-CKD)"],
    ["Inputs", "14 numerical, 10 categorical (34 features after one-hot encoding)"],
    ["Target", "class — CKD is the positive class (1)"],
    ["Split", "320 train (200 / 120) · 80 test (50 / 30), stratified"],
    ["Missing values", "Up to 38 % per column (rbc 152, rbcc 131, wbcc 106, pot 88)"],
  ],
  preprocessing: [
    ["Cleaning", "Whitespace stripped; “?” and empty tokens set to missing; invalid categories (appet = “no”, pe = “good”) set to missing."],
    ["Numerical", "Median imputation, then standardisation z = (x − μ_train) / σ_train."],
    ["Categorical", "Most-frequent imputation, then one-hot encoding (unknown categories ignored)."],
    ["Tuning", "GridSearchCV · 5-fold stratified CV · scoring = ROC-AUC."],
  ],
  models: [
    {
      name: "K-Nearest Neighbours",
      theory: "Classifies a patient by the share of CKD cases among its K nearest training patients. Small K memorises; large K smooths.",
      formula: "p̂(CKD | x) = (1/K) Σ I(yᵢ = 1)",
      grid: "K ∈ {3, 5, 7, 9, 11} × {Euclidean, Manhattan}",
      best: "K = 9, Manhattan", cvLabel: "CV ROC-AUC", cv: "0.9995",
    },
    {
      name: "Decision Tree",
      theory: "Recursively splits on the feature and threshold that most reduce Gini impurity. Depth and leaf size control complexity.",
      formula: "G = 1 − Σ p_c²",
      grid: "max_depth ∈ {2, 3, 5, 10, None} × min_samples_leaf ∈ {1, 2, 5, 10}",
      best: "max_depth = 3, min_samples_leaf = 5", cvLabel: "CV ROC-AUC", cv: "0.9770",
    },
    {
      name: "Support Vector Machine (RBF)",
      theory: "Finds a maximum-margin boundary in a kernel-induced space. C penalises margin violations; γ sets how local the kernel is.",
      formula: "K(x, v) = exp(−γ ‖x − v‖²)",
      grid: "C ∈ {0.1, 1, 10, 100} × γ ∈ {scale, 0.01, 0.1, 1}",
      best: "C = 0.1, γ = scale", cvLabel: "CV ROC-AUC", cv: "1.0000",
    },
    {
      name: "1D Convolutional Neural Network (bonus)",
      theory: "Treats the 34 encoded features as a 1-D sequence. Two Conv1D blocks with pooling and dropout feed a dense layer and a sigmoid output.",
      formula: "Conv16 → Pool → Conv32 → Pool → Dense16 → σ",
      grid: "Adam 1e-3 · batch 16 · ≤ 100 epochs · early stopping (patience 15) · 256 fit / 64 validation",
      best: "Lowest validation loss at epoch 97", cvLabel: "Parameters", cv: "5,761",
    },
  ],
  metrics: ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
  results: [
    { model: "KNN", Accuracy: 0.9625, Precision: 1.0, Recall: 0.94, F1: 0.9691, "ROC-AUC": 1.0, TN: 30, FP: 0, FN: 3, TP: 47 },
    { model: "Decision Tree", Accuracy: 0.975, Precision: 0.98, Recall: 0.98, F1: 0.98, "ROC-AUC": 0.9873, TN: 29, FP: 1, FN: 1, TP: 49 },
    { model: "SVM (RBF)", Accuracy: 0.975, Precision: 1.0, Recall: 0.96, F1: 0.9796, "ROC-AUC": 1.0, TN: 30, FP: 0, FN: 2, TP: 48 },
    { model: "1D CNN (bonus)", Accuracy: 0.975, Precision: 1.0, Recall: 0.96, F1: 0.9796, "ROC-AUC": 0.9947, TN: 30, FP: 0, FN: 2, TP: 48 },
  ],
  depth: [
    { depth: "2", train: 0.9812, test: 0.975 },
    { depth: "3", train: 0.9906, test: 0.975 },
    { depth: "5", train: 0.9969, test: 0.9875 },
    { depth: "10", train: 1.0, test: 0.9875 },
    { depth: "None", train: 1.0, test: 0.9875 },
  ],
  findings: [
    ["ROC-AUC (assignment criterion)", "KNN and SVM tie at 1.0000."],
    ["Tie-break", "SVM: higher accuracy and F1 than KNN, with zero false positives."],
    ["Best recall", "Decision Tree, 0.98 — only one CKD patient missed."],
    ["Caveat", "With 80 test records, one prediction changes accuracy by 1.25 points; a perfect AUC describes this test set, not future patients."],
  ],
};

/* ---------------------------- Question 2 ---------------------------- */
export const q2 = {
  dataset: [
    ["Records", "400 applicants (Admission_Predict.csv), no missing values"],
    ["Inputs", "GRE, TOEFL, University Rating, SOP, LOR, CGPA, Research"],
    ["Target", "Chance of Admit (0.34–0.97, mean ≈ 0.724)"],
    ["Split", "320 train · 80 test · StandardScaler fitted on train only"],
    ["Strongest correlations", "CGPA 0.873 · GRE 0.803 · TOEFL 0.792"],
  ],
  models: [
    {
      name: "Linear Regression — Gradient Descent (from scratch)",
      theory: "Batch gradient descent on standardised inputs, weights initialised at zero, learning rate 0.05 for 5,000 epochs. Training MSE falls from 0.5449 to 0.00387.",
      formula: "w ← w − η · (2/n) Xᵀ(Xw + b − y)",
      best: "Bias 0.7253 ≈ target mean; CGPA has the largest weight (0.067)",
    },
    {
      name: "Polynomial Regression (degree 1–4)",
      theory: "Expands the inputs into powers and interactions, then fits ordinary least squares. The model is still linear in its coefficients.",
      formula: "features = C(7 + d, d) − 1  →  7, 35, 119, 329",
      best: "Degree 2 generalises best; degree 4 has 329 columns for 320 rows",
    },
    {
      name: "Ridge Regression (L2)",
      theory: "Adds a squared penalty that shrinks every coefficient toward zero. α chosen from 20 values (10⁻⁴ to 10²) by 5-fold CV.",
      formula: "min ‖y − Xw − b‖² + α ‖w‖²",
      best: "α = 23.36 · CV MSE 0.004225",
    },
    {
      name: "Lasso Regression (L1)",
      theory: "Adds an absolute-value penalty that can set coefficients exactly to zero, performing feature selection.",
      formula: "min (1/2n) ‖y − Xw − b‖² + α Σ |wⱼ|",
      best: "α = 0.000886 · CV MSE 0.004256 · SOP set to 0",
    },
  ],
  results: [
    { model: "Linear (GD)", trainMSE: 0.00387, testMSE: 0.00462, testRMSE: 0.06795, trainR2: 0.79519, testR2: 0.82121 },
    { model: "Polynomial d = 2", trainMSE: 0.00346, testMSE: 0.00451, testRMSE: 0.06715, trainR2: 0.81704, testR2: 0.82538 },
    { model: "Polynomial d = 3", trainMSE: 0.00253, testMSE: 0.01069, testRMSE: 0.10338, trainR2: 0.86606, testR2: 0.58615 },
    { model: "Polynomial d = 4", trainMSE: 0.00021, testMSE: 0.87718, testRMSE: 0.93658, trainR2: 0.98898, testR2: -32.96851 },
    { model: "Ridge", trainMSE: 0.00391, testMSE: 0.0048, testRMSE: 0.0693, trainR2: 0.79306, testR2: 0.81401 },
    { model: "Lasso", trainMSE: 0.00387, testMSE: 0.00466, testRMSE: 0.06829, trainR2: 0.79513, testR2: 0.81943 },
  ],
  weights: [
    ["CGPA", 0.0674], ["GRE Score", 0.021], ["TOEFL Score", 0.0164], ["LOR", 0.0141],
    ["Research", 0.0111], ["University Rating", 0.0085], ["SOP", -0.0003],
  ],
  fit: [
    { label: "Underfit", model: "Ridge, α = 10⁶", trainR2: "0.002", testR2: "0.001", text: "The penalty crushes every slope; predictions are almost constant." },
    { label: "Good fit", model: "Polynomial d = 2", trainR2: "0.817", testR2: "0.825", text: "Low error on both training and test data." },
    { label: "Overfit", model: "Polynomial d = 4", trainR2: "0.989", testR2: "−32.97", text: "Near-perfect on training data; test MSE is about 4,177 × training MSE." },
  ],
  findings: [
    ["Best model", "Degree-2 polynomial: test MSE 0.00451, RMSE 0.06715, R² 0.82538."],
    ["Overfitting", "Training error keeps falling with degree while test error explodes after degree 2."],
    ["Regularisation", "Ridge and Lasso did not beat plain linear regression: with 7 features and 320 rows there is little to regularise."],
    ["Caveat", "The degree-2 gain over linear is only 0.00011 MSE, and the degree was compared on the test set."],
  ],
};

/* ---------------------------- Question 3 ---------------------------- */
export const q3 = {
  dataset: [
    ["Records", "17,379 hourly records (2011–2012), no missing values"],
    ["Features", "hour_sin, hour_cos, temp, hum, windspeed"],
    ["Cyclic hour", "sin(2π·hr/24), cos(2π·hr/24) — hour 23 sits next to hour 0"],
    ["Scaling", "Standardised with the sample standard deviation (n − 1)"],
    ["Excluded", "Rental counts (used only afterwards to describe clusters) and atemp (duplicates temp)"],
  ],
  steps: [
    ["Initialise", "Choose K random records as starting centroids."],
    ["Assign", "Give each record to its nearest centroid:  labelᵢ = argminₖ ‖xᵢ − μₖ‖²"],
    ["Update", "Move each centroid to the mean of its records:  μₖ = (1/|Cₖ|) Σ xᵢ"],
    ["Repeat", "Until centroids stop moving (np.allclose) or 100 iterations; empty clusters re-seeded."],
    ["Restart", "10 random initialisations; keep the run with the lowest WCSS."],
  ],
  elbow: [
    { k: 2, wcss: 65562.23, drop: null },
    { k: 3, wcss: 55764.07, drop: 14.9 },
    { k: 4, wcss: 49113.85, drop: 11.9 },
    { k: 5, wcss: 45186.36, drop: 8.0 },
    { k: 6, wcss: 41696.01, drop: 7.7 },
  ],
  clusters: [
    { id: 0, name: "Humid, mild", hours: 4970, share: 28.6, temp: 0.49, hum: 0.71, wind: 0.15, rentals: 119.4 },
    { id: 1, name: "Warm and dry", hours: 3901, share: 22.4, temp: 0.70, hum: 0.49, wind: 0.20, rentals: 360.6 },
    { id: 2, name: "Cool and windy", hours: 3454, share: 19.9, temp: 0.35, hum: 0.46, wind: 0.32, rentals: 181.8 },
    { id: 3, name: "Humid and calm", hours: 5054, share: 29.1, temp: 0.45, hum: 0.76, wind: 0.14, rentals: 131.5 },
  ],
  findings: [
    ["Chosen K", "K = 4 — the improvement in WCSS shrinks clearly after four clusters (6,650 → 3,927)."],
    ["Interpretation", "Warm, dry hours average about 3 × the rentals of humid hours, although demand was never an input."],
    ["Caveat", "Silhouette scores are about 0.2 for every K, so K = 4 is a practical segmentation rather than a proven natural grouping."],
  ],
};

/* ---------------------------- Question 4 ---------------------------- */
export const q4 = {
  dataset: [
    ["Records", "150 flowers, 3 species (labels used only to colour plots)"],
    ["Features", "Sepal length, sepal width, petal length, petal width (cm)"],
    ["Implementation", "NumPy only — no PCA estimator"],
  ],
  steps: [
    ["Standardise", "Centre and scale each feature so no unit dominates.", "Z = (X − μ) / σ"],
    ["Covariance", "Compute the 4 × 4 covariance matrix.", "S = ZᵀZ / (n − 1)"],
    ["Eigen-decompose", "Maximising uᵀSu with ‖u‖ = 1 gives an eigenvalue problem (np.linalg.eigh).", "Su = λu"],
    ["Sort and project", "Order components by eigenvalue and project onto the top k.", "Z_k = Z · W_k"],
  ],
  components: [
    { pc: "PC1", eig: 2.9381, ratio: 72.96, cum: 72.96 },
    { pc: "PC2", eig: 0.9202, ratio: 22.85, cum: 95.81 },
    { pc: "PC3", eig: 0.1477, ratio: 3.67, cum: 99.48 },
    { pc: "PC4", eig: 0.0209, ratio: 0.52, cum: 100.0 },
  ],
  eigenvectors: [
    ["Sepal length", -0.52107, 0.37742, 0.71957, 0.26129],
    ["Sepal width", 0.26935, 0.9233, -0.24438, -0.12351],
    ["Petal length", -0.58041, 0.02449, -0.14213, -0.80145],
    ["Petal width", -0.56486, 0.06694, -0.63427, 0.5236],
  ],
  findings: [
    ["Two components", "Keep 95.81 % of the standardised variance (150 × 2)."],
    ["Three components", "Keep 99.48 % (150 × 3)."],
    ["PC1", "Petal length, petal width and sepal length together — overall flower size."],
    ["PC2", "Dominated by sepal width (loading 0.923)."],
    ["Projection", "Setosa separates cleanly; versicolor and virginica overlap."],
  ],
};

/* ---------------------------- Summary ---------------------------- */
export const summary = {
  best: [
    { q: "Q1 Classification", model: "SVM (RBF)", metric: "Test ROC-AUC 1.000 · accuracy 97.5 %", alt: "Decision Tree for best recall (0.98)" },
    { q: "Q2 Regression", model: "Polynomial, degree 2", metric: "Test R² 0.825 · RMSE 0.067", alt: "Linear GD within 0.00011 MSE" },
    { q: "Q3 Clustering", model: "K-Means, K = 4", metric: "WCSS 49,113.85", alt: "Elbow method" },
    { q: "Q4 PCA", model: "2 principal components", metric: "95.81 % variance retained", alt: "3 components keep 99.48 %" },
  ],
  lessons: [
    ["More complex is not better", "Classical classifiers matched the CNN; polynomial degree 4 collapsed on unseen data."],
    ["Preprocessing matters", "Imputation, scaling and cyclic hour encoding shaped every result."],
    ["Match the metric to the question", "ROC-AUC, recall, MSE, WCSS and explained variance each answer something different."],
  ],
  limits: [
    "Supervised results rest on a single split with 80 test records; repeated or nested CV would be more reliable.",
    "The polynomial degree was compared on test data; a separate validation set would be cleaner.",
    "K-Means has no ground truth; stability and silhouette analysis would strengthen the choice of K.",
    "PCA retained variance was not tested inside a downstream classifier.",
  ],
};
