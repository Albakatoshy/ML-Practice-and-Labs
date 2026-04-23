# 🚀 Machine Learning Practice and Labs

Welcome to my Machine Learning portfolio! This repository serves as a growing collection of my hands-on labs, university assignments, and personal projects. It demonstrates my practical experience with various ML algorithms, deep learning architectures, data preprocessing techniques, and model evaluation metrics.

---

## 📂 Repository Structure

Here is an overview of the current projects and labs in this repository:

    ML-Practice-and-Labs/
    │
    ├── README.md
    │
    ├── 01-KNN/
    │   ├── KNN_Predict_Diabetes/               # Predicting diabetes using K-Nearest Neighbors
    │   ├── KNN_Predict_Gamma_Rays/             # MAGIC Gamma Telescope dataset classification
    │   └── SimpleKnn/                          # Foundational KNN implementation and practice
    │
    ├── 02-Linear-Regression/
    │   ├── LinearRegression_CaliforniaHousePricePrediction/  # Predicting housing prices (Linear, Ridge, Lasso)
    │   └── LinearRegression_PredictCustomerSpend/            # Predicting E-commerce yearly spend
    │
    └── 03-Neural-Networks/
        └── MINST_DigitRecognition.ipynb        # Custom PyTorch CNN for Kaggle MNIST Classification

---

## 🧠 Current Modules & Projects

### [01. K-Nearest Neighbors (KNN)](./01-KNN)
This module explores classification tasks using the K-Nearest Neighbors algorithm.

* **Predicting Diabetes:** Building a model to classify patient health outcomes based on medical predictor variables.
* **Gamma Rays Classification:** Handling class imbalances, feature scaling, and hyperparameter tuning (finding the optimal $K$) to classify high-energy gamma particles vs. background hadrons.
* **Key Skills:** Train/Validation/Test splitting, Stratification, Feature Scaling (`StandardScaler`), Confusion Matrices, Precision, Recall, and F1-Scores.

### [02. Linear Regression](./02-Linear-Regression)
This module focuses on predicting continuous target variables and dealing with overfitting through regularization.

* **California House Price Prediction:** Predicting median house values using 1990 census data. Explores the differences between standard **Linear Regression**, **Ridge Regression (L2)**, and **Lasso Regression (L1)**.
* **Predict Customer Spend:** A business-focused project analyzing e-commerce engagement metrics (Time on App/Website, Length of Membership) to predict yearly customer spending.
* **Key Skills:** Exploratory Data Analysis (EDA) with Seaborn, Residual Analysis (Q-Q plots, distributions), Hyperparameter tuning (alpha penalties), Mean Absolute Error (MAE), and Mean Squared Error (MSE).

### [03. Neural Networks & Deep Learning](./03-Neural-Networks)
This module transitions into deep learning, focusing on computer vision and custom network architectures.

* **MNIST Digit Recognition:** Built a custom Convolutional Neural Network (CNN) from scratch to classify handwritten digits using Kaggle's flattened CSV dataset. Includes custom PyTorch `Dataset` classes to dynamically reshape 1D arrays into 2D spatial grids.
* **Model Optimization:** Addressed overfitting and stabilized the loss landscape by implementing **Dropout** and **Layer Normalization**, achieving >98% accuracy on the test set.
* **Key Skills:** PyTorch (`nn.Module`, `Conv2d`), Custom DataLoaders, Spatial Reshaping, Cross-Entropy Loss, SGD Optimization, and Deep Learning model evaluation.

---

## 🛠️ Technologies & Libraries Used

* **Language:** Python 3
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`
* **Deep Learning:** `torch` (PyTorch), `torchvision`
* **Data Visualization:** `matplotlib`, `seaborn`, `scipy`

---

## 💻 How to Run Locally

1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/Albakatoshy/ML-Practice-and-Labs.git](https://github.com/Albakatoshy/ML-Practice-and-Labs.git)
