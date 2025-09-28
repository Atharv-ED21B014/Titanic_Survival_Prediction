# 🚢 Titanic Survival Prediction with Machine Learning

> Predicting who survived the Titanic disaster using advanced machine learning techniques.

![Titanic](https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg)

---

## 📌 Overview

This project aims to predict the survival of passengers aboard the Titanic using machine learning techniques. We use the **Kaggle Titanic dataset** to explore data preprocessing, feature engineering, and model optimization. The final model—an optimized **XGBoost classifier**—achieves an impressive **88% accuracy**.

---

## 🧭 Table of Contents

- [📌 Overview](#-overview)
- [📊 Dataset Description](#-dataset-description)
- [⚙️ Installation](#️-installation)
- [🔍 Data Preprocessing](#-data-preprocessing)
- [📈 Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- [🧠 Feature Engineering](#-feature-engineering)
- [🛠️ Model Building](#️-model-building)
- [📊 Evaluation & Results](#-evaluation--results)
- [📌 Conclusion](#-conclusion)
- [🚀 Future Work](#-future-work)
- [📁 Project Structure](#-project-structure)
- [💬 Feedback](#-feedback)

---

## 📊 Dataset Description

The dataset contains information about Titanic passengers and their survival status.

**Files:**
- `train.csv`: Training data
- `test.csv`: Test data for final predictions
- `gender_submission.csv`: Sample submission format

**Key Features:**

| Feature       | Description                                                  |
|---------------|--------------------------------------------------------------|
| PassengerId   | Unique ID for each passenger                                 |
| Pclass        | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)                     |
| Name          | Passenger name                                               |
| Sex           | Gender                                                       |
| Age           | Age in years                                                 |
| SibSp         | Number of siblings/spouses aboard                           |
| Parch         | Number of parents/children aboard                           |
| Ticket        | Ticket number                                                |
| Fare          | Fare paid                                                    |
| Cabin         | Cabin number                                                 |
| Embarked      | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |
| Survived      | Target variable (0 = Not Survived, 1 = Survived)             |

---

## ⚙️ Installation

Install the required Python libraries using the following command:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

---

## 🔍 Data Preprocessing

**Missing values handled:**

- **Age**: Imputed with median
- **Embarked**: Filled with most frequent value
- **Fare**: Filled with median
- **Cabin**: Extracted the first letter; missing values filled with 'X'

**Dropped irrelevant columns:**

- Ticket, Name, and PassengerId (before training)

**Encodings and transformations:**

- Categorical features encoded using Label Encoding or One-Hot Encoding
- Numerical features scaled appropriately

---

## 📈 Exploratory Data Analysis (EDA)

Key insights derived from the data:

- **Gender**: Females had significantly higher survival rates than males
- **Class**: 1st class passengers had better chances of survival
- **Age**: Children were more likely to survive than adults
- **Family Size**: Medium-sized families had higher survival probabilities

Visualizations included bar charts, histograms, pie charts, and correlation heatmaps using matplotlib and seaborn.

---

## 🧠 Feature Engineering

New features were created to enhance model performance:

- **AgeGroup**: Age categorized into 'Child', 'Adult', 'Middle-aged', 'Senior'
- **FamilySize**: Sum of SibSp and Parch
- **CabinDeck**: Extracted first character of the Cabin value (A, B, ..., G, X)

These features helped capture underlying patterns in the dataset.

---

## 🛠️ Model Building

We tested several machine learning models using scikit-learn, and selected the best-performing one via grid search.

**Models Evaluated:**

- Random Forest
- Gradient Boosting
- AdaBoost
- XGBoost (best)

**XGBoost Best Parameters:**

- Learning Rate: 0.2
- Max Depth: 5
- Number of Estimators: 30

A full machine learning pipeline was created using Pipeline and ColumnTransformer to manage preprocessing and training seamlessly.

---

## 📊 Evaluation & Results

### ✅ Accuracy Scores:
- Cross-validation Accuracy: 85%
- Test Set Accuracy: 88%

### 📄 Classification Report:
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Not Survived (0) | 0.89 | 0.92 | 0.90 |
| Survived (1) | 0.85 | 0.80 | 0.82 |
| **Accuracy** | | | **0.88** |

### 📉 Confusion Matrix:
|  | Predicted: No | Predicted: Yes |
|---|---------------|----------------|
| **Actual: Not Survived** | 245 | 21 |
| **Actual: Survived** | 30 | 122 |

---

## 📌 Conclusion

- The XGBoost model demonstrated high predictive performance with 88% accuracy.
- Feature engineering and hyperparameter tuning significantly boosted results.
- The model was robust across different demographic and travel-related features.

---

## 🚀 Future Work

- Apply ensemble techniques (Stacking, Voting)
- Explore deep learning models using TensorFlow or PyTorch
- Incorporate more external data (e.g., passenger titles, ticket groups)
- Deploy the model using Streamlit or Flask for interactive prediction

---

## 📁 Project Structure

```
├── Titanic2.ipynb             # Jupyter Notebook with full ML pipeline
├── README.md                  # Project documentation
├── train.csv                  # Training dataset
├── test.csv                   # Test dataset for submission
├── gender_submission.csv      # Sample submission format
```

---

## 💬 Feedback

Feel free to contribute, ask questions, or open issues. Collaboration and suggestions are always welcome!